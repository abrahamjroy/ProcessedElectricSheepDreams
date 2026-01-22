import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from PIL import Image, ImageTk
import re
from backend import ImageGenerator

# Helper for Collapsible Frame
class ToggledFrame(ttk.Frame):
    def __init__(self, parent, text="", *args, **options):
        super().__init__(parent, *args, **options)
        self.show = tk.IntVar()
        self.show.set(0)
        
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(fill=X, expand=1)

        self.toggle_btn = ttk.Checkbutton(
            self.title_frame, 
            width=2, 
            text='+', 
            command=self.toggle,
            variable=self.show, 
            style='Toolbutton',
            bootstyle="secondary"
        )
        self.toggle_btn.pack(side=LEFT, padx=(0, 5))
        
        ttk.Label(self.title_frame, text=text, font=("Consolas", 11, "bold")).pack(side=LEFT, fill=X)

        self.sub_frame = ttk.Frame(self, padding=10)

    def toggle(self):
        if self.show.get():
            self.sub_frame.pack(fill=X, expand=1)
            self.toggle_btn.configure(text='-')
        else:
            self.sub_frame.forget()
            self.toggle_btn.configure(text='+')

class ZImageApp(ttk.Window):
    def __init__(self):
        # "cyborg" is a dark theme, we will customize further for AMOLED
        super().__init__(themename="cyborg") 
        
        self.title("Processed Electric Sheep Dreams")
        self.geometry("1600x1000")  # Larger to ensure viewport is visible
        self.minsize(1400, 900)  # Set minimum size
        
        # --- AMOLED Customizations ---
        # Force background to be pure black for key components
        style = ttk.Style()
        style.configure('.', background='#000000') # Global Black
        style.configure('TFrame', background='#000000')
        style.configure('TLabelframe', background='#000000') 
        style.configure('TLabelframe.Label', background='#000000', foreground='#a0a0a0')
        style.configure('TLabel', background='#000000', foreground='#e0e0e0')
        style.configure('TButton', font=("Consolas", 10, "bold"))
        style.configure('TNotebook', background='#000000')
        style.configure('TNotebook.Tab', background='#222222', foreground='#888888', font=("Consolas", 10))
        style.map('TNotebook.Tab', background=[('selected', '#444444')], foreground=[('selected', '#ffffff')])
        
        self.generator = None
        self.generated_image = None
        self.source_image = None # For Img2Img
        
        # Initialize status var early for threading
        self.status_var = tk.StringVar(value="[SYSTEM] Booting Neural Core...")
        
        self.create_widgets()
        
        # Start backend loading
        threading.Thread(target=self.init_backend, daemon=True).start()

    def init_backend(self):
        try:
            self.status_var.set("Initializing Neural Engine...")
            self.generator = ImageGenerator()
            self.status_var.set("System Ready. Waiting for input.")
            self.generate_btn.configure(state=NORMAL)
        except Exception as e:
            self.status_var.set(f"Initialization Failed: {e}")

    def create_widgets(self):
        # Main Layout
        main_pane = ttk.Panedwindow(self, orient=HORIZONTAL)
        main_pane.pack(fill=BOTH, expand=True, padx=0, pady=0) # Edge to edge
        
        # --- Sidebar / Controls (Left) ---
        # Wrap in ScrolledFrame for overflow support
        from ttkbootstrap.widgets.scrolled import ScrolledFrame
        
        # Intermediate container for PanedWindow compatibility
        sidebar_container = ttk.Frame(main_pane)
        main_pane.add(sidebar_container, weight=1)
        
        # FIXED FOOTER (Generatation Controls)
        footer_frame = ttk.Frame(sidebar_container, padding=20)
        footer_frame.pack(fill=X, side=BOTTOM)
        
        controls_frame = ScrolledFrame(sidebar_container, padding=(20, 20, 20, 0), autohide=True) 
        controls_frame.pack(fill=BOTH, expand=True)
        
        # Header
        header = ttk.Label(controls_frame, text="ELECTRIC SHEEP DREAMS", font=("OCR A Extended", 20, "bold"), foreground='#00ff00')
        header.pack(fill=X, pady=(10, 20))

        # Main Input (Shared)
        ttk.Label(controls_frame, text="CREATIVE VISION", font=("Consolas", 10, "bold"), foreground="#00cc00").pack(anchor="w")
        
        # Style Preset Selection (New)
        self.style_var = tk.StringVar(value="No Style Preset")
        style_frame = ttk.Frame(controls_frame)
        style_frame.pack(fill=X, pady=(5, 0))
        
        self.style_combo = ttk.Combobox(style_frame, textvariable=self.style_var, values=[
            "No Style Preset",
            "Style: Cinematic (Dramatic Lighting)",
            "Style: Anime/Manga (Vibrant 2D)",
            "Style: Digital Art (Polished)",
            "Style: Oil Painting (Textured)",
            "Style: Cyberpunk (Neon/Tech)",
            "Style: Vintage Photo (Film Grain)",
            "Style: 3D Render (Octane/Unreal)"
        ], state="readonly", bootstyle="dark", font=("Consolas", 9))
        self.style_combo.pack(fill=X)
        
        # Prompt input with scrollbar
        prompt_frame = ttk.Frame(controls_frame)
        prompt_frame.pack(fill=X, pady=(5, 20))
        
        prompt_scroll = ttk.Scrollbar(prompt_frame, orient=VERTICAL)
        prompt_scroll.pack(side=RIGHT, fill=Y)
        
        self.prompt_text = tk.Text(prompt_frame, height=8, wrap="word", font=("Consolas", 11), 
                                   bg="#050505", fg="#00ff00", borderwidth=1, relief="solid",
                                   yscrollcommand=prompt_scroll.set)
        self.prompt_text.pack(fill=X, side=LEFT, expand=True)
        prompt_scroll.config(command=self.prompt_text.yview)
        
        # TABS: Creation Mode vs Remix Mode
        self.notebook = ttk.Notebook(controls_frame, bootstyle="dark")
        self.notebook.pack(fill=X, pady=(0, 20))
        
        # -- Tab 1: Create (Txt2Img) --
        tab_create = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_create, text="  CREATE  ")
        
        # Aspect Ratio (For Create Mode)
        ttk.Label(tab_create, text="Form Factor", font=("Consolas", 9), foreground="#888888").pack(anchor="w", pady=(5,5))
        self.aspect_var = tk.StringVar(value="1:1 Square (1024x1024)")
        self.aspect_combo = ttk.Combobox(tab_create, textvariable=self.aspect_var, values=[
            "1:1 Square (1024x1024)",
            "16:9 Cinema (1344x768)",
            "9:16 Mobile (768x1344)", 
            "4:3 Classic (1152x896)",
            "3:4 Portrait (896x1152)",
            "Custom"
        ], state="readonly", bootstyle="dark", font=("Consolas", 10))
        self.aspect_combo.pack(fill=X)
        self.aspect_combo.bind("<<ComboboxSelected>>", self.update_dimensions)

        # -- Tab 2: Remix (Img2Img) --
        tab_remix = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_remix, text="  REMIX  ")
        
        # Upload Button
        ttk.Button(tab_remix, text="UPLOAD REFERENCE IMAGE", command=self.upload_source_image, bootstyle="secondary").pack(fill=X, pady=(0, 10))
        
        # Preview Thumb
        self.source_thumb_lbl = ttk.Label(tab_remix, text="No Image Selected", font=("Consolas", 9, "italic"), foreground="#666")
        self.source_thumb_lbl.pack(pady=5)

        # MASK Upload Button (New)
        ttk.Button(tab_remix, text="UPLOAD MASK (OPTIONAL)", command=self.upload_mask_image, bootstyle="secondary-outline").pack(fill=X, pady=(10, 5))
        self.mask_thumb_lbl = ttk.Label(tab_remix, text="No Mask (Changes Grid)", font=("Consolas", 9, "italic"), foreground="#666")
        self.mask_thumb_lbl.pack(pady=5)
        
        # Strength Slider
        ttk.Label(tab_remix, text="Creativity / Influence", font=("Consolas", 9), foreground="#888888").pack(anchor="w", pady=(10, 5))
        self.strength_var = tk.DoubleVar(value=0.40)
        self.strength_scale = ttk.Scale(tab_remix, from_=0.0, to=1.0, orient=HORIZONTAL, variable=self.strength_var, bootstyle="warning")
        self.strength_scale.pack(fill=X)
        self.strength_lbl = ttk.Label(tab_remix, text="0.40 (Balanced Edit)", font=("Consolas", 8), foreground="#666")
        self.strength_lbl.pack(anchor="e")
        self.strength_scale.bind("<Motion>", self.update_strength_lbl)
        
        # Inpainting Options (for mask mode)
        ttk.Label(tab_remix, text="Inpaint Options", font=("Consolas", 9), foreground="#888888").pack(anchor="w", pady=(15, 5))
        
        self.color_match_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_remix, text="Color Match (match lighting/tone)", variable=self.color_match_var, bootstyle="success-round-toggle").pack(anchor="w")
        
        self.blend_edges_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_remix, text="Blend Edges (feather transitions)", variable=self.blend_edges_var, bootstyle="success-round-toggle").pack(anchor="w")
        
        self.preserve_edges_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_remix, text="Preserve Structure (edge guidance)", variable=self.preserve_edges_var, bootstyle="success-round-toggle").pack(anchor="w")

        # Advanced Settings (Shared, Collapsible)
        adv_section = ToggledFrame(controls_frame, text="ADVANCED CONFIGURATION")
        adv_section.pack(fill=X, pady=10)
        
        # Content of Advanced Section
        adv_grid = adv_section.sub_frame
        
        # Negative Prompt
        ttk.Label(adv_grid, text="Exclusions (Negative)", font=("Consolas", 9), foreground="#888888").pack(fill=X, anchor="w")
        self.neg_presets = ttk.Combobox(adv_grid, values=[
            "None",
            "Preset: Photography Cleanup",
            "Preset: Illustration Cleanup",
            "Preset: NSFW Safety",
            "Preset: Artistic Enhancer (Anti-Realism)",
            "Preset: AIO (Anti-Digital/Realism)"
        ], state="readonly", bootstyle="dark")
        self.neg_presets.current(0)
        self.neg_presets.pack(fill=X, pady=(2, 5))
        self.neg_presets.bind("<<ComboboxSelected>>", self.apply_neg_preset)
        
        self.neg_prompt_text = ttk.Text(adv_grid, height=3, wrap="word", font=("Consolas", 10), bg="#050505", fg="#aaaaaa", borderwidth=1, relief="solid")
        self.neg_prompt_text.pack(fill=X, pady=(0, 15))
        
        # Sliders grid
        sliders_frame = ttk.Frame(adv_grid)
        sliders_frame.pack(fill=X)
        
        # Steps
        ttk.Label(sliders_frame, text="Sampling Steps", font=("Consolas", 9), foreground="#888888").grid(row=0, column=0, sticky="w")
        self.steps_var = tk.IntVar(value=9)
        self.steps_spin = ttk.Spinbox(sliders_frame, from_=1, to=50, textvariable=self.steps_var, bootstyle="secondary", width=5)
        self.steps_spin.grid(row=0, column=1, padx=10, sticky="e")
        
        # Guidance
        ttk.Label(sliders_frame, text="Prompt Adherence", font=("Consolas", 9), foreground="#888888").grid(row=1, column=0, sticky="w", pady=10)
        self.cfg_var = tk.DoubleVar(value=0.0)
        self.cfg_scale = ttk.Scale(sliders_frame, from_=0.0, to=10.0, orient=HORIZONTAL, variable=self.cfg_var, bootstyle="info")
        self.cfg_scale.grid(row=1, column=1, padx=10, sticky="ew")
        
        sliders_frame.columnconfigure(1, weight=1)
        
        # Width/Height Manual (Hidden unless custom usually, but helpful to see)
        dim_frame = ttk.Frame(adv_grid)
        dim_frame.pack(fill=X, pady=10)
        
        self.width_var = tk.IntVar(value=1024)
        self.height_var = tk.IntVar(value=1024)
        
        # Trace vars for aspect lock logic
        self.width_var.trace_add("write", lambda *args: self.on_dimension_change("w"))
        self.height_var.trace_add("write", lambda *args: self.on_dimension_change("h"))

        ttk.Label(dim_frame, text="W:", font=("Consolas", 9), foreground="#666").pack(side=LEFT)
        ttk.Spinbox(dim_frame, textvariable=self.width_var, from_=64, to=2048, increment=64, width=6).pack(side=LEFT, padx=5)
        
        ttk.Label(dim_frame, text="H:", font=("Consolas", 9), foreground="#666").pack(side=LEFT)
        ttk.Spinbox(dim_frame, textvariable=self.height_var, from_=64, to=2048, increment=64, width=6).pack(side=LEFT, padx=5)

        # Seed
        ttk.Label(adv_grid, text="Seed ID", font=("Consolas", 9), foreground="#888888").pack(anchor="w", pady=(10,0))
        
        seed_frame = ttk.Frame(adv_grid)
        seed_frame.pack(fill=X)
        
        self.seed_var = tk.IntVar(value=-1)
        ttk.Entry(seed_frame, textvariable=self.seed_var, bootstyle="dark").pack(side=LEFT, fill=X, expand=True)
        
        # Dice Button
        ttk.Button(seed_frame, text="🎲", width=3, command=self.roll_dice, bootstyle="secondary-outline").pack(side=LEFT, padx=(5,0))

        # Status Label
        self.status_lbl = ttk.Label(footer_frame, textvariable=self.status_var, wraplength=350, justify=CENTER, font=("Consolas", 9), foreground="#00aa00")
        self.status_lbl.pack(fill=X, pady=(0, 10))
        
        # Generate Button
        self.generate_btn = ttk.Button(footer_frame, text="⚡ GENERATE DREAM ⚡", command=self.start_generation, state=DISABLED, bootstyle="success")
        self.generate_btn.pack(fill=X, pady=(0, 5))
        
        # Action Row (Save + Upscale)
        action_row = ttk.Frame(footer_frame)
        action_row.pack(fill=X)
        
        self.save_btn = ttk.Button(action_row, text="💾 SAVE", command=self.save_image, state=DISABLED, bootstyle="secondary-outline")
        self.save_btn.pack(side=LEFT, fill=X, expand=True, padx=(0, 2))
        
        self.upscale_btn = ttk.Button(action_row, text="🔍 UPSCALE 2x", command=self.upscale_action, state=DISABLED, bootstyle="info-outline")
        self.upscale_btn.pack(side=LEFT, fill=X, expand=True, padx=(2, 0))

        # --- Viewport (Right) ---
        viewport_frame = ttk.Frame(main_pane) 
        main_pane.add(viewport_frame, weight=3)
        
        # Force the sash position to ensure both panes are visible
        self.after(100, lambda: main_pane.sashpos(0, 500))  # Set sidebar to 500px wide
        
        # Viewport background is pure black
        self.canvas_bg = ttk.Frame(viewport_frame)
        self.canvas_bg.pack(fill=BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_bg, bg="#000000", highlightthickness=0, borderwidth=0)
        self.canvas.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        # Floating Save Button
        self.save_btn = ttk.Button(viewport_frame, text="SAVE", command=self.save_image, state=DISABLED, bootstyle="light", width=15)
        self.save_btn.place(relx=0.95, rely=0.95, anchor="se")
        
        # Progress Bar Overlay (Thin line at top of viewport)
        self.progress = ttk.Progressbar(viewport_frame, mode='indeterminate', bootstyle="light", length=300)

    def on_dimension_change(self, which):
        # Prevent recursion loop
        if hasattr(self, '_updating_dims') and self._updating_dims: return
        
        ratio_str = self.aspect_var.get()
        
        # Determine target ratio
        target_ratio = None
        
        if "16:9" in ratio_str: target_ratio = 16/9
        elif "9:16" in ratio_str: target_ratio = 9/16
        elif "4:3" in ratio_str: target_ratio = 4/3
        elif "3:4" in ratio_str: target_ratio = 3/4
        elif "1:1" in ratio_str: target_ratio = 1.0
        elif "Custom" in ratio_str and hasattr(self, 'source_aspect_ratio') and self.source_aspect_ratio:
            # Use source image aspect ratio for Custom mode
            target_ratio = self.source_aspect_ratio
        else:
            return # No ratio to lock to
        
        self._updating_dims = True # Lock
        try:
            if which == "w":
                # User changed Width -> Update Height
                new_w = self.width_var.get()
                new_h = int(new_w / target_ratio)
                new_h = new_h - (new_h % 16) # Snap to 16 (ZImage requirement)
                if new_h != self.height_var.get():
                    self.height_var.set(new_h)
                    
            elif which == "h":
                # User changed Height -> Update Width
                new_h = self.height_var.get()
                new_w = int(new_h * target_ratio)
                new_w = new_w - (new_w % 16) # Snap to 16
                if new_w != self.width_var.get():
                    self.width_var.set(new_w)
        except Exception as e:
            print(f"Resize Error: {e}")
        finally:
            self._updating_dims = False # Unlock

    def update_dimensions(self, event=None):
        # When selecting a preset, we just set values. 
        # We don't need the trace logic to fight us.
        self._updating_dims = True 
        try:
            ratio = self.aspect_var.get()
            import re
            match = re.search(r"\((\d+)x(\d+)\)", ratio)
            if match:
                self.width_var.set(int(match.group(1)))
                self.height_var.set(int(match.group(2)))
        finally:
            self._updating_dims = False 


        # Seed
        ttk.Label(adv_grid, text="Seed ID", font=("Consolas", 9), foreground="#888888").pack(anchor="w", pady=(10,0))
        
        seed_frame = ttk.Frame(adv_grid)
        seed_frame.pack(fill=X)
        
        self.seed_var = tk.IntVar(value=-1)
        ttk.Entry(seed_frame, textvariable=self.seed_var, bootstyle="dark").pack(side=LEFT, fill=X, expand=True)
        
        # Dice Button
        ttk.Button(seed_frame, text="🎲", width=3, command=self.roll_dice, bootstyle="secondary-outline").pack(side=LEFT, padx=(5,0))

    def roll_dice(self):
        # "Animate" the rolling by changing numbers rapidly
        import random
        def roll_step(count):
            if count > 0:
                # Show random temporary number
                self.seed_var.set(random.randint(0, 9999999999))
                self.after(50, lambda: roll_step(count - 1))
            else:
                # Final result (or set to -1 for true random, but usually users want a lockable number)
                # Let's give them a concrete lockable number
                final_seed = random.randint(0, 2**32 - 1)
                self.seed_var.set(final_seed)
        
        roll_step(15) # 15 frames of animation

    def upload_source_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp")])
        if path:
            from PIL import ImageOps
            img = Image.open(path).convert("RGB")
            # Fix orientation based on EXIF data
            self.source_image = ImageOps.exif_transpose(img)
            
            # Show small thumb
            thumb = self.source_image.copy()
            thumb.thumbnail((200, 200))
            self.tk_thumb = ImageTk.PhotoImage(thumb)
            self.source_thumb_lbl.configure(image=self.tk_thumb, text="")
            
            # Auto-detect and apply source image dimensions/ratio
            src_w, src_h = self.source_image.size
            
            # Store the source aspect ratio for locking
            self.source_aspect_ratio = src_w / src_h
            
            # Calculate target dimensions (capped at max 1536 for VRAM safety)
            max_dim = 1536
            if max(src_w, src_h) > max_dim:
                scale = max_dim / max(src_w, src_h)
                src_w = int(src_w * scale)
                src_h = int(src_h * scale)
            
            # Snap to multiples of 16 (ZImage requirement)
            src_w = src_w - (src_w % 16)
            src_h = src_h - (src_h % 16)
            
            # Set aspect combo to "Custom" to allow free ratio from source
            self.aspect_var.set("Custom")
            
            # Update dimension spinboxes (disable trace temporarily)
            self._updating_dims = True
            self.width_var.set(src_w)
            self.height_var.set(src_h)
            self._updating_dims = False
            
            self.status_var.set(f"Source loaded: {self.source_image.size[0]}x{self.source_image.size[1]} → Output: {src_w}x{src_h}")

    def upload_mask_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp")])
        if path:
            from PIL import ImageOps
            img = Image.open(path).convert("RGB") # Keep RGB for now, backend can handle
            # Fix orientation based on EXIF data
            self.mask_image = ImageOps.exif_transpose(img)
            
            # Show small thumb
            thumb = self.mask_image.copy()
            thumb.thumbnail((200, 200))
            self.tk_mask_thumb = ImageTk.PhotoImage(thumb)
            self.mask_thumb_lbl.configure(image=self.tk_mask_thumb, text="Mask Ready")

    def update_strength_lbl(self, event=None):
        val = self.strength_var.get()
        txt = "Balanced"
        if val < 0.3: txt = "Subtle Change"
        elif val > 0.8: txt = "Heavy Transformation"
        self.strength_lbl.configure(text=f"{val:.2f} ({txt})")

    def apply_neg_preset(self, event=None):
        selection = self.neg_presets.get()
        presets = {
            "Preset: Photography Cleanup": "cartoon, illustration, painting, drawing, sketch, anime, 3d render, cgi, artwork, digital art, worst quality, low quality, blurry, pixelated, grainy, jpeg artifacts, deformed, disfigured, bad anatomy, bad hands, extra limbs, missing limbs, extra fingers, text, watermark, signature, cropped, out of frame",
            "Preset: Illustration Cleanup": "photorealistic, realistic, 3d, cgi, bad anatomy, bad hands, extra digits, missing fingers, worse quality, low quality, blurry, jpeg artifacts, compression artifacts, watermark, text, error, signature, username, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
            "Preset: NSFW Safety": "nsfw, nude, naked, sexual, gory, violence, blood, injuries",
            "Preset: Artistic Enhancer (Anti-Realism)": "photorealistic, realistic, 3d render, cgi, 8k, high definition, photography, photo, camera, lens, raw photo, digital art, shiny, glossy, plastic, octane render, unreal engine, smooth, polished, perfectly detailed, sharp focus, hdr, hyperrealistic",
            "Preset: AIO (Anti-Digital/Realism)": "动漫风, 二次元, 漫画风, 插画风, 卡通风, Q版, 手绘风, 水彩画, 素描风, 线稿, 草图, 写实油画, 油画风, 版画风, 像素画, pixel art, 点阵画, low poly, voxel, blueprint, 线框图, 3D渲染, 3D模型, CG渲染, CG风格, 游戏模型, 游戏角色, 虚拟形象, vtuber风, VTuber风, cyberpunk, 赛博朋克风, vaporwave, synthwave, glitch art, 赛博风, 霓虹赛博朋克风, 低清晰度, 低分辨率, 模糊, 虚焦, 对焦失败, 失焦, 失真, 噪点严重, 过度噪点, JPEG伪影, 压缩伪影, 过度压缩, 拉丝伪影, 色彩溢出, 颜色断层, 偏色严重, 过度锐化, 过度降噪, 过度HDR, HDR风, 光晕, 爆边, 过曝高光, 死黑阴影, 轮廓发光, 边缘发光, 锯齿, 粗糙细节, 光影不真实, 不真实反射, 不真实光影, 网红脸, AI网红脸, 网红模板脸, 模板脸, 默认人脸模板, 默认风格人脸, 千人一面, 千篇一律的脸, 同一张脸, 统一脸型, 统一五官, 标准化脸, 完美对称脸, 黄金比例脸, 完美无瑕的脸, 硬凹精致脸, 假精致脸, 统一瓜子脸, 统一尖下巴, 统一高鼻梁, 统一双眼皮, 娃娃脸, Barbie脸, 假娃娃脸, 过度少女感脸, 不自然幼态脸, 统一女神脸, 神仙颜值模板, 美颜滤镜, 过度磨皮, 磨皮过度, 磨皮滤镜, 磨皮皮肤, 玻璃皮, 玻璃皮肤, 瓷娃娃皮肤, “完美皮肤”, 过度美白, 过曝高光在皮肤上, 失真皮肤, 不真实皮肤纹理, 虚假皮肤纹理, 塑料质感皮肤, 蜡像脸, 假脸, 假皮肤, 过度修图, 过度液化, 液化变形, 修图痕迹, 过度瘦脸, 过度尖脸, 过度大眼, 不真实五官比例, 不真实头身比, 不自然身体比例, PS痕迹明显, 过度滤镜, 影楼风, 写真棚风, 写真棚打光, 影楼精修, “精修大片”, 棚拍大片, 棚拍大片风, 杂志封面风, 时尚杂志棚拍风, glamour, idol poster, idol promo, KOL头像, KOL风, 主播脸, 直播脸, 直播间滤镜, 广告硬照, 强烈商业广告感, 商业图库模板风, 过度时尚大片感, 夸张棚拍感, 过度高级感, 统一海报风, 通用广告模特感, 自拍风, 自拍感, 自拍杆视角, 手机前置摄像头, 过近广角畸变, 大头畸变, 鱼眼畸变, 超广角畸变, 高举手机俯拍, 低角度仰拍夸张畸变, 直播滤镜, 自拍滤镜, 美颜相机, 网红自拍, 自拍美颜, 抖音滤镜, 快手滤镜, 社交平台网红滤镜, 统一网红自拍模板, stock photo, 库存照片感, 典型stock photo, 通用图库模特, 商业图库风, 千篇一律图库模特, 过于刻意的摆拍, 僵硬姿势, 僵硬表情, 塑料笑容, 假笑, 虚假的表情, 僵尸脸, 僵硬的眼神, 过度摆拍姿势, 统一姿势, 重复姿势, 统一构图, 广告模板, 通用海报背景, 通用广告背景, template background, AI感很强, 一眼看出是AI图, 人工痕迹, 不自然, 虚假背景, 假景深, 过度景深虚化, 背景乱糟糟, 低质量, 低细节, 草率细节, 不真实, 非照片, 非摄影, 非自然光, 假光源, 不自然高光, 过度锐化线条, 轮廓过硬, 边缘过硬, 轮廓不干净, 噪点块状感, 模拟风格而不是实际照片"
        }
        if selection in presets:
            current = self.neg_prompt_text.get("1.0", tk.END).strip()
            add_text = presets[selection]
            # Avoid duplicate append
            if add_text not in current:
                new_text = f"{current}, {add_text}" if current else add_text
                self.neg_prompt_text.delete("1.0", tk.END)
                self.neg_prompt_text.insert("1.0", new_text)

    def update_dimensions(self, event=None):
        ratio = self.aspect_var.get()
        match = re.search(r"\((\d+)x(\d+)\)", ratio)
        if match:
            self.width_var.set(int(match.group(1)))
            self.height_var.set(int(match.group(2)))
            
        
        # Toggle for upscaling (New control)
        # We can add this to the advanced section or right next to Generate
        
    def start_generation(self):
        if not self.generator: return
        self.generate_btn.configure(state=DISABLED)
        
        # Show specific progress
        self.progress.place(relx=0, rely=0, relwidth=1) 
        self.progress.start(15)
        
        self.status_var.set("Processing Request...")
        
        # Determine Mode based on Active Tab
        current_tab_index = self.notebook.index(self.notebook.select())
        
        img_input = None
        mask_input = None
        strength = 0.0
        
        if current_tab_index == 1: # Remix Mode
            if self.source_image:
                # [NEW] Use Smart Resize on Input
                img_input = self.generator.smart_resize(self.source_image)
                strength = self.strength_var.get()
                
                # Check for mask (inpainting mode)
                if hasattr(self, 'mask_image') and self.mask_image:
                    mask_input = self.mask_image
                    self.status_var.set("Inpainting with Mask...")
                else:
                    self.status_var.set("Remixing Visual Reference...")
            else:
                if messagebox.askyesno("No Image", "No reference image uploaded. Switch to Create Mode?"):
                    self.notebook.select(0)
                    self.reset_ui()
                    return
                else:
                    self.reset_ui()
                    return

        params = {
            "prompt": self.prompt_text.get("1.0", tk.END).strip(),
            "negative_prompt": self.neg_prompt_text.get("1.0", tk.END).strip(),
            "width": self.width_var.get(),
            "height": self.height_var.get(),
            "steps": self.steps_var.get(),
            "guidance_scale": self.cfg_var.get(),
            "seed": self.seed_var.get(),
            "image": img_input,
            "mask_image": mask_input,
            "strength": strength,
            "color_match": self.color_match_var.get() if hasattr(self, 'color_match_var') else False,
            "blend_edges": self.blend_edges_var.get() if hasattr(self, 'blend_edges_var') else False,
            "preserve_edges": self.preserve_edges_var.get() if hasattr(self, 'preserve_edges_var') else False
        }
        
        if hasattr(self, 'style_var'):
            style = self.style_var.get()
            if style != "No Style Preset":
                styles_map = {
                    "Style: Cinematic (Dramatic Lighting)": "cinematic shot, dramatic lighting, movie scene, 8k, highly detailed, color graded",
                    "Style: Anime/Manga (Vibrant 2D)": "anime style, manga style, vibrant colors, studio ghibli, makoto shinkai, 2d, illustration",
                    "Style: Digital Art (Polished)": "digital art, concept art, trending on artstation, highly detailed, sharp focus, smooth",
                    "Style: Oil Painting (Textured)": "oil painting, thick brushstrokes, canvas texture, impressionist, traditional art",
                    "Style: Cyberpunk (Neon/Tech)": "cyberpunk, neon lights, futuristic, sci-fi, high tech, dark atmosphere, glowing",
                    "Style: Vintage Photo (Film Grain)": "vintage photograph, film grain, analog style, polaroid, faded colors, retro",
                    "Style: 3D Render (Octane/Unreal)": "3d render, octane render, unreal engine 5, ray tracing, physically based rendering"
                }
                if style in styles_map:
                    params["prompt"] = f"{params['prompt']}, {styles_map[style]}"

        threading.Thread(target=self.run_generation, args=(params,), daemon=True).start()
        
    def run_generation(self, params):
        try:
            image = self.generator.generate(**params)
            self.generated_image = image
            self.after(0, self.display_image, image)
            self.after(0, lambda: self.status_var.set("Rendering Complete."))
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: self.status_var.set(f"Error: {msg}"))
            print(f"Gen Error: {e}")
        finally:
            self.after(0, self.reset_ui)

    def reset_ui(self):
        self.progress.stop()
        self.progress.place_forget()
        self.generate_btn.configure(state=NORMAL)

    def display_image(self, image):
        c_width, c_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if c_width <= 1: c_width, c_height = 800, 800
        
        ratio = min(c_width / image.width, c_height / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        self.tk_image = ImageTk.PhotoImage(image.resize(new_size, Image.LANCZOS))
        
        self.canvas.delete("all")
        self.canvas.create_image(c_width//2, c_height//2, image=self.tk_image, anchor=CENTER)
        self.save_btn.configure(state=NORMAL)
        if hasattr(self, 'upscale_btn'):
            self.upscale_btn.configure(state=NORMAL)

    def save_image(self):
        if self.generated_image:
            # Generate default filename from prompt
            raw_prompt = self.prompt_text.get("1.0", tk.END).strip()
            # Extract first 2 words, sanitize
            words = re.findall(r'\w+', raw_prompt)[:2]
            filename = "_".join(words) if words else "generated_image"
            
            path = filedialog.asksaveasfilename(
                defaultextension=".png", 
                filetypes=[("PNG", "*.png")],
                initialfile=f"{filename}.png"
            )
            if path:
                self.generated_image.save(path)
                messagebox.showinfo("Saved", f"Image saved to {path}")

    def upscale_action(self):
        if not self.generated_image: return
        self.upscale_btn.configure(state=DISABLED)
        self.status_var.set("Upscaling Image (2x)... Please wait.")
        
        def run_upscale():
            try:
                upscaled = self.generator.upscale_image(self.generated_image)
                
                # Update UI in main thread
                def update_ui():
                    self.generated_image = upscaled
                    self.display_image(self.generated_image)
                    self.status_var.set("Upscale Complete!")
                    self.upscale_btn.configure(state=NORMAL)
                    
                self.after(0, update_ui)
                
            except Exception as e:
                def show_error():
                    self.status_var.set(f"Upscale Error: {e}")
                    messagebox.showerror("Error", str(e))
                    self.upscale_btn.configure(state=NORMAL)
                self.after(0, show_error)
        
        threading.Thread(target=run_upscale, daemon=True).start()

    def upscale_action(self):
        if not self.generated_image: return
        self.upscale_btn.configure(state=DISABLED)
        self.status_var.set("Upscaling Image (2x)... Please wait.")
        
        def run_upscale():
            try:
                upscaled = self.generator.upscale_image(self.generated_image)
                
                # Update UI in main thread
                def update_ui():
                    self.generated_image = upscaled
                    self.display_image(self.generated_image)
                    self.status_var.set("Upscale Complete!")
                    self.upscale_btn.configure(state=NORMAL)
                    
                self.after(0, update_ui)
                
            except Exception as e:
                def show_error():
                    self.status_var.set(f"Upscale Error: {e}")
                    messagebox.showerror("Error", str(e))
                    self.upscale_btn.configure(state=NORMAL)
                self.after(0, show_error)
        
        threading.Thread(target=run_upscale, daemon=True).start()

if __name__ == "__main__":
    app = ZImageApp()
    app.mainloop()
