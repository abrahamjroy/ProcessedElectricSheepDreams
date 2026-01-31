import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from PIL import Image, ImageTk, ImageGrab
import re
import os
import shutil
from backend import ImageGenerator
from config_manager import ConfigManager
from ui_widgets import (KeyboardShortcutsDialog, PromptHistoryPanel, 
                        GenerationHistoryPanel, ComparisonSlider)

# Try to import drag-and-drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False
    print("[INFO] tkinterdnd2 not available. Drag-and-drop disabled.")

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
            
# --- Custom KITT Scanner Widget ---
class KITTScroller(tk.Canvas):
    def __init__(self, parent, height=6, bg="#000000", **kwargs):
        super().__init__(parent, height=height, bg=bg, highlightthickness=0, **kwargs)
        self.width = 0
        self.cells = []
        self.num_cells = 40
        self.cell_width = 0
        self.head_pos = 0
        self.direction = 1
        self.running = False
        self.delay = 35 # ms per frame (approx 30fps)
        
        # Colors: Bright Red -> Med -> Dark -> Black
        self.colors = ["#ff0000", "#cc0000", "#990000", "#660000", "#330000", "#000000"]
        
        self.bind("<Configure>", self.on_resize)
        
    def on_resize(self, event):
        self.width = event.width
        self.cell_width = self.width / self.num_cells
        self.init_cells()
        
    def init_cells(self):
        self.delete("all")
        self.cells = []
        h = int(self["height"])
        
        for i in range(self.num_cells):
            x1 = i * self.cell_width
            x2 = x1 + self.cell_width - 1 # 1px gap
            rect = self.create_rectangle(x1, 0, x2, h, fill="#000000", outline="")
            self.cells.append(rect)
            
    def start(self):
        if not self.running:
            self.running = True
            self.animate()
            
    def stop(self):
        self.running = False
        # Clear lights
        for cell in self.cells:
            self.itemconfig(cell, fill="#000000")
            
    def animate(self):
        if not self.running: return
        
        # Fade Logic:
        # Instead of strict trail, we calculate distance from head
        # This allows for a perfect "comet" tail
        
        for i, cell in enumerate(self.cells):
            dist = abs(self.head_pos - i)
            # Leading edge sharp, trailing edge fade
            # But simpler: just render head and N neighbors with fading colors
            
            color = "#000000"
            if i == self.head_pos:
                color = self.colors[0]
            elif i == self.head_pos - self.direction: # Immediate trailing
                color = self.colors[1]
            elif i == self.head_pos - (2 * self.direction):
                color = self.colors[2]
            elif i == self.head_pos - (3 * self.direction):
                color = self.colors[3]
            elif i == self.head_pos - (4 * self.direction):
                color = self.colors[4]
                
            # Keep previous colors (fade effect) - actually simple cycle is safer
            # Let's just redraw based on checking array indices to avoid flicker
            
            self.itemconfig(cell, fill=color)
            
        # Move head
        self.head_pos += self.direction
        
        # bounce
        if self.head_pos >= self.num_cells - 1:
            self.direction = -1
        elif self.head_pos <= 0:
            self.direction = 1
            
        self.after(self.delay, self.animate)


class ZImageApp(ttk.Window):
    def __init__(self, stealth_mode=False):
        # "cyborg" is a dark theme, we will customize further for AMOLED
        super().__init__(themename="cyborg") 
        self.stealth_mode = stealth_mode 
        self.gallery_images = [] # List of (image_obj, prompt_str)
        
        # Initialize config manager
        self.config = ConfigManager()
        self.current_theme = self.config.get_setting("theme", "dark")
        
        self.title("Processed Electric Sheep Dreams")
        self.geometry("1600x1000")  # Larger to ensure viewport is visible
        self.minsize(1400, 900)  # Set minimum size
        
        # --- AMOLED Customizations ---
        # Force background to be pure black for key components
        self.apply_theme(self.current_theme)
        
        self.generator = None
        self.generated_image = None
        self.source_image = None # For Img2Img
        self.comparison_before = None  # For before/after comparison
        
        # Initialize status var early for threading
        self.status_var = tk.StringVar(value="[SYSTEM] Booting Neural Core...")
        
        self.create_widgets()
        self.setup_keyboard_shortcuts()
        
        # Start backend loading
        threading.Thread(target=self.init_backend, daemon=True).start()

    def init_backend(self):
        try:
            self.status_var.set("Initializing Neural Engine...")
            model_id = self.model_var.get()
            self.generator = ImageGenerator(model_id=model_id)
            self.status_var.set("System Ready. Waiting for input.")
            self.generate_btn.configure(state=NORMAL)
        except ImportError as e:
            error_msg = f"Missing dependency: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror(
                "Dependency Error",
                f"{error_msg}\n\nPlease install required packages:\n"
                "pip install transformers diffusers accelerate"
            )
        except FileNotFoundError as e:
            error_msg = "Model files not found"
            self.status_var.set(error_msg)
            messagebox.showerror(
                "Model Not Found",
                f"Cannot find model files.\n\n{str(e)}\n\n"
                "The model will download automatically on first use,\n"
                "but requires an internet connection."
            )
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                error_msg = "GPU Memory Error"
                self.status_var.set(error_msg)
                messagebox.showerror(
                    "VRAM Error",
                    f"Insufficient GPU memory.\n\n{str(e)}\n\n"
                    "Try closing other applications or reducing image dimensions."
                )
            else:
                error_msg = f"Runtime error: {str(e)}"
                self.status_var.set(error_msg)
                messagebox.showerror("Initialization Error", error_msg)
        except Exception as e:
            error_msg = f"Initialization Failed: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror(
                "Initialization Error",
                f"{error_msg}\n\nCheck console for details."
            )
            print(f"Detailed error: {e}")
            import traceback
            traceback.print_exc()
    
    def apply_theme(self, theme: str):
        """Apply color theme to the application."""
        style = ttk.Style()
        
        if theme == "light":
            # Light theme colors
            style.configure('.', background='#f0f0f0')
            style.configure('TFrame', background='#f0f0f0')
            style.configure('TLabel', background='#f0f0f0', foreground='#222222')
            style.configure('TLabelframe', background='#f0f0f0')
            style.configure('TLabelframe.Label', background='#f0f0f0', foreground='#444444')
            style.configure('TNotebook', background='#f0f0f0')
            style.configure('TNotebook.Tab', background='#d0d0d0', foreground='#444444')
            style.map('TNotebook.Tab', background=[('selected', '#ffffff')], foreground=[('selected', '#000000')])
        else:
            # Dark theme (AMOLED black)
            style.configure('.', background='#000000')
            style.configure('TFrame', background='#000000')
            style.configure('TLabelframe', background='#000000')
            style.configure('TLabelframe.Label', background='#000000', foreground='#a0a0a0')
            style.configure('TLabel', background='#000000', foreground='#e0e0e0')
            style.configure('TButton', font=("Consolas", 10, "bold"))
            style.configure('TNotebook', background='#000000')
            style.configure('TNotebook.Tab', background='#222222', foreground='#888888', font=("Consolas", 10))
            style.map('TNotebook.Tab', background=[('selected', '#444444')], foreground=[('selected', '#ffffff')])
        
        self.current_theme = theme
        self.config.set_setting("theme", theme)
    
    def toggle_theme(self):
        """Toggle between dark and light themes."""
        new_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme(new_theme)
        
        # Update theme button text
        if hasattr(self, 'theme_btn'):
            icon = "🌙" if new_theme == "dark" else "☀️"
            self.theme_btn.configure(text=icon)
    
    def setup_keyboard_shortcuts(self):
        """Setup global keyboard shortcuts."""
        # Generation
        self.bind("<Return>", lambda e: self.start_generation() if self.prompt_text.focus_get() == self.prompt_text else None)
        self.bind("<Escape>", lambda e: self.reset_ui())
        self.bind("<Control-g>", lambda e: self.start_generation())
        
        # File operations
        self.bind("<Control-s>", lambda e: self.save_image())
        self.bind("<Control-Shift-S>", lambda e: self.save_image())  # Same for now
        self.bind("<Control-o>", lambda e: self.open_output_folder())
        
        # Navigation
        self.bind("<Tab>", lambda e: self._cycle_tabs())
        self.bind("<Control-h>", lambda e: self.toggle_history_panel())
        self.bind("<Control-Shift-G>", lambda e: self.toggle_gallery())
        
        # Editing
        self.bind("<Control-v>", lambda e: self.paste_from_clipboard())
        
        # View
        self.bind("<F11>", lambda e: self.toggle_fullscreen())
        
        # Help (press Shift+/ which is ?)
        self.bind("<question>", lambda e: self.show_shortcuts())
        self.bind("<Shift-slash>", lambda e: self.show_shortcuts())
        
        # Advanced
        self.bind("<Control-e>", lambda e: self.export_workspace())
        self.bind("<Control-i>", lambda e: self.import_workspace())
    
    def show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        KeyboardShortcutsDialog(self)
    
    def _cycle_tabs(self):
        """Cycle through notebook tabs."""
        current = self.notebook.index(self.notebook.select())
        next_tab = (current + 1) % self.notebook.index("end")
        self.notebook.select(next_tab)
        return "break"  # Prevent default tab behavior
    
    def toggle_history_panel(self):
        """Toggle history panel visibility."""
        if hasattr(self, 'history_panel_frame'):
            if self.history_panel_frame.winfo_viewable():
                self.history_panel_frame.pack_forget()
            else:
                self.history_panel_frame.pack(fill=Y, side=LEFT, before=self.sidebar_container)
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))
    
    def paste_from_clipboard(self):
        """Paste image from clipboard."""
        try:
            img = ImageGrab.grabclipboard()
            if img and isinstance(img, Image.Image):
                self.source_image = img
                # Update thumbnail
                thumb = img.copy()
                thumb.thumbnail((200, 200))
                self.tk_thumb = ImageTk.PhotoImage(thumb)
                if hasattr(self, 'source_thumb_lbl'):
                    self.source_thumb_lbl.configure(image=self.tk_thumb, text="")
                self.status_var.set("Image pasted from clipboard")
                # Switch to remix tab
                self.notebook.select(1)
            else:
                self.status_var.set("No image in clipboard")
        except Exception as e:
            self.status_var.set(f"Paste error: {e}")
    
    def export_workspace(self):
        """Export workspace settings."""
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="workspace.json"
        )
        if path:
            if self.config.export_workspace(path):
                messagebox.showinfo("Export Success", f"Workspace exported to {path}")
            else:
                messagebox.showerror("Export Failed", "Could not export workspace")
    
    def import_workspace(self):
        """Import workspace settings."""
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json")]
        )
        if path:
            merge = messagebox.askyesno("Import Mode", 
                                       "Merge with existing settings?\n(No = Replace all)")
            if self.config.import_workspace(path, merge=merge):
                messagebox.showinfo("Import Success", 
                                  "Workspace imported. Please restart the application.")
            else:
                messagebox.showerror("Import Failed", "Could not import workspace")

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
        
        # HEADER (Fixed at top, outside scrolled area)
        header_frame = ttk.Frame(sidebar_container, padding=(20, 10))
        header_frame.pack(fill=X, side=TOP)
        
        # Use grid for better control - title on left, buttons on right
        header_frame.grid_columnconfigure(0, weight=0)  # Title column - don't expand
        header_frame.grid_columnconfigure(1, weight=1)  # Spacer - expands
        header_frame.grid_columnconfigure(2, weight=0)  # Buttons column - don't expand
        
        # Title with stealth mode indicator
        title_text = "ELECTRIC SHEEP DREAMS"
        if self.stealth_mode:
            title_text = "🔒 TOP SECRET MODE 🔒"
        
        header = ttk.Label(header_frame, text=title_text, 
                          font=("OCR A Extended", 14, "bold"),  # Smaller font
                          foreground='#ff0000' if self.stealth_mode else '#00ff00')
        header.grid(row=0, column=0, sticky='w')
        
        # Utility buttons row (always on the right)
        utils_frame = ttk.Frame(header_frame)
        utils_frame.grid(row=0, column=2, sticky='e')
        
        # Theme toggle
        theme_icon = "🌙" if self.current_theme == "dark" else "☀️"
        self.theme_btn = ttk.Button(utils_frame, text=theme_icon, width=3,
                                   command=self.toggle_theme, bootstyle="secondary-outline")
        self.theme_btn.pack(side=LEFT, padx=2)
        
        # History panel toggle
        history_btn = ttk.Button(utils_frame, text="📚", width=3,
                                command=self.toggle_history_panel, bootstyle="secondary-outline")
        history_btn.pack(side=LEFT, padx=2)
        
        # Help/Shortcuts
        help_btn = ttk.Button(utils_frame, text="?", width=3,
                             command=self.show_shortcuts, bootstyle="info-outline")
        help_btn.pack(side=LEFT, padx=2)
        
        # Separator
        separator = ttk.Separator(sidebar_container, orient='horizontal')
        separator.pack(fill=X, padx=20, pady=(0, 10))
        
        # FIXED FOOTER (Generation Controls)
        footer_frame = ttk.Frame(sidebar_container, padding=20)
        footer_frame.pack(fill=X, side=BOTTOM)
        
        # SCROLLED CONTROLS (Between header and footer)
        controls_frame = ScrolledFrame(sidebar_container, padding=(20, 0, 20, 0), autohide=True) 
        controls_frame.pack(fill=BOTH, expand=True)
        
        # Presets dropdown
        presets_frame = ttk.Frame(controls_frame)
        presets_frame.pack(fill=X, pady=(5, 10))
        
        ttk.Label(presets_frame, text="Preset:", font=("Consolas", 9), 
                 foreground="#888888").pack(side=LEFT, padx=(0, 5))
        
        preset_names = ["Custom"] + list(self.config.presets.keys())
        self.preset_var = tk.StringVar(value="Custom")
        self.preset_combo = ttk.Combobox(presets_frame, textvariable=self.preset_var,
                                        values=preset_names, state="readonly",
                                        bootstyle="dark", font=("Consolas", 9))
        self.preset_combo.pack(side=LEFT, fill=X, expand=True, padx=(0, 5))
        self.preset_combo.bind("<<ComboboxSelected>>", self.apply_preset)
        
        # Save preset button
        save_preset_btn = ttk.Button(presets_frame, text="💾", width=3,
                                    command=self.save_current_preset, bootstyle="success-outline")
        save_preset_btn.pack(side=LEFT)

        # Main Input (Shared)
        ttk.Label(controls_frame, text="CREATIVE VISION", font=("Consolas", 10, "bold"), 
                 foreground="#00cc00").pack(anchor="w", pady=(10, 5))
        
        # Style Preset Selection
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
        
        # Prevent scroll propagation to parent ScrolledFrame
        def on_mousewheel(event):
            # Check if mouse is over the text widget
            widget = event.widget
            if widget == self.prompt_text:
                # Only scroll the text if there's scrollable content
                view_range = self.prompt_text.yview()
                if view_range != (0.0, 1.0):  # Has scrollable content
                    self.prompt_text.yview_scroll(int(-1*(event.delta/120)), "units")
                    return "break"  # Stop propagation
                # If no scrollable content, also stop propagation when over text widget
                return "break"
            return None
        
        # Bind to text widget AND its parent frame to catch all events
        self.prompt_text.bind("<MouseWheel>", on_mousewheel)
        prompt_frame.bind("<MouseWheel>", on_mousewheel)
        
        
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
        
        # Smart Mode Select
        ttk.Label(tab_remix, text="Remix Mode", font=("Consolas", 9), foreground="#888888").pack(anchor="w", pady=(0, 5))
        
        self.remix_mode_var = tk.StringVar(value="standard")
        
        rm_frame = ttk.Frame(tab_remix)
        rm_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Radiobutton(rm_frame, text="Standard", variable=self.remix_mode_var, value="standard", bootstyle="info").pack(side=LEFT, padx=(0,10))
        ttk.Radiobutton(rm_frame, text="Smart: Outfit", variable=self.remix_mode_var, value="outfit", bootstyle="warning").pack(side=LEFT, padx=(0,10))
        ttk.Radiobutton(rm_frame, text="Smart: Backgrnd", variable=self.remix_mode_var, value="bg", bootstyle="success").pack(side=LEFT)
        
        # Preview Mask Button
        self.preview_mask_btn = ttk.Button(rm_frame, text="👁 Preview Mask", command=self.preview_mask_action, bootstyle="link", state=DISABLED)
        self.preview_mask_btn.pack(side=LEFT, padx=10)
        
        # Enable Preview only when image + smart mode
        self.remix_mode_var.trace_add("write", self.check_preview_state)
        
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
        ttk.Checkbutton(tab_remix, text="Color Match (match lighting/tone)", variable=self.color_match_var, bootstyle="toolbutton").pack(anchor="w")
        
        self.blend_edges_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_remix, text="Blend Edges (feather transitions)", variable=self.blend_edges_var, bootstyle="toolbutton").pack(anchor="w")
        
        self.preserve_edges_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab_remix, text="Preserve Structure (edge guidance)", variable=self.preserve_edges_var, bootstyle="toolbutton").pack(anchor="w")

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
        
        # Model Selector (New)
        ttk.Label(sliders_frame, text="Model", font=("Consolas", 9), foreground="#888888").grid(row=0, column=0, sticky="w", pady=10)
        
        self.model_var = tk.StringVar(value="Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32")
        self.model_combo = ttk.Combobox(sliders_frame, textvariable=self.model_var, values=[
            "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
            "Abrahamm3r/Z-Image-SDNQ-uint4-svd-r32",
            "Tongyi-MAI/Z-Image-Turbo"
        ], state="readonly", bootstyle="dark", width=35)
        self.model_combo.grid(row=0, column=1, padx=10, sticky="ew")
        
        # LoRA Selector
        ttk.Label(sliders_frame, text="LoRA Model", font=("Consolas", 9), foreground="#888888").grid(row=2, column=0, sticky="w", pady=10)
        
        # Scan for LoRAs
        self.lora_files = ["None"]
        lora_dir = os.path.join(os.getcwd(), "models", "loras")
        if os.path.exists(lora_dir):
            files = [f for f in os.listdir(lora_dir) if f.endswith(".safetensors")]
            self.lora_files.extend(files)
            
        self.lora_var = tk.StringVar(value="None")
        self.lora_combo = ttk.Combobox(sliders_frame, textvariable=self.lora_var, values=self.lora_files, state="readonly", bootstyle="dark", width=15)
        self.lora_combo.grid(row=2, column=1, padx=10, sticky="ew")

        # LoRA Strength Slider
        ttk.Label(sliders_frame, text="LoRA Strength", font=("Consolas", 9), foreground="#888888").grid(row=3, column=0, sticky="w", pady=10)
        self.lora_scale_var = tk.DoubleVar(value=0.8)
        self.lora_scale_slider = ttk.Scale(sliders_frame, from_=0.0, to=1.5, orient=HORIZONTAL, variable=self.lora_scale_var, bootstyle="info")
        self.lora_scale_slider.grid(row=3, column=1, padx=10, sticky="ew")

        # Steps
        ttk.Label(sliders_frame, text="Sampling Steps", font=("Consolas", 9), foreground="#888888").grid(row=4, column=0, sticky="w")
        self.steps_var = tk.IntVar(value=9)
        self.steps_spin = ttk.Spinbox(sliders_frame, from_=1, to=50, textvariable=self.steps_var, bootstyle="secondary", width=5)
        self.steps_spin.grid(row=4, column=1, padx=10, sticky="e")
        
        # Guidance
        ttk.Label(sliders_frame, text="Prompt Adherence", font=("Consolas", 9), foreground="#888888").grid(row=5, column=0, sticky="w", pady=10)
        self.cfg_var = tk.DoubleVar(value=0.0)
        self.cfg_scale = ttk.Scale(sliders_frame, from_=0.0, to=10.0, orient=HORIZONTAL, variable=self.cfg_var, bootstyle="info")
        self.cfg_scale.grid(row=5, column=1, padx=10, sticky="ew")
        
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
        self.seed_entry = ttk.Entry(seed_frame, textvariable=self.seed_var, bootstyle="dark")
        self.seed_entry.pack(side=LEFT, fill=X, expand=True)
        
        # Randomize Checkbox (Smart Seed)
        self.random_seed_var = tk.BooleanVar(value=True)
        
        def toggle_seed_lock():
            if self.random_seed_var.get():
                self.seed_entry.configure(state=DISABLED)
                self.seed_var.set(-1)
            else:
                self.seed_entry.configure(state=NORMAL)
                # If it was -1, give a starting seed
                if self.seed_var.get() == -1:
                    self.seed_var.set(12345678)
                    
        self.chk_random = ttk.Checkbutton(seed_frame, text="Randomize", variable=self.random_seed_var, command=toggle_seed_lock, bootstyle="toolbutton")
        self.chk_random.pack(side=LEFT, padx=(5,0))
        
        # Init state
        toggle_seed_lock()

        # Status Label
        self.status_lbl = ttk.Label(footer_frame, textvariable=self.status_var, wraplength=350, justify=CENTER, font=("Consolas", 9), foreground="#00aa00")
        self.status_lbl.pack(fill=X, pady=(0, 10))
        
        # Generate Button
        self.generate_btn = ttk.Button(footer_frame, text="⚡ GENERATE DREAM ⚡", command=self.start_generation, state=DISABLED, bootstyle="success")
        self.generate_btn.pack(fill=X, pady=(0, 5))
        
        # Action Row (Save + Upscale + Folder)
        action_row = ttk.Frame(footer_frame)
        action_row.pack(fill=X)
        
        self.open_folder_btn = ttk.Button(action_row, text="📂", width=3, command=self.open_output_folder, bootstyle="secondary-outline")
        self.open_folder_btn.pack(side=LEFT, padx=(0, 5))
        
        self.footer_save_btn = ttk.Button(action_row, text="💾 SAVE", command=self.save_image, state=DISABLED, bootstyle="secondary-outline")
        self.footer_save_btn.pack(side=LEFT, fill=X, expand=True, padx=(0, 2))
        
        self.upscale_btn = ttk.Button(action_row, text="🔍 UPSCALE 2x", command=self.upscale_action, state=DISABLED, bootstyle="info-outline")
        self.upscale_btn.pack(side=LEFT, fill=X, expand=True, padx=(2, 2))
        
        # Compare button (for img2img)
        self.compare_btn = ttk.Button(action_row, text="🔀", width=3, command=self.show_comparison, state=DISABLED, bootstyle="warning-outline")
        self.compare_btn.pack(side=LEFT, padx=(0, 0))

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
        
        # Send to Remix Button (Overlay)
        self.remix_btn = ttk.Button(viewport_frame, text="↦ REMIX", command=self.send_to_remix, state=DISABLED, bootstyle="warning", width=15)
        self.remix_btn.place(relx=0.05, rely=0.95, anchor="sw")
        
        # Gallery Strip (Bottom Overlay - Collapsible)
        self.gallery_frame = ttk.Frame(viewport_frame, height=120, bootstyle="dark")
        self.gallery_frame.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0) # Start hidden/low
        
        # Show/Hide Gallery Button
        self.gallery_toggle = ttk.Checkbutton(viewport_frame, text="Show Gallery", bootstyle="toolbutton", command=self.toggle_gallery)
        self.gallery_toggle.place(relx=0.5, rely=0.96, anchor="s")
        
        # Internal horizontal scrolling for gallery
        # Create a canvas with horizontal scrollbar
        gallery_canvas = tk.Canvas(self.gallery_frame, bg="#000000", highlightthickness=0, height=110)
        h_scrollbar = ttk.Scrollbar(self.gallery_frame, orient=HORIZONTAL, command=gallery_canvas.xview)
        gallery_canvas.configure(xscrollcommand=h_scrollbar.set)
        
        h_scrollbar.pack(side=BOTTOM, fill=X)
        gallery_canvas.pack(side=TOP, fill=BOTH, expand=True)
        
        # Create a frame inside the canvas to hold gallery items
        self.gallery_content = ttk.Frame(gallery_canvas)
        gallery_canvas.create_window((0, 0), window=self.gallery_content, anchor="nw")
        
        # Configure canvas scroll region when items are added
        def configure_scroll_region(event=None):
            gallery_canvas.configure(scrollregion=gallery_canvas.bbox("all"))
        self.gallery_content.bind("<Configure>", configure_scroll_region)
        
        # Progress Bar Overlay (Thin line at top of viewport)
        # Standard determinate bar (Green/Default) for generation steps
        self.progress = ttk.Progressbar(viewport_frame, mode='determinate', bootstyle="success", length=300)
        
        # Custom KITT Scanner for indeterminate waits (Upscaling, Loading)
        self.kitt_scanner = KITTScroller(viewport_frame, height=4)

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
        
        # Initialize history panels (hidden by default)
        self.history_panel_frame = ttk.Frame(main_pane, width=250, bootstyle="dark")
        
        # Add Prompt History
        self.prompt_history = PromptHistoryPanel(
            self.history_panel_frame, 
            self.config,
            on_select=self.load_prompt_from_history
        )
        self.prompt_history.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # Add Generation History
        self.generation_history = GenerationHistoryPanel(
            self.history_panel_frame,
            self.config,
            on_load=self.load_generation_from_history
        )
        self.generation_history.pack(fill=BOTH, expand=True)
        
        # History panel starts hidden unless configured otherwise
        if self.config.get_setting("show_history", False):
            self.sidebar_container = sidebar_container  # Store ref for toggle
            self.history_panel_frame.pack(fill=Y, side=LEFT, before=sidebar_container)
        
        # Store reference to sidebar container for later
        self.sidebar_container = sidebar_container
        
        # Drag-and-drop support (if available)
        if HAS_DND:
            self.setup_drag_drop()
    
    def setup_drag_drop(self):
        """Setup drag-and-drop for image uploads."""
        try:
            # Make the source thumbnail label a drop target
            if hasattr(self, 'source_thumb_lbl'):
                self.source_thumb_lbl.drop_target_register(DND_FILES)
                self.source_thumb_lbl.dnd_bind('<<Drop>>', self.on_drop_image)
        except Exception as e:
            print(f"Drag-and-drop setup failed: {e}")
    
    def on_drop_image(self, event):
        """Handle dropped image files."""
        files = self.tk.splitlist(event.data)
        if files:
            filepath = files[0]
            # Remove curly braces if present (Windows)
            filepath = filepath.strip('{}')
            
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                try:
                    from PIL import ImageOps
                    img = Image.open(filepath).convert("RGB")
                    self.source_image = ImageOps.exif_transpose(img)
                    
                    # Update thumbnail
                    thumb = self.source_image.copy()
                    thumb.thumbnail((200, 200))
                    self.tk_thumb = ImageTk.PhotoImage(thumb)
                    self.source_thumb_lbl.configure(image=self.tk_thumb, text="")
                    
                    # Auto-detect dimensions
                    self._auto_set_dimensions_from_source()
                    
                    # Switch to remix tab
                    self.notebook.select(1)
                    
                    self.status_var.set(f"Loaded: {os.path.basename(filepath)}")
                except Exception as e:
                    self.status_var.set(f"Error loading image: {e}")
    
    def _auto_set_dimensions_from_source(self):
        """Auto-set dimensions from source image."""
        if not self.source_image:
            return
            
        src_w, src_h = self.source_image.size
        self.source_aspect_ratio = src_w / src_h
        
        max_dim = 1536
        if max(src_w, src_h) > max_dim:
            scale = max_dim / max(src_w, src_h)
            src_w = int(src_w * scale)
            src_h = int(src_h * scale)
        
        src_w = src_w - (src_w % 16)
        src_h = src_h - (src_h % 16)
        
        self.aspect_var.set("Custom")
        self._updating_dims = True
        self.width_var.set(src_w)
        self.height_var.set(src_h)
        self._updating_dims = False
        
        self.status_var.set(f"Source: {self.source_image.size[0]}×{self.source_image.size[1]} → {src_w}×{src_h}")
        self.check_preview_state()
    
    def apply_preset(self, event=None):
        """Apply selected preset parameters."""
        preset_name = self.preset_var.get()
        
        if preset_name == "Custom":
            return
        
        preset = self.config.get_preset(preset_name)
        if not preset:
            return
        
        # Apply preset values
        self._updating_dims = True
        
        if "width" in preset:
            self.width_var.set(preset["width"])
        if "height" in preset:
            self.height_var.set(preset["height"])
        if "style" in preset:
            self.style_var.set(preset["style"])
        if "steps" in preset:
            self.steps_var.set(preset["steps"])
        if "guidance_scale" in preset:
            self.cfg_var.set(preset["guidance_scale"])
        
        self._updating_dims = False
        
        self.status_var.set(f"Applied preset: {preset_name}")
    
    def save_current_preset(self):
        """Save current parameters as a preset."""
        from tkinter import simpledialog
        
        name = simpledialog.askstring("Save Preset", "Enter preset name:")
        if not name:
            return
        
        # Gather current params
        params = {
            "width": self.width_var.get(),
            "height": self.height_var.get(),
            "style": self.style_var.get(),
            "steps": self.steps_var.get(),
            "guidance_scale": self.cfg_var.get()
        }
        
        self.config.add_preset(name, params)
        
        # Update preset combo
        preset_names = ["Custom"] + list(self.config.presets.keys())
        self.preset_combo.configure(values=preset_names)
        self.preset_var.set(name)
        
        messagebox.showinfo("Success", f"Preset '{name}' saved!")
    
    def load_prompt_from_history(self, prompt_data):
        """Load a prompt from history."""
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", prompt_data["prompt"])
        
        if "negative_prompt" in prompt_data and prompt_data["negative_prompt"]:
            self.neg_prompt_text.delete("1.0", tk.END)
            self.neg_prompt_text.insert("1.0", prompt_data["negative_prompt"])
        
        self.status_var.set("Loaded prompt from history")
    
    def load_generation_from_history(self, gen_data):
        """Load generation parameters from history."""
        # Load prompt
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", gen_data.get("prompt", ""))
        
        if gen_data.get("negative_prompt"):
            self.neg_prompt_text.delete("1.0", tk.END)
            self.neg_prompt_text.insert("1.0", gen_data["negative_prompt"])
        
        # Load parameters
        self._updating_dims = True
        self.width_var.set(gen_data.get("width", 1024))
        self.height_var.set(gen_data.get("height", 1024))
        self.steps_var.set(gen_data.get("steps", 9))
        self.cfg_var.set(gen_data.get("guidance", 0.0))
        
        if gen_data.get("seed", -1) != -1:
            self.random_seed_var.set(False)
            self.seed_var.set(gen_data["seed"])
            self.seed_entry.configure(state=NORMAL)
        
        if gen_data.get("style"):
            self.style_var.set(gen_data["style"])
        
        self._updating_dims = False
        
        self.status_var.set("Loaded parameters from history")

    def open_output_folder(self):
        # Default save loc is current dir, but let's try to open where user last saved or cwd
        path = os.getcwd()
        os.startfile(path)

    def send_to_remix(self):
        if not self.generated_image: return
        
        self.source_image = self.generated_image.copy()
        
        # Update thumb
        thumb = self.source_image.copy()
        thumb.thumbnail((200, 200))
        self.tk_thumb = ImageTk.PhotoImage(thumb)
        self.source_thumb_lbl.configure(image=self.tk_thumb, text="")
        
        # Switch tab
        self.notebook.select(1)
        
        self.status_var.set("Image sent to Remix. Adjust params and generate.")

    def toggle_gallery(self):
        # Simple animation or place/forget
        if self.gallery_frame.winfo_y() < self.canvas.winfo_height():
             # Hide
             self.gallery_frame.place(rely=1.2) # Move offscreen
        else:
             # Show (move up)
             self.gallery_frame.place(rely=0.88, relheight=0.12) # 12% height at bottom

    def add_to_gallery(self, image, prompt):
        # Create thumbnail
        thumb = image.copy()
        thumb.thumbnail((100, 100))
        tk_thumb = ImageTk.PhotoImage(thumb)
        
        # Create widget
        btn = ttk.Button(self.gallery_content, image=tk_thumb, command=lambda img=image: self.display_image(img))
        btn.image = tk_thumb # Keep ref
        btn.pack(side=LEFT, padx=5, pady=5)
        
        # Keep track
        self.gallery_images.append(image)
        
        # If gallery is hidden, maybe flash indicator? For now just add.

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
            
            # Update preview capability
            self.check_preview_state()

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

    def check_preview_state(self, *args):
        mode = self.remix_mode_var.get()
        if mode in ["outfit", "bg"] and self.source_image:
            self.preview_mask_btn.configure(state=NORMAL)
        else:
            self.preview_mask_btn.configure(state=DISABLED)

    def preview_mask_action(self):
        if not self.source_image or not self.generator: return
        
        mode = self.remix_mode_var.get()
        self.status_var.set(f"Generating Mask Preview ({mode})...")
        self.preview_mask_btn.configure(state=DISABLED)
        
        def run_preview():
            try:
                # Resize for speed if large
                src = self.source_image.copy()
                if max(src.size) > 1024:
                    src.thumbnail((1024, 1024))
                    
                mask = self.generator.preview_smart_mask(src, mode=mode)
                
                # Update UI in main thread
                self.after(0, lambda: self.show_mask_preview(mask))
            except Exception as e:
                print(f"Preview Failed: {e}")
                self.after(0, lambda: self.status_var.set(f"Mask Error: {e}"))
            finally:
                self.after(0, lambda: self.preview_mask_btn.configure(state=NORMAL))
                
        threading.Thread(target=run_preview, daemon=True).start()

    def show_mask_preview(self, mask):
        # Update the mask thumb
        thumb = mask.copy()
        thumb.thumbnail((200, 200))
        # Invert for visual clarity? Mask is White=Regen.
        # Let's show it as is.
        self.tk_mask_thumb = ImageTk.PhotoImage(thumb)
        self.mask_thumb_lbl.configure(image=self.tk_mask_thumb, text="Smart Mask Generated")
        self.status_var.set("Mask Preview Ready. White areas will be changed.")
        
        # Store as if uploaded
        # self.mask_image = mask # No, don't store as self.mask_image because that overrides smart logic gen
        # Just show visualization

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
        
    def check_nsfw_content(self, text):
        """Check if text contains NSFW keywords. Returns True if NSFW content detected."""
        nsfw_keywords = [
            # Explicit terms
            'nude', 'naked', 'nipple', 'breast', 'boob', 'tit', 'vagina', 
            'pussy', 'penis', 'dick', 'cock', 'sex', 'sexual', 'porn', 
            'nsfw', 'xxx', 'explicit', 'erotic', 'genitals', 'topless',
            'bottomless', 'underwear exposed', 'bra visible', 'panties',
            # Add more as needed
            'fellatio', 'cunnilingus', 'intercourse', 'masturbat',
            'orgasm', 'arousal', 'hentai', 'ahegao', 'lewd',
            'nudity', 'undressed', 'unclothed', 'provocative pose'
        ]
        
        text_lower = text.lower()
        for keyword in nsfw_keywords:
            if keyword in text_lower:
                return True
        return False
    
    def start_generation(self):
        if not self.generator:
            messagebox.showwarning("Not Ready", "Model still loading...")
            return
        
        self.generate_btn.configure(state=DISABLED)

        # Start animated seed cycling if random seed is enabled
        if self.random_seed_var.get():
            self.animate_seed_numbers()
        
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
                if self.remix_mode_var.get() in ["outfit", "bg"]:
                    # Smart Mode Logic
                    # We don't use manual mask here, we pass logic to backend
                    self.status_var.set(f"Smart Remix: {self.remix_mode_var.get().upper()}...")
                    
                    # Launch Smart Thread directly here or pass special flag?
                    # Better to inject into params so standard runner handles it?
                    # No, generate_smart is a different method signature or logic branch.
                    # Let's handle it by adding a 'smart_mode' key to params
                elif hasattr(self, 'mask_image') and self.mask_image:
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
        
        # NSFW Content Filter for Remix Mode (unless in TopSecret mode)
        if current_tab_index == 1 and not self.stealth_mode:  # Remix mode and not TopSecret
            prompt = self.prompt_text.get("1.0", tk.END).strip()
            
            if self.check_nsfw_content(prompt):
                # Block generation and show dialog
                self.reset_ui()
                messagebox.showwarning(
                    "Content Restricted",
                    "Remix mode cannot be used for NSFW content generation.\n\n"
                    "Please use appropriate prompts for Remix mode."
                )
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
            "preserve_edges": self.preserve_edges_var.get() if hasattr(self, 'preserve_edges_var') else False,
            "lora_path": os.path.join(os.getcwd(), "models", "loras", self.lora_var.get()) if self.lora_var.get() != "None" else None,
            "lora_scale": self.lora_scale_var.get() if hasattr(self, 'lora_scale_var') else 0.8,
            "smart_mode": self.remix_mode_var.get() if hasattr(self, 'remix_mode_var') else "standard"
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
            # Setup Progress Bar
            steps = params.get("steps", 9)
            
            # Reset and Configure
            def setup_progress():
                self.progress.stop()
                self.progress.configure(mode='determinate', maximum=steps, value=0)
                self.progress.place(relx=0, rely=0, relwidth=1)
            self.after(0, setup_progress)
            
            def step_callback(pipe, step, timestep, callback_kwargs):
                # Ensure step is int
                current_step = int(step)
                self.after(0, lambda: self.progress.configure(value=current_step + 1))
                return callback_kwargs

            params["callback"] = step_callback

            # Branch for Smart Mode
            # Remove smart_mode from params so it doesn't break standard generator
            smart_mode = params.pop("smart_mode", "standard")
            
            if smart_mode != "standard":
                print(f"Triggering Smart GEN: {smart_mode}")
                image = self.generator.generate_smart(
                    prompt=params["prompt"],
                    image=params["image"],
                    mode=smart_mode,
                    steps=params["steps"],
                    seed=params["seed"],
                    strength=params["strength"],
                    guidance=params["guidance_scale"]
                )
            else:
                # Standard Gen
                image = self.generator.generate(**params)
            
            # Apply invisible watermark immediately (unless in TopSecret mode)
            # This ensures even screenshots contain the signature
            image = self.generator.apply_watermark(image, include_id=not self.stealth_mode)

            # Store before image for comparison (if img2img)
            if params.get("image") is not None:
                self.comparison_before = params["image"]
            
            self.generated_image = image
            self.last_params = params  # Store for metadata persistence
            
            # Add to history ONLY if NOT in stealth mode
            if not self.stealth_mode:
                # Add to prompt history
                self.config.add_prompt(
                    params.get("prompt", ""),
                    params.get("negative_prompt", ""),
                    favorite=False
                )
                
                # Add to generation history
                gen_mode = "inpaint" if params.get("mask_image") is not None else (
                    "img2img" if params.get("image") is not None else "txt2img"
                )
                
                self.config.add_generation({
                    "prompt": params.get("prompt", ""),
                    "negative_prompt": params.get("negative_prompt", ""),
                    "model": self.model_var.get(),
                    "width": params.get("width", 1024),
                    "height": params.get("height", 1024),
                    "steps": params.get("steps", 9),
                    "guidance_scale": params.get("guidance_scale", 0.0),
                    "seed": params.get("seed", -1),
                    "strength": params.get("strength", 0.0),
                    "style": self.style_var.get(),
                    "lora": self.lora_var.get(),
                    "mode": gen_mode
                })
                
                # Refresh history panels
                if hasattr(self, 'prompt_history'):
                    self.after(0, self.prompt_history.refresh)
                if hasattr(self, 'generation_history'):
                    self.after(0, self.generation_history.refresh)
            
            self.after(0, self.display_image, image)
            self.after(0, lambda: self.add_to_gallery(image, params.get("prompt", "")))
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
        self.kitt_scanner.stop()
        self.kitt_scanner.place_forget()
        self.generate_btn.configure(state=NORMAL)
        
        # Stop seed animation if running
        if hasattr(self, 'seed_animating') and self.seed_animating:
            self.seed_animating = False
    
    def animate_seed_numbers(self):
        """Animate seed numbers cycling like a slot machine."""
        import random
        self.seed_animating = True
        self.seed_animation_count = 0
        
        # Temporarily enable seed entry to show animation
        was_disabled = str(self.seed_entry.cget('state')) == 'disabled'
        if was_disabled:
            self.seed_entry.configure(state='normal')
            # Also change background to show it's active
            self.seed_entry.configure(style='TEntry')
        
        def cycle_seed():
            if not self.seed_animating or self.seed_animation_count > 100:  # Increased cycles
                # Restore disabled state if it was disabled
                if was_disabled and self.random_seed_var.get():
                    self.seed_var.set(-1)
                    self.seed_entry.configure(state='disabled')
                return
            
            # Show random seed values cycling - make them look different each time
            random_seed = random.randint(10000000, 99999999)
            self.seed_var.set(random_seed)
            self.seed_animation_count += 1
            
            # Continue animation with faster speed for more dramatic effect
            self.after(30, cycle_seed)  # Faster: 30ms instead of 50ms
        
        cycle_seed()
        
        cycle_seed()

    def display_image(self, image):
        c_width, c_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if c_width <= 1: c_width, c_height = 800, 800
        
        ratio = min(c_width / image.width, c_height / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        self.tk_image = ImageTk.PhotoImage(image.resize(new_size, Image.LANCZOS))
        
        self.canvas.delete("all")
        self.canvas.create_image(c_width//2, c_height//2, image=self.tk_image, anchor=CENTER)
        self.save_btn.configure(state=NORMAL)
        if hasattr(self, 'footer_save_btn'):
            self.footer_save_btn.configure(state=NORMAL)
            
        if hasattr(self, 'upscale_btn'):
            self.upscale_btn.configure(state=NORMAL)
        
        # Enable compare button if we have a before image
        if hasattr(self, 'compare_btn') and hasattr(self, 'comparison_before') and self.comparison_before:
            self.compare_btn.configure(state=NORMAL)
        
        if hasattr(self, 'remix_btn'):
            self.remix_btn.configure(state=NORMAL)
    
    def show_comparison(self):
        """Show before/after comparison slider."""
        if hasattr(self, 'comparison_before') and self.comparison_before and self.generated_image:
            ComparisonSlider(self, self.comparison_before, self.generated_image)

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
                # Prepare Metadata (PNG Info)
                from PIL.PngImagePlugin import PngInfo
                metadata = PngInfo()
                
                # Add basic attribution
                metadata.add_text("Software", "Electric Sheep Dreams v0.1")
                metadata.add_text("Source", "AI Generated (Z-Image-Turbo)")
                
                # Image is already watermarked during generation
                # No need to watermark again on save
                final_image = self.generated_image
                     
                # Add Generation Params if available
                if hasattr(self, 'last_params'):
                    p = self.last_params
                    metadata.add_text("Prompt", p.get("prompt", ""))
                    metadata.add_text("Negative Prompt", p.get("negative_prompt", ""))
                    metadata.add_text("Seed", str(p.get("seed", -1)))
                    metadata.add_text("Steps", str(p.get("steps", "")))
                    metadata.add_text("Guidance", str(p.get("guidance_scale", "")))
                    if "style" in p.get("prompt", ""): # Just a heuristic, or we can store style separately
                        pass 

                final_image.save(path, pnginfo=metadata)
                self.status_var.set("Saved Securely.")
                messagebox.showinfo("Saved", f"Image saved to {path}\n(Metadata & Watermark Embedded)")

    def upscale_action(self):
        if not self.generated_image: return
        self.upscale_btn.configure(state=DISABLED)
        self.status_var.set("Upscaling Image (2x)... Please wait.")
        
        # Show indeterminate progress for upscaling
        # KITT effect: Custom Canvas Widget
        self.kitt_scanner.place(relx=0, rely=0, relwidth=1)
        self.kitt_scanner.start()
        
        def run_upscale():
            try:
                upscaled = self.generator.upscale_image(self.generated_image)
                
                # Update UI in main thread
                def update_ui():
                    self.generated_image = upscaled
                    self.display_image(self.generated_image)
                    self.status_var.set("Upscale Complete!")
                    self.upscale_btn.configure(state=NORMAL)
                    self.kitt_scanner.stop()
                    self.kitt_scanner.place_forget()
                    
                self.after(0, update_ui)
                
            except Exception as e:
                def show_error():
                    self.status_var.set(f"Upscale Error: {e}")
                    messagebox.showerror("Error", str(e))
                    self.upscale_btn.configure(state=NORMAL)
                    self.kitt_scanner.stop()
                    self.kitt_scanner.place_forget()
                self.after(0, show_error)
        
        threading.Thread(target=run_upscale, daemon=True).start()



if __name__ == "__main__":
    import sys
    stealth = "--TopSecret" in sys.argv
    if stealth:
        print(">> TOP SECRET MODE ENGAGED: Device Fingerprinting DISABLED <<")
        
    app = ZImageApp(stealth_mode=stealth)
    app.place_window_center()
    app.mainloop()
    app.mainloop()
