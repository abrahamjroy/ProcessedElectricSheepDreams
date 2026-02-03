"""
UI Widgets Module
Custom widgets and dialogs for the image generation application.
"""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from typing import Callable, Optional, Dict, Any


class KeyboardShortcutsDialog(tk.Toplevel):
    """Dialog showing keyboard shortcuts."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Keyboard Shortcuts")
        self.geometry("600x500")
        self.resizable(False, False)
        
        # Style to match main app
        self.configure(bg="#000000")
        
        # Header
        header = ttk.Label(self, text="⌨️ KEYBOARD SHORTCUTS", 
                          font=("Consolas", 16, "bold"), 
                          foreground="#00ff00",
                          background="#000000")
        header.pack(pady=20)
        
        # Create scrollable frame
        from ttkbootstrap.scrolled import ScrolledFrame
        scroll_frame = ScrolledFrame(self, autohide=True, bootstyle="dark")
        scroll_frame.pack(fill=BOTH, expand=True, padx=20, pady=(0, 20))
        
        shortcuts = {
            "Generation": [
                ("Shift+Enter", "Trigger generation"),
                ("Esc", "Cancel current generation"),
                ("Ctrl+G", "Generate with current settings"),
            ],
            "File Operations": [
                ("Ctrl+S", "Save generated image"),
                ("Ctrl+Shift+S", "Save with full metadata"),
                ("Ctrl+O", "Open output folder"),
            ],
            "Navigation": [
                ("Tab", "Switch between CREATE/REMIX tabs"),
                ("Ctrl+H", "Toggle history panel"),
                ("Ctrl+G", "Toggle gallery"),
            ],
            "Editing": [
                ("Ctrl+C", "Copy prompt to clipboard"),
                ("Ctrl+V", "Paste image from clipboard"),
                ("Ctrl+Z", "Undo (prompt text)"),
            ],
            "View": [
                ("F11", "Toggle fullscreen viewport"),
                ("Scroll", "Zoom viewport"),
                ("Ctrl++", "Increase UI scale"),
                ("Ctrl+-", "Decrease UI scale"),
            ],
            "Advanced": [
                ("Ctrl+E", "Export workspace"),
                ("Ctrl+I", "Import workspace"),
                ("F5", "Reload models"),
                ("?", "Show this help"),
            ]
        }
        
        for category, items in shortcuts.items():
            # Category header
            cat_label = ttk.Label(scroll_frame, text=category, 
                                 font=("Consolas", 12, "bold"),
                                 foreground="#00cc00",
                                 background="#000000")
            cat_label.pack(anchor="w", pady=(15, 5))
            
            # Shortcuts table
            for key, description in items:
                row = ttk.Frame(scroll_frame, bootstyle="dark")
                row.pack(fill=X, pady=2)
                
                key_label = ttk.Label(row, text=key, 
                                     font=("Consolas", 10, "bold"),
                                     foreground="#ffffff",
                                     background="#222222",
                                     padding=5,
                                     width=20)
                key_label.pack(side=LEFT, padx=(0, 10))
                
                desc_label = ttk.Label(row, text=description,
                                      font=("Consolas", 9),
                                      foreground="#cccccc",
                                      background="#000000")
                desc_label.pack(side=LEFT, fill=X, expand=True)
        
        # Close button
        close_btn = ttk.Button(self, text="Close", command=self.destroy, 
                              bootstyle="secondary", width=15)
        close_btn.pack(pady=(0, 20))
        
        # Center on parent
        self.transient(parent)
        self.grab_set()
        
        # Center window
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")


class PromptHistoryPanel(ttk.Frame):
    """Panel showing prompt history with favorites."""
    
    def __init__(self, parent, config_manager, on_select: Callable, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config_manager
        self.on_select = on_select
        
        # Header with controls
        header = ttk.Frame(self)
        header.pack(fill=X, padx=5, pady=5)
        
        ttk.Label(header, text="📚 Prompt History", 
                 font=("Consolas", 11, "bold"),
                 foreground="#00cc00").pack(side=LEFT)
        
        # Search box
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._on_search)
        search_entry = ttk.Entry(header, textvariable=self.search_var, 
                                width=20, bootstyle="dark")
        search_entry.pack(side=RIGHT, padx=5)
        ttk.Label(header, text="🔍", font=("Consolas", 10)).pack(side=RIGHT)
        
        # Filter buttons
        filter_frame = ttk.Frame(self)
        filter_frame.pack(fill=X, padx=5, pady=(0, 5))
        
        self.filter_var = tk.StringVar(value="all")
        ttk.Radiobutton(filter_frame, text="All", variable=self.filter_var, 
                       value="all", command=self.refresh, 
                       bootstyle="toolbutton").pack(side=LEFT, padx=2)
        ttk.Radiobutton(filter_frame, text="⭐ Favorites", variable=self.filter_var, 
                       value="favorites", command=self.refresh,
                       bootstyle="toolbutton").pack(side=LEFT, padx=2)
        
        # Scrollable list
        from ttkbootstrap.scrolled import ScrolledFrame
        self.scroll_frame = ScrolledFrame(self, autohide=True, bootstyle="dark")
        self.scroll_frame.pack(fill=BOTH, expand=True)
        
        self.refresh()
    
    def _on_search(self, *args):
        """Handle search input."""
        self.refresh()
    
    def refresh(self):
        """Refresh the prompt list."""
        # Clear existing widgets
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        
        # Get prompts based on filter
        query = self.search_var.get().strip()
        
        if self.filter_var.get() == "favorites":
            prompts = self.config.get_favorites()
        elif query:
            prompts = self.config.search_prompts(query)
        else:
            prompts = self.config.prompt_history
        
        # Display prompts
        for idx, prompt_data in enumerate(prompts):
            self._create_prompt_entry(prompt_data, idx)
        
        if not prompts:
            ttk.Label(self.scroll_frame, text="No prompts found", 
                     foreground="#666666",
                     font=("Consolas", 9, "italic")).pack(pady=20)
    
    def _create_prompt_entry(self, prompt_data: Dict[str, Any], index: int):
        """Create a single prompt entry widget."""
        frame = ttk.Frame(self.scroll_frame, bootstyle="dark", padding=5)
        frame.pack(fill=X, padx=5, pady=2)
        
        # Alternate background
        bg = "#0a0a0a" if index % 2 == 0 else "#050505"
        frame.configure(style="dark")
        
        # Favorite star button
        is_fav = prompt_data.get("favorite", False)
        star_btn = ttk.Button(frame, text="⭐" if is_fav else "☆", 
                             width=3,
                             command=lambda: self._toggle_favorite(prompt_data["prompt"]),
                             bootstyle="link")
        star_btn.pack(side=LEFT, padx=(0, 5))
        
        # Prompt text (truncated)
        prompt_text = prompt_data["prompt"]
        if len(prompt_text) > 80:
            prompt_text = prompt_text[:77] + "..."
        
        prompt_label = ttk.Label(frame, text=prompt_text,
                                font=("Consolas", 9),
                                foreground="#e0e0e0",
                                cursor="hand2")
        prompt_label.pack(side=LEFT, fill=X, expand=True)
        prompt_label.bind("<Button-1>", lambda e: self.on_select(prompt_data))
        
        # Use count badge
        use_count = prompt_data.get("use_count", 1)
        if use_count > 1:
            badge = ttk.Label(frame, text=f"{use_count}×",
                            font=("Consolas", 8),
                            foreground="#888888")
            badge.pack(side=RIGHT, padx=5)
    
    def _toggle_favorite(self, prompt: str):
        """Toggle favorite status."""
        self.config.toggle_favorite(prompt)
        self.refresh()


class GenerationHistoryPanel(ttk.Frame):
    """Panel showing generation history."""
    
    def __init__(self, parent, config_manager, on_load: Callable, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config_manager
        self.on_load = on_load
        
        # Header
        header = ttk.Frame(self)
        header.pack(fill=X, padx=5, pady=5)
        
        ttk.Label(header, text="🕐 Generation History",
                 font=("Consolas", 11, "bold"),
                 foreground="#00cc00").pack(side=LEFT)
        
        ttk.Button(header, text="Clear", command=self._clear_history,
                  bootstyle="danger-outline", width=8).pack(side=RIGHT)
        
        # Scrollable list
        from ttkbootstrap.scrolled import ScrolledFrame
        self.scroll_frame = ScrolledFrame(self, autohide=True, bootstyle="dark")
        self.scroll_frame.pack(fill=BOTH, expand=True)
        
        self.refresh()
    
    def _clear_history(self):
        """Clear all history."""
        if messagebox.askyesno("Clear History", "Delete all generation history?"):
            self.config.clear_history()
            self.refresh()
    
    def refresh(self):
        """Refresh the history list."""
        # Clear existing
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        
        generations = self.config.get_recent_generations(20)
        
        for idx, gen in enumerate(generations):
            self._create_generation_entry(gen, idx)
        
        if not generations:
            ttk.Label(self.scroll_frame, text="No generations yet",
                     foreground="#666666",
                     font=("Consolas", 9, "italic")).pack(pady=20)
    
    def _create_generation_entry(self, gen: Dict[str, Any], index: int):
        """Create a single generation entry."""
        frame = ttk.Frame(self.scroll_frame, bootstyle="dark", padding=5)
        frame.pack(fill=X, padx=5, pady=2)
        
        # Info section
        info_frame = ttk.Frame(frame)
        info_frame.pack(side=LEFT, fill=X, expand=True)
        
        # Prompt (truncated)
        prompt = gen.get("prompt", "")
        if len(prompt) > 60:
            prompt = prompt[:57] + "..."
        
        prompt_label = ttk.Label(info_frame, text=prompt,
                                font=("Consolas", 9, "bold"),
                                foreground="#ffffff",
                                cursor="hand2")
        prompt_label.pack(anchor="w")
        prompt_label.bind("<Button-1>", lambda e: self.on_load(gen))
        
        # Metadata
        meta = f"{gen['width']}×{gen['height']} • {gen['steps']} steps • Seed: {gen['seed']}"
        ttk.Label(info_frame, text=meta,
                 font=("Consolas", 8),
                 foreground="#888888").pack(anchor="w")
        
        # Timestamp
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(gen["timestamp"])
            time_str = dt.strftime("%H:%M:%S")
        except:
            time_str = "Unknown"
        
        ttk.Label(frame, text=time_str,
                 font=("Consolas", 8),
                 foreground="#666666").pack(side=RIGHT)


class ComparisonSlider(tk.Toplevel):
    """Before/After comparison slider widget."""
    
    def __init__(self, parent, before_image: Image.Image, after_image: Image.Image):
        super().__init__(parent)
        self.title("Compare: Before ↔ After")
        self.geometry("1000x700")
        
        self.before_img = before_image
        self.after_img = after_image
        self.slider_position = 0.5  # 0 to 1
        
        # Canvas for display
        self.canvas = tk.Canvas(self, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        
        # Slider control
        control_frame = ttk.Frame(self, bootstyle="dark", padding=10)
        control_frame.pack(fill=X)
        
        ttk.Label(control_frame, text="← BEFORE",
                 font=("Consolas", 9, "bold")).pack(side=LEFT, padx=10)
        
        self.slider = ttk.Scale(control_frame, from_=0, to=1, orient=HORIZONTAL,
                               command=self._on_slider_change,
                               bootstyle="info")
        self.slider.set(0.5)
        self.slider.pack(side=LEFT, fill=X, expand=True, padx=10)
        
        ttk.Label(control_frame, text="AFTER →",
                 font=("Consolas", 9, "bold")).pack(side=LEFT, padx=10)
        
        # Bind keyboard
        self.bind("<Left>", lambda e: self.slider.set(max(0, self.slider.get() - 0.05)))
        self.bind("<Right>", lambda e: self.slider.set(min(1, self.slider.get() + 0.05)))
        self.bind("<Escape>", lambda e: self.destroy())
        
        # Initial render
        self.after(100, self._render)
    
    def _on_slider_change(self, value):
        """Handle slider movement."""
        self.slider_position = float(value)
        self._render()
    
    def _render(self):
        """Render the comparison view."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10:
            return  # Not ready yet
        
        # Resize images to fit
        img_ratio = min(canvas_width / self.before_img.width,
                       canvas_height / self.before_img.height)
        
        new_size = (int(self.before_img.width * img_ratio),
                   int(self.before_img.height * img_ratio))
        
        before_resized = self.before_img.resize(new_size, Image.LANCZOS)
        after_resized = self.after_img.resize(new_size, Image.LANCZOS)
        
        # Create composite image
        split_x = int(new_size[0] * self.slider_position)
        
        composite = Image.new("RGB", new_size)
        composite.paste(before_resized.crop((0, 0, split_x, new_size[1])), (0, 0))
        composite.paste(after_resized.crop((split_x, 0, new_size[0], new_size[1])), 
                       (split_x, 0))
        
        # Draw divider line
        from PIL import ImageDraw
        draw = ImageDraw.Draw(composite)
        draw.line([(split_x, 0), (split_x, new_size[1])], fill="white", width=3)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(composite)
        
        # Display
        self.canvas.delete("all")
        x_offset = (canvas_width - new_size[0]) // 2
        y_offset = (canvas_height - new_size[1]) // 2
        self.canvas.create_image(x_offset, y_offset, image=self.photo, anchor="nw")
