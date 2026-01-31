"""
Configuration Manager
Handles persistent storage of user preferences, history, and settings.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class ConfigManager:
    """Manages application configuration and persistent data."""
    
    def __init__(self, config_dir: str = ".config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # File paths
        self.settings_file = self.config_dir / "settings.json"
        self.prompt_history_file = self.config_dir / "prompt_history.json"
        self.generation_history_file = self.config_dir / "generation_history.json"
        self.presets_file = self.config_dir / "presets.json"
        
        # Initialize default settings
        self.settings = self._load_settings()
        self.prompt_history = self._load_prompt_history()
        self.generation_history = self._load_generation_history()
        self.presets = self._load_presets()
    
    # ----- Settings Management -----
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load application settings."""
        default_settings = {
            "theme": "dark",  # dark, light, auto
            "last_model": "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
            "gallery_view": "grid",  # grid or list
            "gallery_sort": "date",  # date, prompt, model, seed
            "auto_save": False,
            "show_gallery": False,
            "show_history": False,
            "stealth_mode": False
        }
        
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    default_settings.update(loaded)
            except Exception as e:
                print(f"Failed to load settings: {e}")
        
        return default_settings
    
    def save_settings(self):
        """Save current settings to disk."""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save settings: {e}")
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        return self.settings.get(key, default)
    
    def set_setting(self, key: str, value: Any):
        """Set a specific setting value and save."""
        self.settings[key] = value
        self.save_settings()
    
    # ----- Prompt History Management -----
    
    def _load_prompt_history(self) -> List[Dict[str, Any]]:
        """Load prompt history."""
        if self.prompt_history_file.exists():
            try:
                with open(self.prompt_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load prompt history: {e}")
        return []
    
    def save_prompt_history(self):
        """Save prompt history to disk."""
        try:
            with open(self.prompt_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.prompt_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save prompt history: {e}")
    
    def add_prompt(self, prompt: str, negative_prompt: str = "", favorite: bool = False):
        """Add a prompt to history."""
        # Check if prompt already exists
        existing = next((p for p in self.prompt_history if p["prompt"] == prompt), None)
        
        if existing:
            # Update timestamp and move to top
            existing["timestamp"] = datetime.now().isoformat()
            existing["use_count"] = existing.get("use_count", 1) + 1
            self.prompt_history.remove(existing)
            self.prompt_history.insert(0, existing)
        else:
            # Add new prompt
            self.prompt_history.insert(0, {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "favorite": favorite,
                "timestamp": datetime.now().isoformat(),
                "use_count": 1
            })
        
        # Keep only last 100 prompts
        self.prompt_history = self.prompt_history[:100]
        self.save_prompt_history()
    
    def toggle_favorite(self, prompt: str):
        """Toggle favorite status for a prompt."""
        for p in self.prompt_history:
            if p["prompt"] == prompt:
                p["favorite"] = not p.get("favorite", False)
                self.save_prompt_history()
                return p["favorite"]
        return False
    
    def get_favorites(self) -> List[Dict[str, Any]]:
        """Get all favorite prompts."""
        return [p for p in self.prompt_history if p.get("favorite", False)]
    
    def search_prompts(self, query: str) -> List[Dict[str, Any]]:
        """Search prompts by text."""
        query = query.lower()
        return [p for p in self.prompt_history if query in p["prompt"].lower()]
    
    # ----- Generation History Management -----
    
    def _load_generation_history(self) -> List[Dict[str, Any]]:
        """Load generation history."""
        if self.generation_history_file.exists():
            try:
                with open(self.generation_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load generation history: {e}")
        return []
    
    def save_generation_history(self):
        """Save generation history to disk."""
        try:
            with open(self.generation_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.generation_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save generation history: {e}")
    
    def add_generation(self, params: Dict[str, Any], image_path: Optional[str] = None):
        """Add a generation to history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": params.get("prompt", ""),
            "negative_prompt": params.get("negative_prompt", ""),
            "model": params.get("model", ""),
            "width": params.get("width", 1024),
            "height": params.get("height", 1024),
            "steps": params.get("steps", 9),
            "guidance": params.get("guidance_scale", 0.0),
            "seed": params.get("seed", -1),
            "strength": params.get("strength", 0.0),
            "style": params.get("style", ""),
            "lora": params.get("lora", ""),
            "mode": params.get("mode", "txt2img"),
            "image_path": image_path
        }
        
        self.generation_history.insert(0, entry)
        
        # Keep only last 50 generations
        self.generation_history = self.generation_history[:50]
        self.save_generation_history()
    
    def get_recent_generations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent generations."""
        return self.generation_history[:limit]
    
    def clear_history(self):
        """Clear all generation history."""
        self.generation_history = []
        self.save_generation_history()
    
    # ----- Presets Management -----
    
    def _load_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load parameter presets."""
        default_presets = {
            "Portrait": {
                "width": 896,
                "height": 1152,
                "style": "Style: 3D Render (Octane/Unreal)",
                "steps": 12,
                "guidance_scale": 0.0
            },
            "Landscape": {
                "width": 1344,
                "height": 768,
                "style": "Style: Cinematic (Dramatic Lighting)",
                "steps": 12,
                "guidance_scale": 0.0
            },
            "Square Social": {
                "width": 1024,
                "height": 1024,
                "style": "Style: Digital Art (Polished)",
                "steps": 12,
                "guidance_scale": 0.0
            },
            "Concept Art": {
                "width": 1280,
                "height": 768,
                "style": "Style: Digital Art (Polished)",
                "steps": 15,
                "guidance_scale": 2.0
            },
            "Quick Draft": {
                "width": 512,
                "height": 512,
                "style": "No Style Preset",
                "steps": 5,
                "guidance_scale": 0.0
            }
        }
        
        if self.presets_file.exists():
            try:
                with open(self.presets_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    default_presets.update(loaded)
            except Exception as e:
                print(f"Failed to load presets: {e}")
        
        return default_presets
    
    def save_presets(self):
        """Save presets to disk."""
        try:
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                json.dump(self.presets, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save presets: {e}")
    
    def add_preset(self, name: str, params: Dict[str, Any]):
        """Add or update a preset."""
        self.presets[name] = params
        self.save_presets()
    
    def delete_preset(self, name: str):
        """Delete a custom preset (built-in presets can't be deleted)."""
        builtin = ["Portrait", "Landscape", "Square Social", "Concept Art", "Quick Draft"]
        if name not in builtin and name in self.presets:
            del self.presets[name]
            self.save_presets()
            return True
        return False
    
    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific preset."""
        return self.presets.get(name)
    
    # ----- Export/Import -----
    
    def export_workspace(self, filepath: str):
        """Export all settings, history, and presets to a single file."""
        workspace = {
            "settings": self.settings,
            "prompt_history": self.prompt_history,
            "generation_history": self.generation_history,
            "presets": self.presets,
            "export_date": datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(workspace, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to export workspace: {e}")
            return False
    
    def import_workspace(self, filepath: str, merge: bool = False):
        """Import workspace from file. If merge=True, combines with existing data."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                workspace = json.load(f)
            
            if merge:
                # Merge settings
                self.settings.update(workspace.get("settings", {}))
                
                # Merge prompts (avoid duplicates)
                existing_prompts = {p["prompt"] for p in self.prompt_history}
                for prompt in workspace.get("prompt_history", []):
                    if prompt["prompt"] not in existing_prompts:
                        self.prompt_history.append(prompt)
                
                # Merge generations
                self.generation_history.extend(workspace.get("generation_history", []))
                
                # Merge presets
                self.presets.update(workspace.get("presets", {}))
            else:
                # Replace all data
                self.settings = workspace.get("settings", {})
                self.prompt_history = workspace.get("prompt_history", [])
                self.generation_history = workspace.get("generation_history", [])
                self.presets = workspace.get("presets", {})
            
            # Save everything
            self.save_settings()
            self.save_prompt_history()
            self.save_generation_history()
            self.save_presets()
            
            return True
        except Exception as e:
            print(f"Failed to import workspace: {e}")
            return False
