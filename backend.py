"""
Z-Image Turbo Backend
Image generation engine using Z-Image-Turbo with SDNQ quantization.
Supports Text-to-Image, Image-to-Image, and Inpainting modes.
"""

import torch
import diffusers
import triton
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from PIL import Image


class ImageGenerator:
    """
    Core image generation class wrapping ZImagePipeline.
    Automatically selects optimal device and precision settings.
    """
    
    def __init__(self, model_id: str = "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        # Enable TF32 for faster matrix operations on Ampere+ GPUs
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print(f"Loading model: {model_id} on {self.device}...")
        
        self.pipe = diffusers.ZImagePipeline.from_pretrained(
            model_id, 
            torch_dtype=self.dtype
        )

        # Apply SDNQ quantization optimizations if Triton is available
        if self.device == "cuda" and triton_is_available:
            print("Applying SDNQ optimizations...")
            self.pipe.transformer = apply_sdnq_options_to_model(
                self.pipe.transformer, use_quantized_matmul=True
            )
            self.pipe.text_encoder = apply_sdnq_options_to_model(
                self.pipe.text_encoder, use_quantized_matmul=True
            )

        # Enable CPU offload for efficient VRAM management
        self.pipe.enable_model_cpu_offload()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        guidance_scale: float = 0.0,
        seed: int = -1,
        image: Image.Image = None,
        mask_image: Image.Image = None,
        strength: float = 0.7,
        color_match: bool = False,
        blend_edges: bool = False,
        preserve_edges: bool = False,
        lora_path: str = None,
        lora_scale: float = 0.8,
        callback: callable = None
    ) -> Image.Image:
        """
        Generate an image from text prompt or transform an existing image.
        
        Args:
            prompt: Text description of desired image
            negative_prompt: Elements to exclude from generation
            width, height: Output dimensions (must be divisible by 16)
            steps: Number of inference steps (default 9 for turbo model)
            guidance_scale: CFG scale (0.0 for turbo model)
            seed: Random seed (-1 for random)
            image: Source image for Img2Img/Inpainting modes
            mask_image: Mask for inpainting (white=regenerate, black=preserve)
            strength: Transformation strength for Img2Img (0.0-1.0)
            color_match: Match color statistics to source in masked areas
            blend_edges: Feather mask edges for smooth transitions
            preserve_edges: Maintain structural edges from source
            
        Returns:
            Generated PIL Image
        """
        # Initialize random generator
        if seed != -1:
            generator = torch.manual_seed(seed)
        else:
            generator = torch.manual_seed(torch.randint(0, 2**32, (1,)).item())

        print(f"Generating with seed: {generator.initial_seed()}")
        
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "callback_on_step_end": callback,
        }



        # Apply LoRA if requested
        if lora_path and "None" not in lora_path:
            try:
                print(f"Loading LoRA: {lora_path}")
                adapter_name = "custom_lora"
                self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                self.pipe.set_adapters([adapter_name], adapter_weights=[lora_scale]) 
            except Exception as e:
                print(f"LoRA Load Failed: {e}")

        try:
            # Inpainting mode: image + mask provided
            if image and mask_image:
                result = self._generate_inpaint(args, image, mask_image, width, height, 
                                            strength, color_match, blend_edges, preserve_edges)
            
            # Img2Img mode: image without mask
            elif image:
                result = self._generate_img2img(args, image, width, height, strength, color_match, preserve_edges)
            
            # Text-to-Image mode: no source image
            else:
                args["width"] = width
                args["height"] = height
                result = self.pipe(**args).images[0]
                
        finally:
            # Always unload LoRA to prevent pollution of future generations
            if lora_path and "None" not in lora_path:
                try:
                    print("Unloading LoRA...")
                    self.pipe.unload_lora_weights()
                except: 
                    pass
        
        return result

    def _generate_img2img(self, args, image, width, height, strength, color_match=False, preserve_edges=False):
        """Handle Image-to-Image generation."""
        if not hasattr(self, 'img2img_pipe'):
            print("Initializing Img2Img Pipeline...")
            from diffusers import AutoPipelineForImage2Image
            self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
        
        # Resize source
        resized_source = image.resize((width, height), Image.LANCZOS)
        
        args["image"] = resized_source
        args["strength"] = strength
        
        # Generate
        generated = self.img2img_pipe(**args).images[0]
        
        # Create full mask for post-processing logic (white = all changed, used to apply uniform effect)
        full_mask = Image.new("L", (width, height), 255)
        
        # Apply Post-Processing
        if preserve_edges:
            generated = self._apply_edge_preservation(resized_source, generated, full_mask)
            
        if color_match:
            generated = self._apply_color_match(resized_source, generated, full_mask)
            
        return generated

    def _generate_inpaint(self, args, image, mask_image, width, height, 
                          strength, color_match, blend_edges, preserve_edges):
        """Handle Inpainting generation with post-processing options."""
        if not hasattr(self, 'img2img_pipe'):
            print("Initializing Img2Img Pipeline for Inpainting...")
            from diffusers import AutoPipelineForImage2Image
            self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
        
        # Resize inputs to target dimensions
        resized_image = image.resize((width, height), Image.LANCZOS)
        resized_mask = mask_image.convert("L").resize((width, height), Image.NEAREST)
        
        args["image"] = resized_image
        args["strength"] = strength
        
        # Generate base result
        generated = self.img2img_pipe(**args).images[0]
        
        # Apply post-processing options
        if preserve_edges:
            generated = self._apply_edge_preservation(resized_image, generated, resized_mask)
        
        if color_match:
            generated = self._apply_color_match(resized_image, generated, resized_mask)
        
        # Feather mask edges if requested
        composite_mask = resized_mask
        if blend_edges:
            from PIL import ImageFilter
            composite_mask = resized_mask.filter(ImageFilter.GaussianBlur(radius=8))
        
        # Composite: original where mask is black, generated where mask is white
        return Image.composite(generated, resized_image, composite_mask)

    def _apply_edge_preservation(self, source, generated, mask):
        """Blend structural edges from source into generated image."""
        import numpy as np
        from PIL import ImageFilter
        
        edges = source.convert("L").filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        gen_arr = np.array(generated).astype(np.float32)
        edge_arr = np.array(edges).astype(np.float32) / 255.0
        mask_arr = np.array(mask).astype(np.float32) / 255.0
        
        edge_strength = 0.15
        for c in range(3):
            edge_effect = edge_arr * edge_strength * mask_arr
            gen_arr[:, :, c] = gen_arr[:, :, c] * (1 - edge_effect * 0.3)
        
        return Image.fromarray(np.clip(gen_arr, 0, 255).astype(np.uint8))

    def _apply_color_match(self, source, generated, mask):
        """Transfer color statistics from source to generated in masked regions."""
        import numpy as np
        
        src_arr = np.array(source).astype(np.float32)
        gen_arr = np.array(generated).astype(np.float32)
        mask_arr = np.array(mask).astype(np.float32) / 255.0
        
        for c in range(3):
            src_mean, src_std = np.mean(src_arr[:, :, c]), np.std(src_arr[:, :, c])
            masked_gen = gen_arr[:, :, c][mask_arr > 0.5]
            
            if len(masked_gen) > 0:
                gen_mean = np.mean(masked_gen)
                gen_std = np.std(masked_gen) + 1e-5
                adjusted = (gen_arr[:, :, c] - gen_mean) * (src_std / gen_std) + src_mean
                gen_arr[:, :, c] = np.where(mask_arr > 0.5, adjusted, gen_arr[:, :, c])
        
        return Image.fromarray(np.clip(gen_arr, 0, 255).astype(np.uint8))

    def smart_resize(self, image: Image.Image, max_dim: int = 1536) -> Image.Image:
        """Resize image to safe dimensions (capped size, multiple of 16)."""
        w, h = image.size
        
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            w, h = int(w * scale), int(h * scale)
        
        w, h = w - (w % 16), h - (h % 16)
        
        if (w, h) != image.size:
            return image.resize((w, h), Image.LANCZOS)
        return image

    def upscale_image(self, image: Image.Image) -> Image.Image:
        """Upscale image 2x using Swin2SR or Lanczos fallback."""
        print("Upscaling image 2x...")
        try:
            from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
            import numpy as np
            
            if not hasattr(self, 'upscaler'):
                print("Loading Swin2SR upscaler...")
                self.processor = Swin2SRImageProcessor.from_pretrained(
                    "caidas/swin2SR-classical-sr-x2-64"
                )
                self.upscaler = Swin2SRForImageSuperResolution.from_pretrained(
                    "caidas/swin2SR-classical-sr-x2-64"
                ).to(self.device).to(self.dtype)
                
            inputs = self.processor(image, return_tensors="pt").to(self.device).to(self.dtype)
            with torch.no_grad():
                outputs = self.upscaler(**inputs)
            
            output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.moveaxis(output, 0, -1)
            return Image.fromarray((output * 255.0).round().astype(np.uint8))
            
        except Exception as e:
            print(f"Swin2SR unavailable: {e}. Using Lanczos fallback.")
            w, h = image.size
            return image.resize((w * 2, h * 2), Image.LANCZOS)

    def get_device_id(self) -> str:
        """Generate a unique identifier for the running machine."""
        import uuid
        import hashlib
        
        # Get MAC address
        mac = uuid.getnode()
        # Hash it for privacy and shortness
        return hashlib.sha256(str(mac).encode()).hexdigest()[:8]

    def apply_watermark(self, image: Image.Image, include_id: bool = True) -> Image.Image:
        """Apply invisible watermark (AI Signature + Optional Device ID) to the image."""
        try:
            import cv2
            import numpy as np
            from imwatermark import WatermarkEncoder
            
            # Message construction
            if include_id:
                dev_id = self.get_device_id()
                wm_msg = f'ESD_{dev_id}'.encode()
            else:
                wm_msg = b'ESD_ANON' # Anonymous signature
            
            print(f"[Watermark] Encoding {len(wm_msg)*8} bits: {wm_msg}")
            
            encoder = WatermarkEncoder()
            # method 'dwtDct' is robust; 'dwtDctSvd' is slower but stronger
            encoder.set_watermark('bytes', wm_msg)
            
            # Convert PIL to BGR OpenCV
            # Ensure RGB and uint8
            img_rgb = image.convert("RGB")
            img_cv = cv2.cvtColor(np.array(img_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            
            # Encode using faster algorithm (dwtDct)
            img_encoded = encoder.encode(img_cv, 'dwtDct') 
            # img_encoded = encoder.get_im_bgr() # Not needed for dwtDct 
            
            # Convert back to PIL RGB
            img_out = Image.fromarray(cv2.cvtColor(img_encoded, cv2.COLOR_BGR2RGB))
            
            # --- DEBUG: Verify immediately (Fast check) ---
            # We keep a simpler check or remove it if speed is critical.
            # Let's keep a minimal log but not break execution if it fails.
            print(f"[Watermark] Applied signature: {wm_msg}")
            # ---------------------------------
            # ---------------------------------
            
            return img_out
            
        except ImportError:
            print("Warning: 'invisible-watermark' or 'opencv-python' not installed. Skipping watermark.")
            return image
        except Exception as e:
            print(f"Watermarking failed: {e}")
            return image
