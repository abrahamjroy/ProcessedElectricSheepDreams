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


from sdnq.loader import apply_sdnq_options_to_model
from PIL import Image, ImageOps, ImageFilter
import numpy as np


class SmartMasker:
    """Helper for automated subject/face masking."""
    def __init__(self):
        self.rembg_session = None
        self.face_cascade = None
        self.face_detector = None  # MediaPipe Tasks FaceDetector
        
    def _init_rembg(self):
        if not self.rembg_session:
            print("Loading RemBG model...")
            import rembg
            self.rembg_session = rembg.new_session()

    def _init_face(self):
        if not self.face_detector and not self.face_cascade:
            # Try MTCNN (deep learning face detector with 5 landmarks)
            try:
                from mtcnn import MTCNN
                print("[INFO] Initializing MTCNN face detector...")
                self.face_detector = MTCNN()
                self.face_cascade = None
                print("[SUCCESS] MTCNN face detector loaded (5 facial landmarks)")
                return
                
            except Exception as e:
                print(f"[WARNING] MTCNN initialization failed: {e}")
                print("[FALLBACK] Falling back to Haar Cascade...")
            
            # Fallback: Use Haar Cascade (improved with convex hull)
            import cv2
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.face_detector = None
            print("[INFO] Using Haar Cascade for face detection (improved convex hull algorithm)")

    def get_subject_mask(self, image: Image.Image) -> Image.Image:
        """Returns binary mask of the subject (White=Subject)."""
        self._init_rembg()
        import rembg
        return rembg.remove(image, session=self.rembg_session, only_mask=True)

    def get_face_mask(self, image: Image.Image, padding=15) -> Image.Image:
        """Returns binary mask of the face(s) with precise contours (White=Face)."""
        self._init_face()
        import cv2
        import numpy as np
        
        img_arr = np.array(image)
        h, w = img_arr.shape[:2]
        print(f"[DEBUG] Input image: {w}x{h}, channels={img_arr.shape[2] if len(img_arr.shape) > 2 else 1}, dtype={img_arr.dtype}")
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Try MTCNN (deep learning with 5 facial landmarks)
        if self.face_detector:
            print("[INFO] Using MTCNN for face detection...")
            
            # Ensure RGB format
            if len(img_arr.shape) == 2:  # Grayscale
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
            elif img_arr.shape[2] == 4:  # RGBA
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
                
            try:
                # MTCNN expects RGB uint8
                detections = self.face_detector.detect_faces(img_arr)
                print(f"[DEBUG] MTCNN detected {len(detections)} face(s)")
                
                if len(detections) > 0:
                    # Use largest face (highest confidence)
                    face = max(detections, key=lambda d: d['confidence'])
                    
                    box = face['box']  # [x, y, width, height]
                    keypoints = face['keypoints']  # dict with left_eye, right_eye, nose, mouth_left, mouth_right
                    confidence = face['confidence']
                    
                    print(f"[SUCCESS] MTCNN detected face with {confidence*100:.1f}% confidence")
                    
                    # Create face contour from bounding box + landmarks
                    x, y, fw, fh = box
                    
                    # Build high-resolution face contour with interpolated points
                    # Using more points for smoother curves
                    contour_points = []
                    
                    # Extract landmark positions
                    left_eye = keypoints['left_eye']
                    right_eye = keypoints['right_eye']
                    nose = keypoints['nose']
                    mouth_left = keypoints['mouth_left']
                    mouth_right = keypoints['mouth_right']
                    
                    center_x = x + fw//2
                    
                    # RIGHT SIDE (top to bottom) - 8 points
                    contour_points.append([center_x, y])  # Top forehead center
                    contour_points.append([int(center_x + fw*0.25), y])  # Top right forehead
                    contour_points.append([int(x + fw*0.9), int(y + fh*0.1)])  # Right forehead corner
                    contour_points.append([x + fw, int(right_eye[1] - 10)])  # Above right eye
                    contour_points.append([x + fw, int(right_eye[1])])  # Right eye level
                    contour_points.append([int(x + fw*0.95), int((right_eye[1] + mouth_right[1])/2)])  # Right cheek upper
                    contour_points.append([int(mouth_right[0]) + 15, int(mouth_right[1])]) # Right mouth area
                    contour_points.append([int(x + fw*0.85), int(y + fh*0.9)])  # Right jaw
                    
                    # BOTTOM (right to left) - 5 points
                    contour_points.append([int(x + fw*0.65), y + fh])  # Right chin
                    contour_points.append([center_x, int(y + fh*1.05)])  # Center chin (slightly below)
                    contour_points.append([int(x + fw*0.35), y + fh])  # Left chin
                    
                    # LEFT SIDE (bottom to top) - 8 points
                    contour_points.append([int(x + fw*0.15), int(y + fh*0.9)])  # Left jaw
                    contour_points.append([int(mouth_left[0]) - 15, int(mouth_left[1])])  # Left mouth area
                    contour_points.append([int(x + fw*0.05), int((left_eye[1] + mouth_left[1])/2)])  # Left cheek upper
                    contour_points.append([x, int(left_eye[1])])  # Left eye level
                    contour_points.append([x, int(left_eye[1] - 10)])  # Above left eye
                    contour_points.append([int(x + fw*0.1), int(y + fh*0.1)])  # Left forehead corner
                    contour_points.append([int(center_x - fw*0.25), y])  # Top left forehead
                    
                    contour = np.array(contour_points, dtype=np.int32)
                    
                    # Create convex hull for smooth contour
                    hull = cv2.convexHull(contour)
                    
                    # Draw filled polygon
                    cv2.fillPoly(mask, [hull], 255)
                    
                    # Smooth the mask edges
                    # 1. Apply Gaussian blur to soften edges
                    mask = cv2.GaussianBlur(mask, (15, 15), 0)
                    # 2. Threshold back to binary (removes gray pixels from blur)
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    # 3. Apply morphological closing to smooth further
                    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth)
                    
                    # Apply padding via dilation
                    if padding > 0:
                        kernel = np.ones((padding*2, padding*2), np.uint8)
                        mask = cv2.dilate(mask, kernel, iterations=1)
                        
                    print(f"[SUCCESS] Created MTCNN face mask with {len(contour_points)} contour points")
                    
                    # Save debug image
                    try:
                        debug_img = img_arr.copy()
                        # Draw bounding box
                        cv2.rectangle(debug_img, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                        # Draw landmarks
                        for name, (px, py) in keypoints.items():
                            cv2.circle(debug_img, (int(px), int(py)), 3, (255, 0, 0), -1)
                        # Draw contour
                        cv2.polylines(debug_img, [hull], True, (0, 255, 255), 2)
                        cv2.putText(debug_img, f"MTCNN {confidence*100:.0f}%", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        Image.fromarray(debug_img).save("debug_mtcnn_detection.png")
                        print("[DEBUG] Saved debug_mtcnn_detection.png")
                    except Exception as e:
                        print(f"[WARNING] Could not save debug image: {e}")
                else:
                    print("[WARNING] MTCNN detected 0 faces")
                    print("[FALLBACK] Using approximate center region")
                    center = (w // 2, int(h * 0.25))
                    axes = (int(w * 0.18), int(h * 0.22))
                    cv2.ellipse(mask, center, axes, 0, 0, 360, (255), -1)
                    
            except Exception as e:
                print(f"[ERROR] MTCNN detection failed: {type(e).__name__}: {e}")
                print(f"[FALLBACK] Using approximate center region")
                center = (w // 2, int(h * 0.25))
                axes = (int(w * 0.18), int(h * 0.22))
                cv2.ellipse(mask, center, axes, 0, 0, 360, (255), -1)
                
                
        # 2. Fallback to Haar Cascade (improved contour approximation)
        elif self.face_cascade:
            print("[FALLBACK] Using Haar Cascade for face detection")
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
            
            # Load eye cascade for validation
            try:
                eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
                eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            except:
                eye_cascade = None
            
            # Try multiple detection passes with increasingly relaxed parameters
            faces = []
            detection_params = [
                {"scaleFactor": 1.05, "minNeighbors": 5, "minSize": (30, 30)},  # Strict
                {"scaleFactor": 1.1, "minNeighbors": 3, "minSize": (30, 30)},   # Medium
                {"scaleFactor": 1.2, "minNeighbors": 2, "minSize": (20, 20)},   # Relaxed
            ]
            
            valid_faces = []
            for idx, params in enumerate(detection_params):
                faces = self.face_cascade.detectMultiScale(gray, **params)
                print(f"[DEBUG] Pass {idx+1}: scaleFactor={params['scaleFactor']}, minNeighbors={params['minNeighbors']} -> Found {len(faces)} faces")
                
                if len(faces) > 0:
                    # Validate faces - must be in upper portion of image
                    for (x, y, fw, fh) in faces:
                        face_center_y = y + fh//2
                        is_upper_region = face_center_y < h * 0.6  # Face should be in upper 60% of image
                        
                        # Optional: Check for eyes within face region
                        has_eyes = False
                        if eye_cascade is not None:
                            face_roi = gray[y:y+fh, x:x+fw]
                            eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
                            has_eyes = len(eyes) >= 1
                            print(f"[DEBUG] Face at ({x},{y}) {fw}x{fh}, eyes detected: {len(eyes)}")
                        
                        if is_upper_region or has_eyes:
                            valid_faces.append((x, y, fw, fh))
                            print(f"[VALID] Face at ({x},{y}) - upper_region: {is_upper_region}, has_eyes: {has_eyes}")
                    
                    if len(valid_faces) > 0:
                        break
            
            if len(valid_faces) > 0:
                largest_face = max(valid_faces, key=lambda f: f[2] * f[3])
                (x, y, fw, fh) = largest_face
                print(f"[SUCCESS] Detected face at ({x}, {y}) with size {fw}x{fh}")
                
                # Create more organic shape using multiple points (improved over simple ellipse)
                # Approximate facial contour with key points
                center_x = x + fw//2
                center_y = y + fh//2
                
                # Generate contour points approximating a face shape
                points = []
                # Top of head
                points.append([center_x, y])
                # Temples
                points.append([x + int(fw*0.15), y + int(fh*0.15)])
                points.append([x + int(fw*0.85), y + int(fh*0.15)])
                # Cheeks
                points.append([x + fw, y + int(fh*0.5)])
                points.append([x + int(fw*0.9), y + int(fh*0.8)])
                # Jaw/chin
                points.append([center_x, y + fh])
                points.append([x + int(fw*0.1), y + int(fh*0.8)])
                points.append([x, y + int(fh*0.5)])
                
                # Create convex hull
                contour = np.array(points, dtype=np.int32)
                hull = cv2.convexHull(contour)
                
                # Draw filled polygon
                cv2.fillPoly(mask, [hull], 255)
                
                # Apply padding via dilation
                if padding > 0:
                    kernel = np.ones((padding*2, padding*2), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                
                # Save debug visualization
                try:
                    debug_img = img_arr.copy()
                    cv2.rectangle(debug_img, (x, y), (x+fw, y+fh), (0, 255, 0), 3)
                    cv2.polylines(debug_img, [hull], True, (255, 0, 0), 3)
                    cv2.putText(debug_img, "FACE DETECTED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    Image.fromarray(debug_img).save("debug_haar_detection.png")
                    print("[DEBUG] Saved debug_haar_detection.png")
                except:
                    pass
                    
                print(f"[INFO] Created improved face mask using Haar Cascade + convex hull")
            else:
                print("[WARNING] Haar Cascade detected 0 valid faces across all parameter sets")
                print(f"[INFO] Image size: {w}x{h}, detected {len(faces)} candidates but none were valid")
                
                # Save debug image showing why detection failed
                try:
                    debug_img = img_arr.copy()
                    # Draw all rejected candidates in red
                    for (x, y, fw, fh) in faces:
                        cv2.rectangle(debug_img, (x, y), (x+fw, y+fh), (0, 0, 255), 2)
                        cv2.putText(debug_img, "REJECTED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(debug_img, f"NO VALID FACE ({len(faces)} rejected)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    Image.fromarray(debug_img).save("debug_no_face_detected.png")
                    print("[DEBUG] Saved debug_no_face_detected.png")
                except:
                    pass
                    
                print("[FALLBACK] Using approximate center region")
                center = (w // 2, int(h * 0.25))
                axes = (int(w * 0.15), int(h * 0.20))
                cv2.ellipse(mask, center, axes, 0, 0, 360, (255), -1)

        return Image.fromarray(mask)


class ImageGenerator:
    """
    Core image generation class wrapping ZImagePipeline.
    Automatically selects optimal device and precision settings.
    """
    
    def __init__(self, model_id: str = "Tongyi-MAI/Z-Image-Turbo"):
        self.model_id = model_id
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
        # Only apply for quantized models (those with SDNQ in name)
        if self.device == "cuda" and triton_is_available and "SDNQ" in model_id:
            print("Applying SDNQ optimizations...")
            self.pipe.transformer = apply_sdnq_options_to_model(
                self.pipe.transformer, use_quantized_matmul=True
            )
            self.pipe.text_encoder = apply_sdnq_options_to_model(
                self.pipe.text_encoder, use_quantized_matmul=True
            )

        # Enable CPU offload for efficient VRAM management
        self.pipe.enable_model_cpu_offload()
        
        # Init Smart Masker
        self.smart_masker = SmartMasker()

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

    def generate_smart(
        self,
        prompt: str,
        image: Image.Image,
        mode: str = "outfit", # "outfit" or "bg"
        steps: int = 9,
        seed: int = -1,
        strength: float = 0.8, # Higher strength needed for big changes
        guidance: float = 0.0
    ):
        """
        Smart Remix generation.
        mode="outfit": Changes clothes, keeps face and BG.
        mode="bg": Changes BG, keeps subject.
        """
        print(f"Smart Remix Mode: {mode}")
        width, height = image.size
        
        # 1. Get Subject Mask (White=Person, Black=BG)
        subject_mask = self.smart_masker.get_subject_mask(image)
        
        # 2. Refine Mask based on Mode
        final_mask = None
        
        if mode == "bg":
            # For BG change, we want to regenerate Background (Black in subject mask)
            # So Inpaint Mask should be White for BG, Black for Person.
            # Subject mask is White=Person. So we Invert it.
            final_mask = ImageOps.invert(subject_mask)
            
        elif mode == "outfit":
            # For Outfit: Regenerate Body (White), Keep Face (Black), Keep BG (Black).
            # Start with Subject (White body+face), Black BG.
            # Get Face Mask (White=Face)
            face_mask = self.smart_masker.get_face_mask(image)
            
            # Subtract Face from Subject using Numpy for precision
            sub_arr = np.array(subject_mask)
            face_arr = np.array(face_mask)
            
            # Ensure binary (0 or 255)
            # Thresholding 
            sub_bin = sub_arr > 127
            face_bin = face_arr > 127
            
            # Logic: We want output to be White where Subject is True AND Face is False.
            # Output = Subject AND (NOT Face)
            outfit_bool = np.logical_and(sub_bin, np.logical_not(face_bin))
            
            # Convert back to image
            final_mask = Image.fromarray((outfit_bool * 255).astype(np.uint8))
            
        return final_mask

    def preview_smart_mask(self, image: Image.Image, mode: str = "outfit") -> Image.Image:
        """
        Generate only the mask for preview purposes.
        """
        print(f"Previewing Mask Mode: {mode}")
        width, height = image.size
        
        try:
            # 1. Get Subject Mask (White=Person, Black=BG)
            subject_mask = self.smart_masker.get_subject_mask(image)
            
            final_mask = None
            
            if mode == "bg":
                final_mask = ImageOps.invert(subject_mask)
            elif mode == "outfit":
                face_mask = self.smart_masker.get_face_mask(image)
                
                # Logic repeated from generate_smart for consistency
                import numpy as np
                sub_arr = np.array(subject_mask)
                face_arr = np.array(face_mask)
                
                sub_bin = sub_arr > 127
                face_bin = face_arr > 127
                
                # Output = Subject AND (NOT Face)
                outfit_bool = np.logical_and(sub_bin, np.logical_not(face_bin))
                
                final_mask = Image.fromarray((outfit_bool * 255).astype(np.uint8))
                
            if final_mask:
                return final_mask
            return Image.new("L", (width, height), 0) # Empty fallback
            
        except Exception as e:
            print(f"Mask Preview Error: {e}")
            return Image.new("L", (width, height), 128) # Grey error mask

    def generate_smart(
        self,
        prompt: str,
        image: Image.Image,
        mode: str = "outfit", # "outfit" or "bg"
        steps: int = 9,
        seed: int = -1,
        strength: float = 0.8, # Higher strength needed for big changes
        guidance: float = 0.0
    ):
        """
        Smart Remix generation.
        mode="outfit": Changes clothes, keeps face and BG.
        mode="bg": Changes BG, keeps subject.
        """
        print(f"Smart Remix Mode: {mode}")
        width, height = image.size
        
        # 1. Get Subject Mask (White=Person, Black=BG)
        subject_mask = self.smart_masker.get_subject_mask(image)
        
        # 2. Refine Mask based on Mode
        final_mask = None
        
        if mode == "bg":
            # For BG change, we want to regenerate Background (Black in subject mask)
            # So Inpaint Mask should be White for BG, Black for Person.
            # Subject mask is White=Person. So we Invert it.
            from PIL import ImageOps
            final_mask = ImageOps.invert(subject_mask)
            
        elif mode == "outfit":
            # For Outfit: Regenerate Body (White), Keep Face (Black), Keep BG (Black).
            face_mask = self.smart_masker.get_face_mask(image)
            
            # Use numpy for precise boolean mask logic
            import numpy as np
            sub_arr = np.array(subject_mask)
            face_arr = np.array(face_mask)
            
            # Threshold to binary
            sub_bin = sub_arr > 127
            face_bin = face_arr > 127
            
            # Logic: Body = Subject AND NOT(Face)
            # White where body (to regenerate), Black where face (to preserve)
            outfit_bool = np.logical_and(sub_bin, np.logical_not(face_bin))
            
            final_mask = Image.fromarray((outfit_bool * 255).astype(np.uint8))
        
        if final_mask is None:
             # Fallback
             final_mask = Image.new("L", (width, height), 255)

        # 3. Call standard inpaint
        # Make sure mask is L mode
        final_mask = final_mask.convert("L")
        
        # DEBUG: Save masks to verify logic
        try:
            print("Saving Debug Masks...")
            subject_mask.save("debug_mask_subject.png")
            if 'face_mask' in locals(): face_mask.save("debug_mask_face.png")
            final_mask.save("debug_mask_final.png")
        except Exception as e:
            print(f"Debug Save Warning: {e}")
        
        return self.generate(
            prompt=prompt,
            image=image,
            mask_image=final_mask,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            strength=strength,
            guidance_scale=guidance, 
            blend_edges=True, # Critical for mask blending
            color_match=True # Critical for lighting consistency
        )

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
        """Handle Inpainting generation using custom manual compositing approach."""
        print("Using Manual Inpainting (Base Pipeline + Compositing)...")
        
        # Resize inputs to target dimensions
        resized_image = image.resize((width, height), Image.LANCZOS)
        resized_mask = mask_image.convert("L").resize((width, height), Image.NEAREST)
        
        # Initialize img2img pipeline if not already done
        if not hasattr(self, 'img2img_pipe'):
            print("Initializing Img2Img Pipeline for Manual Inpainting...")
            from diffusers import AutoPipelineForImage2Image
            self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
        
        # Use the img2img pipeline to transform the entire image
        args["image"] = resized_image
        args["strength"] = strength
        
        # Generate the full transformed image
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
        
        # DEBUG: Save all intermediate images to trace the issue
        try:
            resized_image.save("debug_original_resized.png")
            generated.save("debug_generated.png")
            composite_mask.save("debug_composite_mask.png")
            print(f"DEBUG: Mask stats - min={composite_mask.getextrema()[0]}, max={composite_mask.getextrema()[1]}")
        except Exception as e:
            print(f"Debug save error: {e}")
        
        # Composite: generated where mask is WHITE, original where mask is BLACK
        # This should keep the original face (black area) and use generated body (white area)
        result = Image.composite(generated, resized_image, composite_mask)
        
        try:
            result.save("debug_final_result.png")
        except: pass
        
        return result

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
