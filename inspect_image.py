
import sys
import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo

def inspect_image(image_path):
    print(f"\n--- Inspecting: {image_path} ---\n")
    
    if not os.path.exists(image_path):
        print("Error: File not found.")
        return

    try:
        img = Image.open(image_path)
        img.load()
        
        # 1. Print Metadata (PNG Info)
        print("--- [1] PNG Metadata ---")
        if img.info:
            for key, value in img.info.items():
                # Filter out standard keys if verbose or just show all
                print(f"{key}: {value}")
        else:
            print("No metadata found.")
            
        # 2. Decode Invisible Watermark
        print("\n--- [2] Invisible Watermark (ESD) ---")
        try:
            import cv2
            import numpy as np
            from imwatermark import WatermarkDecoder
            
            # Convert to BGR for OpenCV
            bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Scan for different bit lengths
            found = False
            # Check 64 bits (ESD_ANON) to 128 bits in steps of 8
            for bits in range(64, 129, 8): 
                try:
                    decoder = WatermarkDecoder('bytes', bits)
                    # Use Faster Algorithm (dwtDct) to match backend
                    watermark = decoder.decode(bgr_img, 'dwtDct')
                    if watermark:
                        try:
                            decoded_str = watermark.decode('utf-8')
                            if all(32 <= ord(c) <= 126 for c in decoded_str):
                                print(f"Detected (Length {bits} bits): {decoded_str}")
                                if decoded_str.startswith('ESD_'):
                                    print("-> Valid Signature Found!")
                                    found = True
                        except:
                            pass
                except Exception:
                    continue
            
            if not found:
                 print("No valid ESD watermark detected.")

        except ImportError:
            print("Watermark decoder libraries (imwatermark, opencv-python) not found.")
        except Exception as e:
            print(f"Watermark decoding error: {e}")

    except Exception as e:
        print(f"Failed to load image: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_image.py <path_to_image.png>")
    else:
        inspect_image(sys.argv[1])
