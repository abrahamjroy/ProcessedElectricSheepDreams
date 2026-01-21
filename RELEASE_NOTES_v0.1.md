# Processed Electric Sheep Dreams v0.1

First official release of Processed Electric Sheep Dreams - a native Windows application for AI image generation.

---

## What's New in v0.1

- **Text-to-Image**: Generate images from text descriptions using Z-Image-Turbo
- **Image-to-Image**: Transform existing images with text guidance
- **Inpainting**: Edit specific regions using masks (white = regenerate, black = preserve)
- **Advanced Editing**:
  - Color Matching
  - Edge Blending
  - Structure Preservation (Canny edge detection)
- **Performance**:
  - SDNQ 4-bit quantization
  - TF32 acceleration
  - AI Upscaling (2x Swin2SR)
- **UI**: AMOLED-optimized dark interface

---

## Installation (Windows)

This release is provided as a **source package with an easy launcher**.

1. **Download Source Code**: Download the `Source code (zip)` below and extract it.
2. **Install Python**: Ensure you have Python 3.10 or newer installed.
   - *Important*: Check "Add Python to environment variables" during installation.
3. **Run**: Double-click **`Launch.bat`**.

The launcher will automatically:
- Create a virtual environment
- Install all dependencies
- Download the model (~5GB) on first run
- Launch the application

---

## Requirements

- Windows 10/11
- NVIDIA GPU with 8GB+ VRAM (CUDA required)
- ~12GB Free Space

## Known Limitations

- Initial startup is slow due to model download.
- Image height/width must be divisible by 16.
