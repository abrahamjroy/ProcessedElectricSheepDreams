# Processed Electric Sheep Dreams

A native desktop application for AI image generation using a highly optimized Z-Image-Turbo model with SDNQ quantization. Supports Text-to-Image, Image-to-Image transformation, and Inpainting with mask-based editing.

---

## Screenshots

### Main Interface
![Main Interface](assets/screenshot_main.png)

### Prompt Entry with Advanced Configuration
![Prompt Entry](assets/screenshot_prompt.png)

### Generated Result
![Generated Result](assets/screenshot_result.png)

---

## Gallery

Sample images generated with Processed Electric Sheep Dreams:

<table>
  <tr>
    <td><img src="assets/gallery_1.png" width="200"/></td>
    <td><img src="assets/gallery_2.jpg" width="200"/></td>
    <td><img src="assets/gallery_3.jpg" width="200"/></td>
  </tr>
  <tr>
    <td><img src="assets/gallery_4.jpg" width="200"/></td>
    <td><img src="assets/gallery_5.jpg" width="200"/></td>
    <td></td>
  </tr>
</table>

---

## Features

### Generation Modes

- **Text-to-Image**: Generate images from text descriptions
- **Image-to-Image**: Transform existing images with text guidance
- **Inpainting**: Edit specific regions using masks (white = regenerate, black = preserve)

### Inpainting Options

- **Color Match**: Transfers color/lighting statistics from source to generated regions
- **Blend Edges**: Feathers mask boundaries for seamless transitions
- **Preserve Structure**: Maintains edge contours from the source image

### Technical Features

- SDNQ 4-bit quantization for efficient VRAM usage
- Automatic aspect ratio detection from source images
- 2x AI upscaling using Swin2SR (optional)
- Negative prompt presets for common use cases
- TF32 acceleration on Ampere+ GPUs

---

## Requirements

- Python 3.10 or higher
- CUDA-compatible GPU with 8GB+ VRAM (recommended)
- Windows, Linux, or macOS

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/processed-electric-sheep-dreams.git
cd processed-electric-sheep-dreams
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install PyTorch with CUDA support (if not already installed):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Usage

Launch the application:

```bash
python app.py
```

### Interface Overview

| Tab | Purpose |
|-----|---------|
| CREATE | Text-to-Image generation with aspect ratio presets |
| REMIX | Image-to-Image transformation and Inpainting |

### Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Prompt | Text description of desired image | Required |
| Negative Prompt | Elements to exclude from generation | Optional |
| Steps | Number of inference steps | 9 |
| Guidance Scale | CFG scale (0.0 recommended for turbo) | 0.0 |
| Strength | Transformation intensity for Img2Img | 0.40 |
| Seed | Random seed (-1 for random) | -1 |

### Inpainting Workflow

1. Switch to the REMIX tab
2. Upload a reference image
3. Upload a mask image (white areas will be regenerated)
4. Enter a prompt describing what to generate in masked areas
5. Adjust strength and enable desired post-processing options
6. Click "INITIATE RENDER"

---

## Project Structure

```
processed-electric-sheep-dreams/
├── app.py           # GUI application (ttkbootstrap)
├── backend.py       # Image generation engine
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## Model Information

This application uses the [Z-Image-Turbo](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) model with SDNQ 4-bit quantization. The model is automatically downloaded on first run (~5GB).

### Performance Notes

- First generation may be slower due to model initialization
- Generation speed depends on resolution and GPU capability
- Lower dimensions (1024x1024) generate faster than higher resolutions

---

## Troubleshooting

### Common Issues

**"Height must be divisible by 16"**
- Adjust dimensions to multiples of 16 (e.g., 1024, 1280, 1536)

**CUDA out of memory**
- Reduce output resolution
- Close other GPU-intensive applications
- The application uses CPU offload to minimize VRAM requirements

**Slow generation**
- Ensure CUDA is properly installed
- Lower the number of inference steps
- Reduce image resolution

---

## License

This project is provided as-is for educational and personal use. The underlying Z-Image-Turbo model may have its own licensing terms.

---

## Acknowledgments

- [Z-Image-Turbo](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) by Disty0
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [SDNQ](https://github.com/huggingface/sdnq) for quantization support
- [Swin2SR](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64) for upscaling
