import sys
import os
import io

# CRITICAL: Redirect stdout to stderr to prevent print() statements from backend 
# from corrupting the JSON-RPC communication over stdio.
sys.stdout = sys.stderr

import contextlib
import base64
from mcp.server.fastmcp import FastMCP
from PIL import Image

# Initialize FastMCP Server
mcp = FastMCP("Z-Image-Turbo")

# Lazy loading global for the generator
generator = None

def get_generator():
    global generator
    if generator is None:
        try:
            # Import here to avoid loading heavy models just for checking available tools if not needed yet
            # Also catch any init prints
            from backend import ImageGenerator
            generator = ImageGenerator()
        except Exception as e:
            print(f"Failed to initialize ImageGenerator: {e}", file=sys.stderr)
            raise RuntimeError(f"Model initialization failed: {e}")
    return generator

@mcp.tool()
def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 9,
    seed: int = -1
) -> str:
    """
    Generate an image using Z-Image-Turbo.
    Returns the path to the saved image file.
    """
    gen = get_generator()
    
    # Generate
    image = gen.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        seed=seed
    )
    
    # Save to a temp file or a specific output folder
    # For MCP, it's often best to return a path or base64. 
    # Returning a path is more friendly for filesystem agents.
    # Let's save to a default outputs directory.
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"mcp_gen_{abs(hash(prompt))}_{seed}.png"
    filepath = os.path.join(output_dir, filename)
    
    image.save(filepath)
    print(f"Image saved to: {filepath}", file=sys.stderr)
    
    return f"Image generated and saved to: {filepath}"

@mcp.tool()
def get_device_info() -> str:
    """Returns the computing device being used (CUDA/CPU) and model info."""
    gen = get_generator()
    return f"Device: {gen.device}, Dtype: {gen.dtype}"

if __name__ == "__main__":
    # We must ensure that the original stdout is used for the MCP protocol
    # FastMCP uses the underlying file descriptor or sys.__stdout__ usually?
    # Actually, mcp library might use sys.stdout. 
    # If we redirected sys.stdout to stderr at the top, FastMCP might write to stderr too?
    # Let's check how FastMCP runs. It usually runs uvicorn or stdio loop.
    # If it is stdio, it writes to stdout.
    
    # Reverting stdout for the actual server execution, but keeping the 'backend' imports 
    # silent is tricky if they are global.
    # Strategy: 
    # 1. Redirect stdout to stderr.
    # 2. Import backend (which prints).
    # 3. Restore stdout for FastMCP.
    pass

# Refined Strategy implementation in valid code flow:
# The `sys.stdout = sys.stderr` at the top protects imports.
# But we need to restore it for `mcp.run()` if it uses stdio.

# However, FastMCP('name') creates an instance.
# To run it: `mcp.run()` handles the loop.
# Does `mcp.run()` write to sys.stdout? Yes for stdio transport.
# So we must restore sys.stdout before calling mcp.run().

# Let's fix the script structure:

import sys
import os

# Save original stdout
original_stdout = sys.stdout

# Redirect stdout to stderr to capture init logs from imports
sys.stdout = sys.stderr

try:
    from backend import ImageGenerator
except ImportError:
    # Handle case if backend isn't found (e.g. env issues)
    ImageGenerator = None

# Restore stdout for MCP communication
sys.stdout = original_stdout

from mcp.server.fastmcp import FastMCP

# Init Server
mcp = FastMCP("Z-Image-Turbo")

# Global instance holder
_generator_instance = None

def get_generator():
    global _generator_instance
    if _generator_instance is None:
        if ImageGenerator is None:
            raise ImportError("Could not import backend.ImageGenerator")
            
        # Redirect again just for init
        mk_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            _generator_instance = ImageGenerator()
        finally:
            sys.stdout = mk_stdout
            
    return _generator_instance

@mcp.tool()
def generate_image(prompt: str, width: int = 1024, height: int = 1024, steps: int = 9) -> str:
    """
    Generates an image from text.
    Args:
        prompt: The description of the image.
        width: Image width (default 1024).
        height: Image height (default 1024).
        steps: Inference steps (default 9).
    Returns:
        Absolute path to the generated image.
    """
    gen = get_generator()
    image = gen.generate(prompt=prompt, width=width, height=height, steps=steps)
    
    output_path = os.path.abspath(os.path.join(os.getcwd(), "mcp_output.png"))
    # Make unique
    import time
    timestamp = int(time.time())
    output_path = os.path.abspath(os.path.join(os.getcwd(), f"output_{timestamp}.png"))
    
    image.save(output_path)
    return output_path

@mcp.tool()
def upscale_image(image_path: str) -> str:
    """
    Upscales an existing image by 2x.
    Args:
        image_path: Absolute path to the source image.
    Returns:
        Absolute path to the upscaled image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    gen = get_generator()
    from PIL import Image
    
    # Redirect stdout during processing to prevent noise in MCP channel
    mk_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        source_img = Image.open(image_path).convert("RGB")
        upscaled = gen.upscale_image(source_img)
    finally:
        sys.stdout = mk_stdout
        
    # Save
    import time
    timestamp = int(time.time())
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.abspath(os.path.join(os.path.dirname(image_path), f"{name}_upscaled_{timestamp}{ext}"))
    
    upscaled.save(output_path)
    return output_path

@mcp.tool()
def get_device_info() -> str:
    """
    Returns information about the current inference device (CPU/CUDA) and model configuration.
    """
    gen = get_generator()
    return f"Device: {gen.device} | Dtype: {gen.dtype} | Model: Z-Image-Turbo | SDNQ: Enabled"

if __name__ == "__main__":
    mcp.run()
