"""
AI Concept Correction (Visual) — Using Replicate Stable Diffusion img2img

This module uses ACTUAL image-to-image generation where the uploaded sketch
IS the input reference and the output PRESERVES its structure.

Key difference from DALL-E:
- Stable Diffusion img2img has a STRENGTH parameter
- Strength 0.3 = subtle changes, preserves ~70% of original
- Strength 0.5 = moderate changes
- Strength 0.8 = major changes

This is TRUE image-to-image, not "inspired by" generation.
"""
import base64
import io
import os
import replicate
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ============================================================================
# PRODUCTION PROMPT — AI Concept Correction (Visual)
# ============================================================================

CORRECTION_PROMPT = """automotive design sketch, improved proportions, clean lines, 
correct roofline continuity, better wheel-to-body balance, improved beltline tension,
hand-sketched style, loose confident studio sketch lines, black and white sketch,
same vehicle same angle, proportional correction only"""

NEGATIVE_PROMPT = """photorealistic, 3d render, different vehicle, different angle,
colored, shaded, detailed, new design, brand logo, text, watermark"""


class CorrectionResponse(BaseModel):
    success: bool
    corrected_image_url: Optional[str] = None
    corrected_image_base64: Optional[str] = None
    message: str
    disclaimer: str = "This is an AI-generated reinterpretation of your sketch with improved proportions. It is meant for visual reference and inspiration, not exact geometric editing."


def image_to_data_uri(image_base64: str) -> str:
    """
    Convert base64 image to data URI format required by Replicate.
    """
    # If already a data URI, return as is
    if image_base64.startswith("data:"):
        return image_base64
    
    # Otherwise, construct the data URI
    return f"data:image/png;base64,{image_base64}"


def prepare_image_for_replicate(image_base64: str) -> str:
    """
    Prepare the image for Replicate API.
    Returns a data URI that Replicate can use.
    """
    # Clean up base64 data
    if "base64," in image_base64:
        image_data = image_base64.split("base64,")[1]
    else:
        image_data = image_base64
    
    # Decode and process with PIL
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (Stable Diffusion works with RGB)
    if img.mode in ('RGBA', 'P'):
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to 1024x1024 (or similar) - SD works best with specific sizes
    # Keep aspect ratio and pad
    target_size = 1024
    
    # Calculate new size maintaining aspect ratio
    ratio = min(target_size / img.width, target_size / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Create square canvas and paste centered
    canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    x_offset = (target_size - new_size[0]) // 2
    y_offset = (target_size - new_size[1]) // 2
    canvas.paste(img, (x_offset, y_offset))
    img = canvas
    
    # Convert back to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    processed_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return f"data:image/png;base64,{processed_base64}"


def generate_corrected_sketch(
    image_base64: str,
    vehicle_type: str = "vehicle",
    creativity: float = 0.5
) -> CorrectionResponse:
    """
    AI Concept Correction (Visual) — TRUE Image-to-Image
    
    Uses Replicate's Stable Diffusion img2img which ACTUALLY preserves
    the input image structure based on the strength parameter.
    
    Strength mapping:
    - creativity 0.3 = strength 0.3 = very subtle, preserves ~70% of original
    - creativity 0.5 = strength 0.4 = moderate changes
    - creativity 0.7 = strength 0.5 = noticeable changes
    """
    # Check for Replicate API token
    replicate_token = os.environ.get("REPLICATE_API_TOKEN")
    if not replicate_token:
        return CorrectionResponse(
            success=False,
            message="REPLICATE_API_TOKEN not set. Please add it to your .env file.",
            disclaimer=""
        )
    
    print("=" * 60)
    print("AI CONCEPT CORRECTION (REPLICATE STABLE DIFFUSION IMG2IMG)")
    print(f"  Creativity: {creativity}")
    print("=" * 60)
    
    try:
        # Prepare the image
        print("  Step 1: Preparing image...")
        image_data_uri = prepare_image_for_replicate(image_base64)
        
        # Map creativity to strength
        # Lower strength = more preservation of original
        # For sketch correction, we want subtle changes
        strength = min(0.5, creativity * 0.6 + 0.2)  # Range: 0.2 to 0.5
        print(f"  Strength (denoising): {strength}")
        
        print("  Step 2: Sending to Replicate Stable Diffusion img2img...")
        
        # Use SDXL img2img model
        # This model actually takes the input image and modifies it based on strength
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "image": image_data_uri,
                "prompt": CORRECTION_PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "prompt_strength": strength,  # Key parameter: how much to change
                "num_outputs": 1,
                "scheduler": "K_EULER",
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 1024,
            }
        )
        
        # Get the output URL
        if output and len(output) > 0:
            corrected_url = output[0]
            print(f"  ✅ Image corrected successfully!")
            print(f"  URL: {str(corrected_url)[:60]}...")
            
            return CorrectionResponse(
                success=True,
                corrected_image_url=str(corrected_url),
                message=f"AI Concept Correction generated (strength: {strength:.2f})",
                disclaimer="This is an AI-generated reinterpretation of your sketch with improved proportions. It is meant for visual reference and inspiration, not exact geometric editing."
            )
        else:
            return CorrectionResponse(
                success=False,
                message="No output received from Replicate",
                disclaimer=""
            )
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ❌ Error: {error_msg}")
        
        return CorrectionResponse(
            success=False,
            message=f"Failed to generate correction: {error_msg}",
            disclaimer=""
        )
