"""
Antigravity V9 — SDXL Turbo CPU (Grok + SDXL Turbo Img2Img)

The FINAL Working Solution:
- `Manjushri/SDXL-Turbo-Img2Img-CPU` Space is ALIVE and supports Img2Img.
- Uses CPU (Stable, won't crash/pause).
- Uses SDXL Turbo (Fast inference even on CPU).

Workflow:
1. Grok Vision (User Key) -> Analyzes detailed design.
2. SDXL Turbo Img2Img (Gradio) -> Apply corrections.
   - Strength: 0.6 (Balanced).
   - Steps: 3 (Turbo sweet spot).
"""
import base64
import json
import os
import io
import time
import tempfile
from typing import Dict, List, Optional
from openai import OpenAI
from pydantic import BaseModel
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
import requests  # For HF Inference API calls

load_dotenv()

# API Keys
XAI_API_KEY = os.getenv("XAI_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Clients
try:
    xai_client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    ) if XAI_API_KEY else None
    
    print("  [Init] Loading SDXL Turbo pipeline (diffusers)...")
    # Use diffusers library locally - most reliable approach
    # This runs SDXL Turbo directly on the server
    try:
        from diffusers import AutoPipelineForImage2Image
        import torch
        
        sdxl_pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16" if torch.cuda.is_available() else None
        )
        
        # Move to GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sdxl_pipe = sdxl_pipe.to(device)
        print(f"  [Init] ✓ SDXL Turbo loaded on {device}")
    except Exception as pipe_err:
        print(f"  [Init] ✗ SDXL Turbo pipeline failed: {pipe_err}")
        sdxl_pipe = None
    
except Exception as e:
    print(f"  [Init] General client init error: {e}")
    xai_client = None


class CorrectionResponse(BaseModel):
    success: bool
    corrected_image_base64: Optional[str] = None
    original_image_base64: Optional[str] = None
    analysis: Optional[Dict] = None
    corrections: List[str] = []
    explanation: Optional[str] = None
    message: str


def decode_base64_to_bytes(image_base64: str) -> bytes:
    if "base64," in image_base64:
        image_data = image_base64.split("base64,")[1]
    else:
        image_data = image_base64
    return base64.b64decode(image_data)


def decode_base64_to_image(image_base64: str) -> Image.Image:
    if "base64," in image_base64:
        image_data = image_base64.split("base64,")[1]
    else:
        image_data = image_base64
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    img.load()  # Force load to prevent "AttributeError: _im" when IO is closed
    return img


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


def analyze_with_grok(image_base64: str, segment: str) -> dict:
    """
    STAGE 1 — ANALYST (Grok Vision · Geometry & Intent Extraction)
    
    Role: OBSERVE and EXTRACT STRUCTURE. Do NOT redesign, improve, or correct.
    Output: Structured JSON with factual observations only.
    """
    if not xai_client:
        # Fallback: minimal observation
        return {
            "segment": segment,
            "view": "unknown",
            "proportion_feel": {"wheelbase": "balanced", "body_height": "balanced", "greenhouse": "balanced"},
            "confidence": 0.3
        }
        
    print("  [Stage 1: Analyst] Observing sketch geometry...")
    
    system_prompt = """You are a senior automotive design analyst.

Your role is to OBSERVE and EXTRACT STRUCTURE.
You do NOT redesign, improve, or correct anything.

Analyze the uploaded sketch and output ONLY factual, observable information about geometry, proportion feel, and intent.

Your responsibilities:

1. Vehicle classification
   - Identify the vehicle segment (e.g. hatchback, sedan, SUV, coupe).
   - Identify the view (side, front_3q, rear_3q).

2. Geometric anchor detection (descriptive, not numeric guessing)
   - Presence and relative position of:
     • front wheel
     • rear wheel
     • beltline

3. Qualitative proportion assessment
   - Wheelbase feels: short / balanced / long
   - Body height feels: tall / balanced / low
   - Greenhouse feels: heavy / balanced / light

4. STYLE & INTENT (NEW):
   - Extract 3 distinct style keywords (e.g., "minimalist", "angular", "organic", "aggressive", "sketchy", "precise").
   - Keep it to single adjectives.

STRICT RULES:
- Do NOT propose fixes.
- Do NOT suggest improvements.
- Do NOT invent numbers or ratios.
- Do NOT describe materials or brand.
- Use precise, restrained language.

Output in structured JSON format only."""

    try:
        response = xai_client.chat.completions.create(
            model="grok-2-vision-1212",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_base64}
                        },
                        {
                            "type": "text",
                            "text": f"Analyze this {segment} sketch. Output structured JSON with: segment, view, style_keywords, proportion_feel."
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.1  # Low temperature for factual observation
        )
        
        content = response.choices[0].message.content
        print(f"  [Stage 1: Raw Output] {content}")
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}
        
    except Exception as e:
        print(f"  [Stage 1: API Error] {e}")
        # Fallback observation
        return {}


def correct_sketch(image_base64: str, segment: str = "hatchback", view: str = "front_3q", strength_override: Optional[float] = None) -> CorrectionResponse:
    print("=" * 60)
    print("ANTIGRAVITY V10 — STAGE 1 ANALYST (Grok + SDXL Turbo)")
    print("=" * 60)

    original_full_base64 = image_base64 if image_base64.startswith("data:") else f"data:image/png;base64,{image_base64}"
    
    # ... (rest of function until strength calculation)
    
    # 1. STAGE 1: Analyze with Grok (Structured Observation)
    analysis = analyze_with_grok(original_full_base64, segment)
    
    # Extract key observations
    # Extract key observations, defaulting to UI inputs if AI is unsure/fails
    observed_segment = analysis.get("segment")
    if not observed_segment or observed_segment == "unknown":
        observed_segment = segment
        
    observed_view = analysis.get("view")
    if not observed_view or observed_view == "unknown":
        observed_view = view
    
    proportion_feel = analysis.get("proportion_feel", {})
    style_keywords = analysis.get("style_keywords", [])
    
    print(f"  [Stage 1: Observations]")
    print(f"    Segment: {observed_segment}")
    print(f"    View: {observed_view}")
    print(f"    Style: {style_keywords}")
    print(f"    Proportions: {proportion_feel}")
    
    # 2. STAGE 2: LOGIC LAYER (Archetype Selector + CV Measurement)
    # Replaces hardcoded prompt logic with data-driven guardrails
    from .archetype_selector import ArchetypeSelector
    from .cv_geometry_extraction import extract_primitives_cv
    
    selector = ArchetypeSelector()
    
    # A. Get Measurements (Physical Proof)
    # We call the CV module to get actual geometry points
    # Passing the base64 image directly
    try:
        print("  [Stage 2: Measurement] Extracting geometry...")
        # Need to know image size for CV extraction? The function decodes base64 itself.
        # But `extract_primitives_cv` wants width/height arguments?
        # Let's check signature: def extract_primitives_cv(image_base64: str, image_width: int, image_height: int) -> Dict:
        # We need to get size first.
        pil_img = decode_base64_to_image(original_full_base64)
        w, h = pil_img.size
        
        cv_data = extract_primitives_cv(original_full_base64, w, h)
        primitives = cv_data.get("primitives", {})
        
        # Calculate Ratios
        measurements = selector.calculate_ratios(primitives, w, h)
        print(f"  [Stage 2: Measurement] Ratios: {measurements}")
        
    except Exception as e:
        print(f"  [Stage 2: Measurement Error] {e}")
        measurements = None  # Fallback to pure qualitative
    
    # B. Generate Corrections (The Decision)
    correction_modifiers = selector.get_corrections(
        segment=observed_segment, 
        view=observed_view, 
        observations=proportion_feel,
        measurements=measurements
    )
    
    print(f"  [Stage 2: Logic] Modifiers: {correction_modifiers}")

    # Build constraint-aware prompt
    prompt_parts = [
        f"Professional automotive sketch of a {observed_segment}",
        f"viewed from {observed_view.replace('_', ' ')}",
    ]
    
    # Inject Style Keywords (Design Intent Preservation)
    if style_keywords:
        style_desc = ", ".join(style_keywords)
        prompt_parts.append(f"style: {style_desc}")
        
    prompt_parts.append("with corrected proportions:")
    
    # Add the Logic Layer's strict corrections
    if correction_modifiers:
        prompt_parts.extend(correction_modifiers)
    else:
        prompt_parts.append("balanced proportions")
    
    # REMOVED: "minimalistic hand-drawn style" to prevent style overwrite
    prompt_parts.append("Clean black lines on white paper.")
    
    sdxl_prompt = ", ".join(prompt_parts)
    
    temp_path = None
    try:
        # Load and pad image to square to prevent skewing
        original_image = decode_base64_to_image(image_base64)
        width, height = original_image.size
        
        # Optimization: Resize if too large to prevent timeouts on CPU
        max_side = 1024
        if width > max_side or height > max_side:
            ratio = min(max_side / width, max_side / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            original_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"  [Optimization] Resized input from {width}x{height} to {new_width}x{new_height}")
            width, height = new_width, new_height

        max_dim = max(width, height)
        
        # Create square canvas
        square_img = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        # Center original image
        offset_x = (max_dim - width) // 2
        offset_y = (max_dim - height) // 2
        square_img.paste(original_image, (offset_x, offset_y))
        
        # Save temp file for Gradio
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            square_img.save(f, format="PNG")
            temp_path = f.name
            
        
        
        
        print(f"  [SDXL Turbo] Running Img2Img with local pipeline...")
        print(f"  [Prompt] {sdxl_prompt}")

        # Validate pipeline is available
        if sdxl_pipe is None:
            raise Exception("SDXL Turbo pipeline is not loaded.")
        
        # Load the square image for processing
        init_image = Image.open(temp_path)
        
        # Run SDXL Turbo img2img
        print(f"  [SDXL Turbo] Generating with local pipeline...")
        # Logic-Driven Dynamic Strength
        # If the Logic Layer requested strict changes (modifiers exist), we boost strength to force the warp.
        # If no modifiers (Balanced/Safe), we use lower strength to just clean/denoise.
        
        if strength_override is not None:
             # MENTOR MODE: User overrides the logic
             dynamic_strength = float(strength_override)
             print(f"  [SDXL Turbo] Strength Override: {dynamic_strength} (User Control)")
        else:
             # AUTO MODE: "Golden Ratio" logic
             # TUNING: Adjusted to 0.665 per user request (The "Golden Ratio" of stability vs correction)
             dynamic_strength = 0.665 if correction_modifiers else 0.55
             
        print(f"  [SDXL Turbo] Final Strength: {dynamic_strength} (Modifiers: {len(correction_modifiers)})")
        
        output_square = sdxl_pipe(
            prompt=sdxl_prompt,
            negative_prompt="photorealistic, shading, color, 3d render, extra wheels, bad anatomy, distorted, messy, blurry",
            image=init_image,
            num_inference_steps=3,  # Turbo uses very few steps
            strength=dynamic_strength,
            guidance_scale=0.0,     # Turbo doesn't use guidance
        ).images[0]
        
        print("  [SDXL Turbo] Success!")
        
        # Crop back to original aspect ratio
        sq_width, sq_height = output_square.size
        if sq_width != max_dim:
            # Calculate new relative offsets
            scale = sq_width / max_dim
            crop_x = int(offset_x * scale)
            crop_y = int(offset_y * scale)
            crop_w = int(width * scale)
            crop_h = int(height * scale)
        else:
            crop_x, crop_y, crop_w, crop_h = offset_x, offset_y, width, height
            
        final_image = output_square.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
            
        
        corrected_base64 = image_to_base64(final_image)
        
        # Clean up
        os.unlink(temp_path)
        
        return CorrectionResponse(
            success=True,
            corrected_image_base64=corrected_base64,
            original_image_base64=original_full_base64,
            corrections=["Proportions corrected via SDXL Turbo (HF Inference API)"],
            explanation="Sketch proportions improved using SDXL Turbo Img2Img via Hugging Face Inference API.",
            message="Sketch corrected"
        )

    except Exception as e:
        print(f"  [Img2Img Error] {e}")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        import traceback
        traceback.print_exc()
        return CorrectionResponse(
            success=False,
            original_image_base64=original_full_base64,
            message=f"Correction failed: {str(e)}"
        )


def warp_sketch(image_base64: str, segment: str, view: str = "front_3q", strength_override: Optional[float] = None):
    """Wrapper for existing API."""
    result = correct_sketch(image_base64, segment, view, strength_override=strength_override)
    
    class WarpResponse(BaseModel):
        success: bool
        warped_image_base64: Optional[str] = None
        comparison_image_base64: Optional[str] = None
        message: str
        corrections: List[str] = []
        analysis: Optional[Dict] = None
        confidence: Optional[float] = None
    
    return WarpResponse(
        success=result.success,
        warped_image_base64=result.corrected_image_base64,
        comparison_image_base64=None,
        message=result.explanation or result.message,
        corrections=result.corrections,
        analysis=result.analysis,
        confidence=0.85 if result.success else 0.0
    )
