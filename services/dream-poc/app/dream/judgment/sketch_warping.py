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
xai_client = None
sdxl_pipe = None

try:
    if XAI_API_KEY:
        xai_client = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1"
        )
    
    # Lazy Load Strategy to prevent startup crashes
    pass
except Exception as e:
    print(f"  [Init Error] Failed to init standard clients: {e}")

def get_sdxl_pipe():
    """Singleton accessor for SDXL Turbo Pipe"""
    global sdxl_pipe
    if sdxl_pipe is not None:
        return sdxl_pipe
        
    print("  [LazyInit] Loading SDXL Turbo...")
    try:
        from diffusers import AutoPipelineForImage2Image
        import torch
        
        device = "cpu"
        variant = None
        dtype = torch.float32
        
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
            variant = "fp16"
            print("  [LazyInit] Detected Apple Silicon (MPS). Enabling GPU acceleration.")
        elif torch.cuda.is_available():
             device = "cuda"
             dtype = torch.float16
             variant = "fp16"
             print("  [LazyInit] Detected Nvidia CUDA. Enabling GPU acceleration.")
        else:
             print("  [LazyInit] No GPU detected. Using CPU with Low-RAM offloading.")

        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=dtype,
            variant=variant
        )
        
        if device == "cpu":
             pipe.enable_model_cpu_offload()
        else:
             pipe = pipe.to(device)
             
        sdxl_pipe = pipe
        print(f"  [LazyInit] ✓ SDXL Turbo loaded (Device: {device}, Dtype: {dtype})")
        return sdxl_pipe
    except Exception as e:
        print(f"  [LazyInit] ✗ SDXL Turbo failed: {e}")
        try:
            from app.processing import log_trace
            log_trace(f"SDXL LOAD ERROR: {e}")
            import traceback
            log_trace(traceback.format_exc())
        except:
            pass
        return None


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
    
    # STAGE 1: ANALYST (Grok Vision · Geometry & Intent Extraction)
    system_prompt = """You are a senior automotive design analyst focused on maximal truthfulness.

Your role is to OBSERVE and EXTRACT STRUCTURE from the uploaded sketch image.
You do NOT redesign, improve, or correct anything. Base all observations solely on visible elements in the image.

Analyze the uploaded sketch and output ONLY factual, observable information about geometry, proportion feel, and intent.

Your responsibilities:

1. Vehicle classification
   - Identify the vehicle segment (e.g., hatchback, sedan, SUV).
   - Identify the view (side, front three-quarter, rear three-quarter).

2. Geometric anchor detection (descriptive, not numeric guessing)
   - Presence and relative position of:
     • front wheel
     • rear wheel
     • ground plane
     • roof peak
     • beltline
   - Wheel size relationship (rear vs front): smaller / similar / larger.

3. Qualitative proportion assessment
   - Wheelbase feels: short / balanced / long
   - Body height feels: tall / balanced / low
   - Greenhouse feels: heavy / balanced / light
   - Rear mass feels: heavy / balanced / light

4. Perspective discipline
   - Rear compression: sufficient / insufficient
   - Character line behavior: converging / parallel

STRICT RULES:
- Do NOT propose fixes.
- Do NOT suggest improvements.
- Do NOT invent numbers or ratios.
- Do NOT describe style, aesthetics, rendering, materials, or brand.
- Use precise, restrained language. If any element is unclear in the image, note it as 'undetectable'.

Output in structured JSON format only. No additional text.
"""

    try:
        if xai_client is None:
             print("  [Stage 1] xAI Client not initialized (Missing API Key). Skipping analysis.")
             return {}
             
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
                            "text": f"Analyze this {segment} sketch. Output structured JSON with: segment, view, anchors, proportion_feel, perspective."
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.0  # Zero temperature for factual observation
        )
        
        content = response.choices[0].message.content
        print(f"  [Stage 1: Raw Output] {content}")
        
        # Strip markdown if present
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "")
        elif "```" in content:
            content = content.replace("```", "")
        
        try:
            data = json.loads(content)
            
            # Normalize Schema (Handle Grok Variations)
            if "geometric_anchors" in data:
                data["anchors"] = data.pop("geometric_anchors")
            if "perspective_discipline" in data:
                data["perspective"] = data.pop("perspective_discipline")
            if "qualitative_proportion_assessment" in data:
                data["proportion_feel"] = data.pop("qualitative_proportion_assessment")
                
            return data
        except json.JSONDecodeError:
            print("  [Stage 1: JSON Decode Error] Failed to parse Grok output.")
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
    
    # 1. STAGE 1: Analyze with Grok (Structured Observation)
    analysis = analyze_with_grok(original_full_base64, segment)
    
    # Extract key observations
    observed_segment = analysis.get("segment")
    if not observed_segment or observed_segment == "unknown":
        observed_segment = segment
        
    observed_view = analysis.get("view")
    if not observed_view or observed_view == "unknown":
        observed_view = view
    
    proportion_feel = analysis.get("proportion_feel", {})
    anchors = analysis.get("anchors", {})
    perspective = analysis.get("perspective", {})
    
    print(f"  [Stage 1: Observations]")
    print(f"    Segment: {observed_segment}")
    print(f"    View: {observed_view}")
    print(f"    Anchors: {anchors}")
    print(f"    Proportions: {proportion_feel}")
    print(f"    Perspective: {perspective}")
    
    # 2. STAGE 2: LOGIC LAYER (Archetype Selector + CV Measurement)
    from .archetype_selector import ArchetypeSelector
    from .cv_geometry_extraction import extract_primitives_cv
    
    selector = ArchetypeSelector()
    
    # A. Get Measurements (Physical Proof)
    try:
        print("  [Stage 2: Measurement] Extracting geometry...")
        pil_img = decode_base64_to_image(original_full_base64)
        w, h = pil_img.size
        
        cv_data = extract_primitives_cv(original_full_base64, w, h)
        primitives = cv_data.get("primitives", {})
        
        # Calculate Ratios
        measurements = selector.calculate_ratios(primitives, w, h)
        print(f"  [Stage 2: Measurement] Ratios: {measurements}")
        
    except Exception as e:
        print(f"  [Stage 2: Measurement Error] {e}")
        measurements = None
    
    # B. Generate Corrections (The Decision)
    correction_modifiers = selector.get_corrections(
        segment=observed_segment, 
        view=observed_view, 
        observations=proportion_feel, # Archetype logic still needs this for now
        measurements=measurements,
        perspective=perspective # Pass perspective data to logic
    )
    
    print(f"  [Stage 2: Logic] Modifiers: {correction_modifiers}")

    # 3. STAGE 3: ARTIST (SDXL Turbo · Constrained Reconstruction)
    # Translate Text Instructions to SDXL Constraints
    
    # STAGE 3: ARTIST (SDXL Turbo · Constrained Reconstruction)
    # Using "Lean Tape Drawing" style as per Grok V10 specifications.
    # MAXIMUM Design Preservation - Explicit pixel-level instructions
    design_preservation = (
        "COPY EXACT DESIGN: headlights EXACTLY as drawn, grille EXACTLY as drawn, mirrors EXACTLY as drawn, "
        "door handles EXACTLY as drawn, character lines EXACTLY as drawn, body surfacing EXACTLY as drawn, "
        "window shape EXACTLY as drawn, all styling details EXACTLY as original sketch. "
        "ONLY move pixels to fix proportions, DO NOT redesign ANY features, DO NOT change ANY styling, "
        "DO NOT add details, DO NOT modify shapes, DO NOT invent features. "
    )
    
    stroke_preservation = (
        "PRESERVE EXACT LINE CHARACTER: Keep every line variation, every pressure change, every sketchy overlap, "
        "every construction line, every hand-drawn imperfection EXACTLY as original. "
        "DO NOT clean, DO NOT consolidate, DO NOT thicken, DO NOT smooth, DO NOT vectorize. "
    )
    
    base_positive_prompt = (
        "Adjust ONLY wheelbase length, ONLY wheel size, ONLY greenhouse height for proportion correction. "
        "Everything else stays EXACTLY as original sketch. "
        "Automotive design sketch, hand-drawn pen on white paper. "
    )
    
    prompt_parts = [
        design_preservation,  # CRITICAL: Design intent FIRST
        stroke_preservation,  # Then stroke preservation
        base_positive_prompt,
        f"Automotive sketch of a {observed_segment}",
        f"viewed from {observed_view}"
    ]
    
    if correction_modifiers:
        prompt_parts.extend(correction_modifiers)
    
    sdxl_prompt = ", ".join(prompt_parts)
    
    # MAXIMUM Anti-redesign negative prompts
    negative_prompt_parts = [
        # Explicit anti-redesign (CRITICAL)
        "redesign", "new design", "different design", "changed design", "modified design",
        "new headlights", "different headlights", "changed headlights",
        "new grille", "different grille", "changed grille",
        "new mirrors", "different mirrors", "new door handles",
        "new character lines", "different body surfacing", "new styling",
        "added details", "modified features", "invented features", "new features",
        # Anti-cleaning
        "clean lines", "uniform thickness", "vector art", "polished", "smoothed", "consolidated",
        # Standard negatives
        "photorealistic", "shading", "color", "3d render", "gradient", "shadows",
        "extra wheels", "bad anatomy", "broken lines", "messy", "blurry",
        "text", "labels", "watermark", "signature"
    ]
    sdxl_negative_prompt = ", ".join(negative_prompt_parts)
    
    temp_path = None
    try:
        # Load and pad image to square to prevent skewing
        original_image = decode_base64_to_image(image_base64)
        width, height = original_image.size
        
        # Optimization: Resize if too large
        max_side = 1024
        if width > max_side or height > max_side:
            ratio = min(max_side / width, max_side / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            original_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            width, height = new_width, new_height

        max_dim = max(width, height)
        
        # Create square canvas
        square_img = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        offset_x = (max_dim - width) // 2
        offset_y = (max_dim - height) // 2
        square_img.paste(original_image, (offset_x, offset_y))
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            square_img.save(f, format="PNG")
            temp_path = f.name
            
        print(f"  [SDXL Turbo] Running Img2Img...")
        print(f"  [Positive] {sdxl_prompt}")
        print(f"  [Negative] {sdxl_negative_prompt}")

        pipe = get_sdxl_pipe()
        if pipe is None:
            raise Exception("SDXL Turbo pipeline could not be loaded.")
        
        init_image = Image.open(temp_path)
        
        # ABSOLUTE MINIMUM strength - final attempt before geometric warping
        if strength_override is not None:
             final_strength = float(strength_override)
             print(f"  [SDXL Turbo] Strength Override: {final_strength}")
        else:
             base_strength = 0.40  # Absolute minimum for any visible changes
             
             if len(correction_modifiers) > 4:
                 base_strength = 0.45
             elif len(correction_modifiers) < 2:
                 base_strength = 0.38
                 
             final_strength = base_strength
             print(f"  [Stage 3] Final Strength: {final_strength} (Absolute Minimum)\")")
        
        print(f"  [SDXL Turbo] Running generation...")
        print(f"  [Positive] {sdxl_prompt}")
        print(f"  [Negative] {sdxl_negative_prompt}")
        
        output_square = pipe(
            prompt=sdxl_prompt,
            negative_prompt=sdxl_negative_prompt,
            image=init_image,
            num_inference_steps=4,  # Minimum for SDXL Turbo
            strength=final_strength,  # Absolute minimum: 0.38-0.45
            guidance_scale=1.0,  # Minimal guidance
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
            corrections=correction_modifiers if correction_modifiers else ["Proportions corrected (Balanced)"],
            analysis=analysis,  # Pass V2 Data
            explanation="Sketch proportions aligned to V2 Strict Logic.",
            message="Sketch corrected"
        )

    except Exception as e:
        error_msg = f"  [Img2Img Error] {e}"
        print(error_msg)
        
        # Log to debug file for Agent visibility
        with open("/Users/samarjit/.gemini/antigravity/scratch/dream-mode-repo/services/dream-poc/debug_error.log", "w") as f:
             f.write(error_msg + "\n")
             import traceback
             traceback.print_exc(file=f)
             
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
