"""
DREAM V3 — Correction Stroke Engine
COMPLETE IMPLEMENTATION per User Specification

Core Principle:
"The AI redraws only the lines it disagrees with, as correction strokes,
on top of the original sketch."

"The original sketch raster must never be altered, regenerated, or replaced."
"""
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
try:
    client = OpenAI()
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    client = None


# ============================================================================
# SCHEMAS — EXACT per User Specification
# ============================================================================

class CurveStyle(BaseModel):
    color: str = "#00FF00"
    thickness: float = 2.0
    opacity: float = 0.85


class OriginalRange(BaseModel):
    start_t: float
    end_t: float


class NewCurve(BaseModel):
    curve_type: str  # "bezier" or "polyline"
    points: List[List[float]]


class WheelGeometry(BaseModel):
    cx: float
    cy: float
    r: float


class CorrectionStroke(BaseModel):
    """Exact schema per user specification"""
    id: str
    element: str  # "roofline", "beltline", "front_wheel", "rear_wheel", etc.
    stroke_type: str  # "replacement_curve", "radius_adjustment", "polyline_replacement"
    
    # For curve corrections
    original_range: Optional[OriginalRange] = None
    new_curve: Optional[NewCurve] = None
    
    # For polyline replacement (beltline)
    new_points: Optional[List[List[float]]] = None
    
    # For wheel corrections
    original: Optional[WheelGeometry] = None
    corrected: Optional[WheelGeometry] = None
    
    # Style
    style: CurveStyle = CurveStyle()
    
    # Metadata
    confidence: float
    reason: str
    
    # Required additions per user spec
    affects_original_visibility: str = "overlay_only"
    stroke_role: str = "primary_correction"


class CorrectionMeta(BaseModel):
    confidence_overall: float
    intent_preserved: bool
    notes: str


class CorrectionStrokeResponse(BaseModel):
    correction_strokes: List[CorrectionStroke]
    meta: CorrectionMeta
    error: Optional[str] = None


# ============================================================================
# GEOMETRY EXTRACTION — Step 1
# ============================================================================

EXTRACTION_PROMPT = """You are a geometry extraction system for automotive sketches.
Extract geometric primitives with NORMALIZED coordinates (0-1).

COORDINATE SYSTEM:
- x=0 is LEFT edge, x=1 is RIGHT edge
- y=0 is TOP edge, y=1 is BOTTOM edge
- Radius is relative to image WIDTH

CRITICAL: Trace the ACTUAL lines in the sketch as precisely as possible.
Polylines must be ordered from FRONT → REAR and parametrizable (t ∈ [0,1]).

OUTPUT FORMAT (JSON only):
{
  "vehicle_type": "hatchback",
  "view": "side",
  "primitives": {
    "roofline": {
      "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "parametric_order": "front_to_rear"
    },
    "beltline": {
      "points": [[x1,y1], [x2,y2], [x3,y3]],
      "parametric_order": "front_to_rear"
    },
    "body_lower_edge": {
      "points": [[x1,y1], [x2,y2], [x3,y3]],
      "parametric_order": "front_to_rear"
    },
    "greenhouse": {
      "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "parametric_order": "front_to_rear"
    },
    "front_wheel": {"cx": 0.25, "cy": 0.72, "r": 0.08},
    "rear_wheel": {"cx": 0.68, "cy": 0.73, "r": 0.085},
    "ground_plane_y": 0.85
  }
}

RULES:
- Roofline: Sample 4 points along the actual roof curve (Bézier control points)
- Beltline: Sample 2-3 points along the window/body line
- Be PRECISE - these coordinates will be used for correction overlay
- All polylines ordered from FRONT of vehicle to REAR"""


def extract_primitives(image_base64: str) -> dict:
    """Extract geometric primitives from sketch using GPT-4o Vision."""
    if not client:
        return {}
    
    if "base64," in image_base64:
        image_data = image_base64.split("base64,")[1]
    else:
        image_data = image_base64
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract geometry from this automotive sketch. Trace the lines precisely. Order all polylines from front to rear."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # DEBUG: Log extracted geometry
        print("=" * 60)
        print("GEOMETRY EXTRACTION RESULT:")
        print(json.dumps(result, indent=2))
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"Geometry extraction failed: {e}")
        return {}


# ============================================================================
# LLM PROMPTS — EXACT per User Specification (with all additions)
# ============================================================================

SYSTEM_PROMPT = """You are a senior automotive and industrial design mentor.

Your role is NOT to redesign sketches.
Your role is to correct proportions and line decisions in an existing sketch.

You must think like a senior designer reviewing a junior's sketch with a green pen.

CRITICAL: The geometry you receive is PIXEL-ACCURATE GROUND TRUTH extracted by computer vision.
- You must NOT invent or modify X coordinates unless explicitly necessary.
- Prefer vertical (Y-axis) adjustments over horizontal shifts.
- Assume input geometry is exact — the lines ARE where the data says they are.
- Your corrections are RELATIVE to this ground truth.

Rules you must follow strictly:
- Do NOT generate images.
- Do NOT generate annotations, arrows, circles, or labels.
- Do NOT explain with diagrams.
- Do NOT invent new features or styling.
- Do NOT change the viewing angle or vehicle type.
- Do NOT redraw the entire sketch.

You may ONLY:
- Identify specific existing sketch lines that are proportionally incorrect
- Replace ONLY those line segments with corrected curves
- Preserve original intent and design language
- Apply moderate corrections only (creativity level = 0.5)

Your output must be STRICT JSON following the Correction Stroke Engine schema.
No prose outside JSON.

Maximum number of correction strokes allowed: 5.
If fewer are sufficient, output fewer.

If no confident correction is needed, return an empty correction_strokes array.

You must reason using proportional relationships (wheel-to-body, roof-to-wheelbase, beltline slope), not absolute coordinate values.

STROKE PRIORITY ORDER (follow strictly):
1. Roofline
2. Wheel size / stance
3. Beltline / main character line
4. Glasshouse outline
5. Lower body line

AUTO-REJECT CONDITIONS (do NOT output strokes that):
- Redraw more than 50% of a line
- Add new geometry not in original
- Contradict vehicle type
- Have confidence below 0.65
- Modify multiple unrelated primitives in a single correction"""


USER_PROMPT_TEMPLATE = """You are given extracted geometric primitives from a vehicle sketch.

Context:
- Vehicle type: {vehicle_type}
- View: {view}
- Creativity level: 0.5
- Goal: Proportion and structural correction only

Input geometry (normalized coordinates 0–1):
{geometry_json}

Task:
Review the geometry as a senior designer.

Identify ONLY the lines you would personally redraw with a green pen.

CRITICAL CONSTRAINT — GEOMETRY ALIGNMENT:
Your corrected curves MUST use the SAME X coordinates as the input geometry.
You may ONLY adjust Y values by small amounts (typically 0.01 to 0.05).
This ensures corrected lines TRACE the original sketch, not float in space.

Example:
- If input roofline is: [[0.18, 0.33], [0.42, 0.27], [0.63, 0.29], [0.78, 0.35]]
- A valid correction: [[0.18, 0.31], [0.42, 0.25], [0.63, 0.27], [0.78, 0.33]]
  (Same X values, Y values adjusted by -0.02)
- An INVALID correction would use different X values

For each line you correct:
- Use the EXACT X coordinates from the input primitive
- Adjust Y values by small deltas (0.01 to 0.05)
- The corrected line must overlay the original, not replace it

Constraints:
- Do NOT redraw correct lines
- Do NOT output annotations
- Do NOT output advice
- Do NOT output explanations outside JSON
- Prefer fewer, high-confidence corrections over many small ones
- Each correction must target exactly ONE primitive (no multi-element corrections)
- X coordinates MUST match input geometry
- Y adjustments must be subtle (max 0.05)

OUTPUT SCHEMA:

For roofline corrections (use Bézier, 4 points max):
{{
  "id": "stroke_roofline_01",
  "element": "roofline",
  "stroke_type": "replacement_curve",
  "original_range": {{ "start_t": 0.0, "end_t": 1.0 }},
  "new_curve": {{
    "curve_type": "bezier",
    "points": [USE THE SAME X VALUES FROM INPUT, ADJUST Y ONLY]
  }},
  "style": {{ "color": "#00FF00", "thickness": 2.0, "opacity": 0.85 }},
  "confidence": 0.84,
  "reason": "Improves roof crown continuity",
  "affects_original_visibility": "overlay_only",
  "stroke_role": "primary_correction"
}}

For wheel corrections (radius adjustment only, center stays same):
{{
  "id": "stroke_rear_wheel_01",
  "element": "rear_wheel",
  "stroke_type": "radius_adjustment",
  "original": {{ "cx": [FROM INPUT], "cy": [FROM INPUT], "r": [FROM INPUT] }},
  "corrected": {{ "cx": [SAME AS ORIGINAL], "cy": [SAME AS ORIGINAL], "r": [ADJUSTED BY 0.01-0.02] }},
  "style": {{ "color": "#00FF00", "thickness": 2.0, "opacity": 0.85 }},
  "confidence": 0.79,
  "reason": "Rear wheel appears underpowered",
  "affects_original_visibility": "overlay_only",
  "stroke_role": "primary_correction"
}}

For beltline corrections (use polyline, 2-3 points, SAME X values):
{{
  "id": "stroke_beltline_01",
  "element": "beltline",
  "stroke_type": "polyline_replacement",
  "new_points": [[0.20, 0.47], [0.55, 0.46], [0.80, 0.48]],
  "style": {{ "color": "#00FF00", "thickness": 1.8, "opacity": 0.85 }},
  "confidence": 0.76,
  "reason": "Adds forward tension",
  "affects_original_visibility": "overlay_only",
  "stroke_role": "primary_correction"
}}

FULL RESPONSE FORMAT:
{{
  "correction_strokes": [...],
  "meta": {{
    "confidence_overall": 0.82,
    "intent_preserved": true,
    "notes": "Moderate proportional adjustments applied"
  }}
}}

Return ONLY valid JSON. No prose."""


# ============================================================================
# CORRECTION GENERATION — Step 2
# ============================================================================

def generate_corrections(geometry: dict) -> dict:
    """
    LLM reviews geometry and outputs correction strokes.
    Uses EXACT prompts from user specification.
    """
    if not client:
        return {"correction_strokes": [], "meta": {"confidence_overall": 0, "intent_preserved": True, "notes": "Client unavailable"}}
    
    primitives = geometry.get("primitives", {})
    vehicle_type = geometry.get("vehicle_type", "vehicle")
    view = geometry.get("view", "side")
    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        vehicle_type=vehicle_type,
        view=view,
        geometry_json=json.dumps({"primitives": primitives}, indent=2)
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=1500
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate strokes against the extracted geometry
        validated_strokes = validate_strokes(result.get("correction_strokes", []), primitives)
        result["correction_strokes"] = validated_strokes
        
        return result
        
    except Exception as e:
        print(f"Correction generation failed: {e}")
        return {"correction_strokes": [], "meta": {"confidence_overall": 0, "intent_preserved": True, "notes": f"Error: {str(e)}"}}


def validate_strokes(strokes: list, geometry: dict) -> list:
    """
    Validate strokes against studio rules AND extracted geometry.
    
    CRITICAL: Ensures output coordinates align with input geometry.
    - X coordinates must match input within tolerance
    - Y adjustments must be <= 0.05
    """
    if not strokes:
        return []
    
    valid_strokes = []
    allowed_elements = {"roofline", "beltline", "front_wheel", "rear_wheel", "body_lower_edge", "greenhouse"}
    
    # Extract original geometry for validation
    def get_original_points(element: str) -> list:
        """Get original points for an element from extracted geometry."""
        if element not in geometry:
            return []
        elem_data = geometry[element]
        if isinstance(elem_data, dict) and "points" in elem_data:
            return elem_data["points"]
        elif isinstance(elem_data, list):
            return elem_data
        return []
    
    def validate_alignment(original: list, corrected: list, max_y_delta: float = 0.05) -> bool:
        """
        Validate that corrected points align with original.
        - X values must match within tolerance (0.01)
        - Y deltas must be <= max_y_delta
        """
        if not original or not corrected:
            return True  # Can't validate if missing data
        
        if len(original) != len(corrected):
            return False  # Must have same number of points
        
        for orig, corr in zip(original, corrected):
            if not isinstance(orig, list) or not isinstance(corr, list):
                continue
            if len(orig) < 2 or len(corr) < 2:
                continue
            
            # X coordinate must match within tolerance
            x_delta = abs(orig[0] - corr[0])
            if x_delta > 0.02:  # Allow small tolerance for rounding
                print(f"Rejecting stroke: X delta too large ({x_delta:.3f})")
                return False
            
            # Y adjustment must be reasonable
            y_delta = abs(orig[1] - corr[1])
            if y_delta > max_y_delta:
                print(f"Rejecting stroke: Y delta too large ({y_delta:.3f})")
                return False
        
        return True
    
    for stroke in strokes[:5]:  # Max 5 strokes
        # Check confidence >= 0.65
        if stroke.get("confidence", 0) < 0.65:
            continue
        
        # Check element is allowed (priority elements only)
        element = stroke.get("element", "")
        if element not in allowed_elements:
            continue
        
        # Get original points for this element
        original_points = get_original_points(element)
        
        # Validate curve corrections
        if stroke.get("new_curve"):
            points = stroke["new_curve"].get("points", [])
            
            # Roofline: max 4 points (Bézier)
            if element == "roofline" and len(points) > 4:
                stroke["new_curve"]["points"] = points[:4]
                points = points[:4]
            
            # Validate alignment with original geometry
            if original_points and not validate_alignment(original_points, points):
                print(f"Skipping {element} stroke: geometry doesn't align with original")
                continue
        
        # Validate polyline replacements
        if stroke.get("new_points"):
            points = stroke["new_points"]
            
            # Beltline: max 3 points
            if element == "beltline" and len(points) > 3:
                stroke["new_points"] = points[:3]
                points = points[:3]
            
            # Validate alignment
            if original_points and not validate_alignment(original_points, points):
                print(f"Skipping {element} stroke: geometry doesn't align with original")
                continue
        
        # Validate wheel corrections (center should stay same, radius adjustment limited)
        if stroke.get("original") and stroke.get("corrected"):
            orig = stroke["original"]
            corr = stroke["corrected"]
            
            # Center should not move
            if abs(orig.get("cx", 0) - corr.get("cx", 0)) > 0.01:
                print(f"Rejecting wheel stroke: center X moved")
                continue
            if abs(orig.get("cy", 0) - corr.get("cy", 0)) > 0.01:
                print(f"Rejecting wheel stroke: center Y moved")
                continue
            
            # Radius adjustment should be limited
            r_delta = abs(orig.get("r", 0) - corr.get("r", 0))
            if r_delta > 0.03:
                print(f"Rejecting wheel stroke: radius change too large ({r_delta:.3f})")
                continue
        
        # Ensure required fields
        if "affects_original_visibility" not in stroke:
            stroke["affects_original_visibility"] = "overlay_only"
        if "stroke_role" not in stroke:
            stroke["stroke_role"] = "primary_correction"
        
        valid_strokes.append(stroke)
    
    return valid_strokes


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def generate_correction_overlay(
    image_base64: str,
    image_width: int,
    image_height: int,
    creativity_factor: float = 0.5
) -> CorrectionStrokeResponse:
    """
    DREAM V3 — Correction Stroke Engine
    
    CORRECTED ARCHITECTURE:
    - Geometry extraction: CV (deterministic, pixel-accurate)
    - Design judgment: LLM (text-only, treats geometry as ground truth)
    
    Pipeline:
    1. User uploads sketch (raster)
    2. **CV** extracts geometric primitives (NOT GPT-4o Vision)
    3. Geometry normalized to 0–1 space
    4. LLM reviews geometry (treats it as ground truth)
    5. LLM outputs correction strokes (deltas only)
    6. Frontend renders green replacement curves
    7. User toggles correction overlay
    
    "We are not asking the LLM to see the sketch.
    We are asking it to judge decisions made in the sketch, using geometry we already trust."
    """
    # Step 1-3: Extract primitives using CV (NOT LLM)
    from .cv_geometry_extraction import extract_primitives_cv
    
    geometry = extract_primitives_cv(image_base64, image_width, image_height)
    
    primitives = geometry.get("primitives", {})
    if not primitives:
        return CorrectionStrokeResponse(
            correction_strokes=[],
            meta=CorrectionMeta(
                confidence_overall=0,
                intent_preserved=True,
                notes="Could not extract geometry from sketch (CV failed)"
            ),
            error="CV geometry extraction failed - no lines detected"
        )
    
    # Step 4-5: Generate corrections
    result = generate_corrections(geometry)
    
    # Build response with proper schema
    strokes = []
    for s in result.get("correction_strokes", []):
        try:
            stroke = CorrectionStroke(
                id=s.get("id", f"stroke_{s.get('element', 'unknown')}"),
                element=s.get("element", "unknown"),
                stroke_type=s.get("stroke_type", "replacement_curve"),
                original_range=OriginalRange(**s["original_range"]) if s.get("original_range") else None,
                new_curve=NewCurve(**s["new_curve"]) if s.get("new_curve") else None,
                new_points=s.get("new_points"),
                original=WheelGeometry(**s["original"]) if s.get("original") else None,
                corrected=WheelGeometry(**s["corrected"]) if s.get("corrected") else None,
                style=CurveStyle(**(s.get("style", {}))),
                confidence=s.get("confidence", 0.7),
                reason=s.get("reason", ""),
                affects_original_visibility=s.get("affects_original_visibility", "overlay_only"),
                stroke_role=s.get("stroke_role", "primary_correction")
            )
            strokes.append(stroke)
        except Exception as e:
            print(f"Failed to parse stroke: {e}")
    
    meta = result.get("meta", {})
    
    return CorrectionStrokeResponse(
        correction_strokes=strokes,
        meta=CorrectionMeta(
            confidence_overall=meta.get("confidence_overall", 0.5),
            intent_preserved=meta.get("intent_preserved", True),
            notes=meta.get("notes", "")
        ),
        error=None
    )
