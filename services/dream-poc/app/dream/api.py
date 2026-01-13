from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from .ingestion.models import IngestRequest, Stroke
from .stroke_engine.clustering import cluster_strokes
from .stroke_engine.simplification import extract_median_line
from .stroke_engine.primitives import detect_primitive, Primitive
from .stroke_engine.structure import analyze_structure, StructuralAnalysis
from .judgment.engine import judge_design, JudgmentResponse

router = APIRouter(prefix="/api/v1/dream", tags=["dream"])

class JudgeRequest(BaseModel):
    image: str # Base64 data url

@router.post("/ingest")
async def ingest_strokes(request: IngestRequest):
    """
    Ingests raw strokes, cleans them (hairy line clustering), and returns the simplified intent.
    """
    raw_strokes = request.strokes
    
    # 1. Cluster hairy lines
    clusters = cluster_strokes(raw_strokes)
    
    # 2. Extract median lines
    clean_strokes = []
    primitives: list[Primitive] = []
    
    for cluster in clusters:
        clean = extract_median_line(cluster)
        clean_strokes.append(clean)
        
        # 3. Detect Primitives
        prim = detect_primitive(clean)
        primitives.append(prim)
        
    # 4. Analyze Structure
    structure = analyze_structure(primitives)
        
    return {
        "raw_count": len(raw_strokes),
        "clean_count": len(clean_strokes),
        "clean_strokes": clean_strokes,
        "primitives": primitives,
        "structure": structure
    }

@router.post("/judge")
async def measure_judgment(request: JudgeRequest):
    """
    Evaluates the design sketch against the DJKB rules.
    """
    return judge_design(request.image)

@router.post("/lock")
async def lock_intent(request: IngestRequest):
    """
    Finalizes the Dream Session.
    Runs Ingestion + Structure + Judgment (Mocked for now if no image) + Graph Build.
    Returns the IntentGraph ready for Build Mode.
    """
    # 1. Pipeline
    clusters = cluster_strokes(request.strokes)
    clean_strokes = []
    primitives = []
    for cluster in clusters:
        clean = extract_median_line(cluster)
        clean_strokes.append(clean)
        primitives.append(detect_primitive(clean))
        
    structure = analyze_structure(primitives)
    
    # 2. Judgment (We need an image for real judgment, but for Lock API we migth skip or require it)
    # For V1, we return an empty judgment if not provided, or frontend calls /judge separate.
    # Let's assume passed validation.
    from .judgment.engine import JudgmentResponse
    judgment = JudgmentResponse(results=[], summary="Locked without visual critique")
    
    # 3. Build Graph
    from .intent.graph import build_intent_graph
    graph = build_intent_graph(clean_strokes, primitives, structure, judgment)
    
    return graph


# --- Sketch Correction Endpoint ---

class CorrectionRequest(BaseModel):
    image: str  # Base64 data URL
    object_type: str = "object"  # Identified object type
    image_width: int = 1024
    image_height: int = 1024
    creativity_factor: float = 0.3  # 0.0 to 1.0
    initial_critique: str = ""  # From the analysis phase


# --- Template-Based Sketch Warping ---

class WarpRequest(BaseModel):
    image: str  # Base64 data URL
    segment: str = "hatchback"  # scooter, hatchback, sedan, suv
    view: str = "front_3q"  # side, front_3q, rear_3q
    strength: Optional[float] = None  # Mentor Mode Override

@router.post("/correct")
async def warp_sketch_proportions(request: WarpRequest):
    """
    Template-Based Sketch Warping — Proportion Correction
    
    Uses reference templates as ground truth for correct proportions.
    
    Flow:
    1. Load template for segment + view
    2. AI Vision maps features between sketch and template
    3. Warp sketch to match template proportions
    
    Output = SAME pixels, repositioned to match template.
    """
    from .judgment.sketch_warping import warp_sketch
    
    result = warp_sketch(
        image_base64=request.image,
        segment=request.segment,
        view=request.view,
        strength_override=request.strength
    )
    
    return result
