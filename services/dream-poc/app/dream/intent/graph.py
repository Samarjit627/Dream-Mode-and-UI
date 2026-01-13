from pydantic import BaseModel
from typing import List
from ..ingestion.models import Stroke
from ..stroke_engine.primitives import Primitive
from ..stroke_engine.structure import StructuralAnalysis
from ..judgment.engine import JudgmentResponse

class IntentNode(BaseModel):
    id: str
    type: str # primitive type
    params: dict
    confidence: float
    origin_stroke_id: str

class IntentGraph(BaseModel):
    nodes: List[IntentNode]
    structure: StructuralAnalysis
    judgment: JudgmentResponse
    timestamp: float
    
def build_intent_graph(
    clean_strokes: List[Stroke],
    primitives: List[Primitive],
    structure: StructuralAnalysis,
    judgment: JudgmentResponse
) -> IntentGraph:
    """
    Constructs the final Handoff Graph from the Dream components.
    """
    nodes = []
    for i, prim in enumerate(primitives):
        # Map primitive back to stroke ID if possible (for now 1:1)
        origin_id = clean_strokes[i].id if i < len(clean_strokes) else "unknown"
        
        nodes.append(IntentNode(
            id=f"node_{i}",
            type=prim.type,
            params=prim.params,
            confidence=prim.confidence,
            origin_stroke_id=origin_id
        ))
        
    import time
    return IntentGraph(
        nodes=nodes,
        structure=structure,
        judgment=judgment,
        timestamp=time.time()
    )
