from fastapi import APIRouter, Response
from ..dream.intent.graph import IntentGraph
from .generator import generate_3d_model

router = APIRouter(prefix="/api/v1/dream", tags=["dream"])

@router.post("/build")
async def build_from_intent(graph: IntentGraph):
    """
    Takes a Locked Intent Graph and generates a 3D STL.
    """
    stl_bytes = generate_3d_model(graph)
    
    return Response(
        content=stl_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=dream_build.stl"}
    )
