from typing import Optional, List
from pydantic import BaseModel
from .sketch_warping import correct_sketch, CorrectionResponse

class MentorTuneRequest(BaseModel):
    base_image: str
    strength: Optional[float] = None
    regions: Optional[List[str]] = None  # Future: ["wheels", "roof"]
    segment: str = "hatchback"
    view: str = "front_3q"

def tune_correction(request: MentorTuneRequest) -> CorrectionResponse:
    """
    Handles interactive tuning requests from the Mentor UI.
    Passes user overrides (strength, regions) to the core engine.
    """
    print(f"  [Mentor] Tuning Request: Strength={request.strength}, Regions={request.regions}")
    
    # Future: Logic to modify prompt based on 'regions' would go here.
    # e.g., if "wheels" not in regions, remove "smaller rear wheels" from prompt.
    
    return correct_sketch(
        image_base64=request.base_image,
        segment=request.segment,
        view=request.view,
        strength_override=request.strength
    )
