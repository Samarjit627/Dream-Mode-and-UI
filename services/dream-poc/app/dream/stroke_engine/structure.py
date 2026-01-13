from typing import List, Dict, Optional
from pydantic import BaseModel
from .primitives import Primitive
import math

class SymmetryAxis(BaseModel):
    type: str # vertical, horizontal
    x: Optional[float] = None
    y: Optional[float] = None
    confidence: float

class Alignment(BaseModel):
    type: str # left, right, top, bottom, center-x, center-y
    value: float
    primitive_indices: List[int]
    confidence: float

class StructuralAnalysis(BaseModel):
    symmetries: List[SymmetryAxis]
    alignments: List[Alignment]

def detect_symmetry(primitives: List[Primitive]) -> List[SymmetryAxis]:
    """
    Detects global symmetry axes.
    """
    if len(primitives) < 2:
        return []

    # Calculate bounding box of all primitives
    min_x = float('inf')
    max_x = float('-inf')
    for p in primitives:
        # Approximate using params (start/end or center)
        if p.type == "line":
            xs = [p.params["start"]["x"], p.params["end"]["x"]]
            min_x = min(min_x, min(xs))
            max_x = max(max_x, max(xs))
        elif p.type == "circle":
            c = p.params["center"]["x"]
            r = p.params["radius"]
            min_x = min(min_x, c - r)
            max_x = max(max_x, c + r)

    mid_x = (min_x + max_x) / 2
    
    # Check for Vertical Symmetry around mid_x
    # For every primitive on left, is there a matching one on right?
    matches = 0
    left_prims = 0
    
    for i, p1 in enumerate(primitives):
        # Get center of p1
        p1_x = 0
        if p1.type == "line":
            p1_x = (p1.params["start"]["x"] + p1.params["end"]["x"]) / 2
        elif p1.type == "circle":
            p1_x = p1.params["center"]["x"]
            
        if p1_x < mid_x:
            left_prims += 1
            # Look for reflection
            target_x = mid_x + (mid_x - p1_x)
            
            found = False
            for p2 in primitives:
                 p2_x = 0
                 if p2.type == "line": p2_x = (p2.params["start"]["x"] + p2.params["end"]["x"]) / 2
                 elif p2.type == "circle": p2_x = p2.params["center"]["x"]
                 
                 if abs(p2_x - target_x) < 20: # Tolerance
                     found = True
                     break
            if found:
                matches += 1
                
    if left_prims > 0 and (matches / left_prims) > 0.7:
        return [SymmetryAxis(type="vertical", x=mid_x, confidence=matches/left_prims)]
        
    return []

def detect_alignments(primitives: List[Primitive]) -> List[Alignment]:
    """
    Detects groups of primitives that align.
    """
    # Group by X and Y coordinates (bucketed)
    x_centers = {}
    
    for i, p in enumerate(primitives):
        cx = 0
        if p.type == "line": cx = (p.params["start"]["x"] + p.params["end"]["x"]) / 2
        elif p.type == "circle": cx = p.params["center"]["x"]
        
        # Bucket by 10px
        bucket = round(cx / 10) * 10
        if bucket not in x_centers:
            x_centers[bucket] = []
        x_centers[bucket].append(i)
        
    alignments = []
    for x, indices in x_centers.items():
        if len(indices) > 2: # At least 3 items to call it an alignment
            alignments.append(Alignment(
                type="center-x",
                value=x,
                primitive_indices=indices,
                confidence=0.9
            ))
            
    return alignments

def analyze_structure(primitives: List[Primitive]) -> StructuralAnalysis:
    return StructuralAnalysis(
        symmetries=detect_symmetry(primitives),
        alignments=detect_alignments(primitives)
    )
