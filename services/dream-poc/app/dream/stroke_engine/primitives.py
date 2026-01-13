from typing import List, Optional
from pydantic import BaseModel
from ..ingestion.models import Stroke, Point
import math

class Primitive(BaseModel):
    type: str  # line, arc, circle, poly
    confidence: float
    params: dict = {} # e.g. {start, end} for line, {center, radius} for circle

def dist(p1: Point, p2: Point) -> float:
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def point_to_line_dist(pt: Point, start: Point, end: Point) -> float:
    # Perpendicular distance from pt to line segment start-end
    if start.x == end.x and start.y == end.y:
        return dist(pt, start)
        
    num = abs((end.y - start.y) * pt.x - (end.x - start.x) * pt.y + end.x * start.y - end.y * start.x)
    den = math.sqrt((end.y - start.y)**2 + (end.x - start.x)**2)
    return num / den

def detect_primitive(stroke: Stroke) -> Primitive:
    """
    Classifies a stroke into a geometric primitive.
    """
    points = stroke.points
    if not points or len(points) < 2:
        return Primitive(type="point", confidence=1.0)
        
    start = points[0]
    end = points[-1]
    
    # 1. Check for Closure (Circle/Loop)
    chord_len = dist(start, end)
    
    # Calculate path length
    path_len = 0
    for i in range(1, len(points)):
        path_len += dist(points[i-1], points[i])
        
    closure_threshold = path_len * 0.2  # If gap is < 20% of length, maybe closed
    is_closed = chord_len < max(20, closure_threshold) 
    
    if is_closed and path_len > 50: # Must have some size
        # Heuristic: Is it a circle?
        # Circle has constant radius from centroid.
        # For V1, if closed, call it a 'loop' or 'circle'
        return Primitive(type="circle", confidence=0.8, params={"center": {"x":0, "y":0}, "radius": path_len / (2 * math.pi)})

    # 2. Check for Line
    # Measure max deviation from the chord (start->end)
    max_deviation = 0
    for p in points:
        d = point_to_line_dist(p, start, end)
        if d > max_deviation:
            max_deviation = d
            
    # Line threshold: Deviation should be small relative to length
    # e.g. < 5% of length or < 10px absolute
    line_threshold = max(5.0, chord_len * 0.05)
    
    if max_deviation < line_threshold:
        return Primitive(type="line", confidence=0.9, params={"start": start.dict(), "end": end.dict()})
        
    # 3. Else Arc / Poly
    return Primitive(type="arc", confidence=0.6)
