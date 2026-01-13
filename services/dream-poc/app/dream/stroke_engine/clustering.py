from typing import List, Dict
import math
from ..ingestion.models import Stroke, Point

# Simple Distance-Based Clustering for V1 "Hairy Lines"
# If strokes are close enough, they belong to the same "Intent Group"

def euclidean_distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def stroke_center(stroke: Stroke) -> Point:
    # Centroid of the stroke
    if not stroke.points:
        return Point(x=0, y=0, t=0)
    
    xs = [p.x for p in stroke.points]
    ys = [p.y for p in stroke.points]
    return Point(x=sum(xs)/len(xs), y=sum(ys)/len(ys), t=0)

def cluster_strokes(strokes: List[Stroke], threshold: float = 50.0) -> List[List[Stroke]]:
    """
    Groups strokes that are spatially close into clusters.
    """
    if not strokes:
        return []

    clusters: List[List[Stroke]] = []
    
    # Primitive greedy clustering
    # TODO: Use spatial index (R-Tree) for performance if N > 1000
    
    centers = [(s, stroke_center(s)) for s in strokes]
    
    used = set()
    
    for i, (s1, c1) in enumerate(centers):
        if s1.id in used:
            continue
            
        current_cluster = [s1]
        used.add(s1.id)
        
        for j, (s2, c2) in enumerate(centers):
            if i == j or s2.id in used:
                continue
            
            # Distance check
            dist = euclidean_distance(c1, c2)
            if dist < threshold:
                current_cluster.append(s2)
                used.add(s2.id)
        
        clusters.append(current_cluster)
        
    return clusters
