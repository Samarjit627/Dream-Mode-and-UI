from typing import List
from ..ingestion.models import Stroke, Point
import statistics

def resample_stroke(stroke: Stroke, num_points: int = 50) -> List[Point]:
    """Resamples a stroke to have exactly `num_points` equidistant points."""
    if not stroke.points:
        return []
        
    # Calculate total length
    total_len = 0
    dists = [0.0]
    for i in range(1, len(stroke.points)):
        p1 = stroke.points[i-1]
        p2 = stroke.points[i]
        d = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
        total_len += d
        dists.append(total_len)
        
    if total_len == 0:
        return [stroke.points[0]] * num_points
        
    new_points = []
    step = total_len / (num_points - 1)
    
    current_dist = 0
    pts_idx = 0
    
    for i in range(num_points):
        target_dist = i * step
        
        # Find segment
        while pts_idx < len(dists) - 1 and dists[pts_idx+1] < target_dist:
            pts_idx += 1
            
        if pts_idx >= len(dists) - 1:
             new_points.append(stroke.points[-1])
             continue
             
        # Interpolate
        p_start = stroke.points[pts_idx]
        p_end = stroke.points[pts_idx+1]
        t = (target_dist - dists[pts_idx]) / (dists[pts_idx+1] - dists[pts_idx])
        
        nx = p_start.x + (p_end.x - p_start.x) * t
        ny = p_start.y + (p_end.y - p_start.y) * t
        new_points.append(Point(x=nx, y=ny, t=0))
        
    return new_points

def extract_median_line(cluster: List[Stroke]) -> Stroke:
    """
    Takes a cluster of strokes and returns the average (median) stroke.
    """
    if not cluster:
        raise ValueError("Empty cluster")
    if len(cluster) == 1:
        return cluster[0]
        
    # Resample all to 50 points
    resampled = [resample_stroke(s, 50) for s in cluster]
    
    # Average x and y for each index
    median_points = []
    for i in range(50):
        xs = [r[i].x for r in resampled]
        ys = [r[i].y for r in resampled]
        median_points.append(Point(
            x=statistics.mean(xs),
            y=statistics.mean(ys),
            t=0
        ))
        
    # Create new ID
    return Stroke(
        id=f"median_{cluster[0].id}",
        points=median_points,
        color="#00ff00", # Green for median
        width=3.0,
        completed=True
    )
