from pydantic import BaseModel
from typing import List, Optional

class Point(BaseModel):
    x: float
    y: float
    p: float = 0.5  # Pressure
    t: float        # Timestamp

class Stroke(BaseModel):
    id: str
    points: List[Point]
    color: str = "#ffffff"
    width: float = 2.0
    completed: bool = True

class IngestRequest(BaseModel):
    strokes: List[Stroke]
    canvas_width: int
    canvas_height: int
