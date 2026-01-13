from build123d import *
from ..dream.intent.graph import IntentGraph
import io

def generate_3d_model(graph: IntentGraph) -> bytes:
    """
    Converts Intent Graph to STL bytes using build123d.
    """
    if not graph.nodes:
        return b""
        
    # V1 Strategy: Naive Extrusion of all primitives
    # We combine all 2D shapes into a single sketch and extrude.
    
    with BuildPart() as part:
        with BuildSketch() as sketch:
            for node in graph.nodes:
                if node.type == "circle":
                    # Circle params: center, radius
                    # Default center for V1 is absolute canvas coords? 
                    # We might need to normalize or use relative.
                    # Assuming backend returned absolute frame.
                    cx = node.params.get("center", {}).get("x", 0)
                    cy = node.params.get("center", {}).get("y", 0)
                    r = node.params.get("radius", 10)
                    
                    # Flip Y because Canvas (0,0) is top-left, CAD is bottom-left usually
                    # But for now let's just place it.
                    with Locations((cx, -cy)):
                        Circle(radius=r)
                        
                elif node.type == "line":
                     # Lines need to form closed loops to be extruded.
                     # If we have disparate lines, we can't extrude easily without a solver.
                     # For V1, we only extrude explicit "Circles" or "Closed Loops".
                     # If it's a line, valid only if it's a wire?
                     # Let's skip loose lines for V1 "Solid" generation.
                     pass
                     
        # Extrude the aggregate sketch
        Extrude(amount=50) # 50mm default height
        
    # Export to STL
    # build123d exports to file, but we want bytes.
    # We can write to a temporary buffer.
    
    with io.BytesIO() as buf:
        part.part.export_stl(buf)
        return buf.getvalue()
