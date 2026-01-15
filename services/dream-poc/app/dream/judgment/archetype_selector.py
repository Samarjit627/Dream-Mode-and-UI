"""
Stage 2 - Archetype Selector Logic Layer
This module is the "Brain" of the proportion correction system.

Responsibilities:
1. Load the correct JSON based on Segment + View + Archetype.
2. Bridge CV measurements (points) to JSON metrics (ratios).
3. Score "Archetype" (Sporty vs Balanced) based on Grok observations.
4. Compare measured ratios vs. Ideal Bands.
5. Generate deterministic correction prompts.

Strict Rules:
- Enforce "Do Nothing" if inside band.
- Correct towards the nearest boundary (min or max).
- Gating: If CV Confidence is low, fallback to "Balanced" default without measurement checks.
"""
import os
import json
from typing import Dict, List, Optional, Tuple

PROPORTIONS_DIR = os.path.join(os.path.dirname(__file__), "proportions")

class ArchetypeSelector:
    def __init__(self):
        self.library_path = PROPORTIONS_DIR
        
    def _load_json(self, segment: str, archetype: str, view: str) -> Optional[Dict]:
        """Loads the specific JSON file for the requested configuration."""
        # Sanitize inputs
        segment = segment.lower().replace(" ", "")
        archetype = archetype.lower()
        view = view.lower()
        
        # Construct filename
        # Pattern: {segment}_{archetype}_{view}.json
        # Exception: Front 3Q usually omits view in legacy files, but we standardized on explicit names or assumed default?
        # Based on file creation: we have 'microcar_balanced.json' (implied front_3q) and 'microcar_balanced_side.json'.
        # Let's try explicit first, then fallback to implicit for front_3q
        
        if view == "front_3q":
            # Try explicit first (e.g. hatchback_sporty_front_3q.json - we didn't create this naming scheme yet)
            # We created 'hatchback_sporty.json' for front_3q.
            # So for front_3q, we look for {segment}_{archetype}.json
            filename = f"{segment}_{archetype}.json"
        else:
            filename = f"{segment}_{archetype}_{view}.json"
            
        path = os.path.join(self.library_path, filename)
        
        if not os.path.exists(path):
            print(f"  [ArchetypeSelector] Warning: {filename} not found. Falling back to 'balanced'.")
            # Fallback to balanced if sporty missing
            if view == "front_3q":
                fallback = f"{segment}_balanced.json"
            else:
                fallback = f"{segment}_balanced_{view}.json"
            path = os.path.join(self.library_path, fallback)
            
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  [ArchetypeSelector] Error loading JSON: {e}")
                return None
        return None

    def calculate_ratios(self, primitives: Dict, width: int, height: int) -> Dict[str, float]:
        """
        Bridge: Converts CV Primitives (Points) -> JSON Metrics (Ratios).
        
        Ratios needed:
        - wheelbase_ratio (Wheelbase / Length)
        - body_height_to_length (Body Height / Length)
        - rear_wheel_scale (Rear R / Front R)
        - greenhouse_height_ratio (Greenhouse H / Body H) - Hard to get from simple lines?
        - ground_clearance_ratio (Ground / Body H)
        - front_overhang_ratio (Front Axle to Nose / Length)
        """
        ratios = {}
        
        # 1. Wheelbase & Length
        # We need Front/Rear Wheels
        fw = primitives.get("front_wheel")
        rw = primitives.get("rear_wheel")
        
        if fw and rw:
            # Distance between centers
            wb_px = rw["cx_px"] - fw["cx_px"]
            
            # Estimate Length:
            # We don't have explicit bumpers. Heuristic: 
            # Length ~= Wheelbase * 1.6 (approx standard) OR
            # Length = Bounding box width of all primitives?
            # Let's use BB of all detected primitives + margin
            all_x = []
            all_y = []
            
            for k, prim in primitives.items():
                if "pixel_points" in prim:
                    pts = prim["pixel_points"]
                    all_x.extend([p[0] for p in pts])
                    all_y.extend([p[1] for p in pts])
            
            if fw: all_x.extend([fw["cx_px"] - fw["r_px"], fw["cx_px"] + fw["r_px"]])
            if rw: all_x.extend([rw["cx_px"] - rw["r_px"], rw["cx_px"] + rw["r_px"]])
            
            if all_x:
                min_x, max_x = min(all_x), max(all_x)
                length_px = max_x - min_x
                
                # Check for sanity
                if length_px > 0:
                    ratios["wheelbase_ratio"] = wb_px / length_px
                    
                    # 2. Body Height
                    # Lowest point (ground) vs Highest point (roof)
                    if all_y:
                        min_y, max_y = min(all_y), max(all_y) # Y increases downwards in CV usually?
                        # Note: OpenCV 0,0 is top-left.
                        # Ground is MAX Y. Roof is MIN Y.
                        # Height = Ground - Roof
                        ground_y = max_y
                        roof_y = min_y
                        body_h_px = ground_y - roof_y
                        
                        ratios["body_height_to_length"] = body_h_px / length_px
                        
                        # 3. Ground Clearance
                        # Lowest wheel point vs Lowest body point?
                        # Heuristic: Gap between wheel bottom and rocker?
                        # Let's assume Ground Y is defined by wheel bottoms
                        wheel_bottom_y = max(fw["cy_px"] + fw["r_px"], rw["cy_px"] + rw["r_px"])
                        # Rocker line approx? Hard to detect.
                        # Skipping complex ones for now.
                        
            # 4. Wheel Scale
            if fw["r_px"] > 0:
                ratios["rear_wheel_scale"] = rw["r_px"] / fw["r_px"]
                
        return ratios

    def score_archetype(self, observations: Dict) -> str:
        """
        Determines 'Sporty' vs 'Balanced' based on Grok's feeling words.
        SCORING MATRIX:
        - greenhouse_light -> Sporty++
        - greenhouse_heavy -> Tall++ (Balanced/Tall)
        - body_height_low -> Sporty++
        - stance_low -> Sporty++
        """
        score_sporty = 0
        score_tall = 0
        
        # Greenhouse
        gh = observations.get("greenhouse", "balanced")
        if gh == "light": score_sporty += 1
        elif gh == "heavy": score_tall += 1
        
        # Body Height
        bh = observations.get("body_height", "balanced")
        if bh == "low": score_sporty += 1
        elif bh == "tall": score_tall += 1
        
        # Normalize/Decision
        # Simple Logic: If Sporty dominant, return sporty.
        # Confidence threshold: 0.65?
        # Here we have sparse data (3-4 points).
        
        if score_sporty > score_tall:
            return "sporty"
        elif score_tall > score_sporty:
            return "balanced" # Or "tall" if we had that JSON, but we map to balanced/tall
        
        return "balanced" # Default

    def get_corrections(self, segment: str, view: str, observations: Dict, measurements: Optional[Dict] = None, perspective: Optional[Dict] = None) -> List[str]:
        """
        Main Entry Point.
        Returns list of Prompt Modifiers (e.g., "extended wheelbase").
        """
        print(f"  [Logic] Selecting Archetype for {segment} ({view})...")
        
        # 1. Determine Archetype
        archetype = self.score_archetype(observations)
        print(f"  [Logic] Inferred Archetype: {archetype}")
        
        # 2. Load JSON
        config = self._load_json(segment, archetype, view)
        if not config:
            print("  [Logic] No JSON found, returning empty corrections.")
            return []
            
        corrections = []
        bands = config.get("proportion_bands", {})
        
        # 2b. Process Unconditional Modifiers (Grok Alignment)
        # These are applied regardless of measurements to ensure style/structure.
        json_modifiers = config.get("modifiers", [])
        print(f"  [Logic] DEBUG: Found {len(json_modifiers)} modifiers in JSON")
        for i, mod in enumerate(json_modifiers):
            print(f"  [Logic] DEBUG: Modifier {i}: {mod}")
            if mod.get("condition") == "always":
                print(f"  [Logic] Applying Always-Modifier: {mod.get('prompt_addition')}")
                corrections.append(mod.get("prompt_addition"))
            else:
                print(f"  [Logic] DEBUG: Skipping modifier (condition={mod.get('condition')})")


        # 3. Compare Measurements (If available)
        if measurements:
            for metric, val in measurements.items():
                if metric in bands:
                    min_v, max_v = bands[metric]
                    if val < min_v:
                        # Violation: Too Small
                        corrections.append(self._get_correction_text(metric, "increase"))
                    elif val > max_v:
                        # Violation: Too Large
                        corrections.append(self._get_correction_text(metric, "decrease"))
                    else:
                        # In Band - Do Nothing
                        pass
        else:
            # 4. Fallback: Use Observations (Qualitative) if no CV measurements
            
            # Wheelbase
            wb_feel = observations.get("wheelbase", "balanced")
            if wb_feel == "short": corrections.append("extended wheelbase")
            elif wb_feel == "long": corrections.append("compact wheelbase")
            
            # Greenhouse
            gh_feel = observations.get("greenhouse", "balanced")
            if gh_feel == "heavy": corrections.append("lower roofline")
            elif gh_feel == "light": corrections.append("increased greenhouse height")
            
            # Body Height
            bh_feel = observations.get("body_height", "balanced")
            if bh_feel == "tall": corrections.append("lowered stance")
            elif bh_feel == "low": corrections.append("raised body height")

        # 5. Perspective Corrections (V2 Strict Mode)
        # New logic to handle perspective failures from Stage 1
        if perspective:
            # Rear Compression
            rear_comp = perspective.get("rear_compression")
            if rear_comp == "insufficient":
                 corrections.append("force perspective convergence")
                 corrections.append("wider rear track")
                 
            # Character Lines
            char_lines = perspective.get("character_line_behavior")
            if char_lines == "parallel": # Should converge in 3/4 view
                 corrections.append("converging character lines")

        return corrections

    def _get_correction_text(self, metric: str, direction: str) -> str:
        """Helper to generate text from metric direction."""
        map_text = {
            "wheelbase_ratio": {"increase": "extended wheelbase", "decrease": "compact wheelbase"},
            "body_height_to_length": {"increase": "taller body", "decrease": "lower stance"},
            "rear_wheel_scale": {"increase": "larger rear wheels", "decrease": "smaller rear wheels"},
            "greenhouse_height_ratio": {"increase": "taller greenhouse", "decrease": "chopped roofline"}
        }
        
        if metric in map_text:
            return map_text[metric].get(direction, "")
        return f"{direction} {metric.replace('_', ' ')}"
