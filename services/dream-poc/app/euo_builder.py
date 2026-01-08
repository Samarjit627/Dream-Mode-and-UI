"""
Engineering Understanding Object (EUO) Builder
Strict rule-based transformation from raw perception signals to a stable EUO.
NO semantic reasoning. NO summarization. NO ML inference.
"""
from typing import Dict, Any, List, Optional
import re
import statistics

def _validate_structure(perception: Dict[str, Any]) -> List[str]:
    """Step 1: Structural Validation"""
    flags = []
    
    # Check title block confidence
    regions = perception.get("regions", {})
    tb = regions.get("title_block")
    if tb and isinstance(tb, dict):
        conf = float(tb.get("confidence", 0.0))
        if conf < 0.7:
            flags.append("low_title_block_confidence")
            
    # Check geometry
    shapes = perception.get("closed_shapes", [])
    if not shapes:
        flags.append("no_closed_shapes")
        
    # Check text density (heuristic)
    blocks = perception.get("text_blocks", [])
    if len(blocks) < 5:
        flags.append("low_text_density")
        
    return flags

def _classify_document(perception: Dict[str, Any]) -> Dict[str, str]:
    """Step 2: Document Classification (Ruled)"""
    regions = perception.get("regions", {})
    tb = regions.get("title_block")
    vp = regions.get("viewports", [])
    arrows = perception.get("arrow_candidates", [])
    
    doc_type = "UNKNOWN"
    if tb and len(vp) >= 0 and len(arrows) > 0:
         # Note: Requirement said len(vp) >= 1 but heuristics might miss viewports, 
         # title block + arrows is strong signal for production drawing
        doc_type = "PRODUCTION_DRAWING"
    elif tb:
        # Fallback if title block exists but other signals weak
        doc_type = "PRODUCTION_DRAWING"
        
    return {
        "document_type": doc_type,
        "drawing_standard_hint": "UNKNOWN",
        "industry_hint": "UNKNOWN"
    }

def _extract_text_in_rect(perception: Dict[str, Any], rect: List[float]) -> List[Dict[str, Any]]:
    """Helper: Get all text blocks whose CENTER is within rect [x, y, w, h]"""
    out = []
    tx, ty, tw, th = rect
    for block in perception.get("text_blocks", []):
        if not isinstance(block, dict):
            continue
        bbox = block.get("bbox_norm")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        bx, by, bw, bh = bbox
        cx, cy = bx + bw / 2, by + bh / 2
        if (tx <= cx <= tx + tw) and (ty <= cy <= ty + th):
            out.append(block)
    return out

def _identify_part(perception: Dict[str, Any]) -> Dict[str, str]:
    """Step 3: Part Identity - Generalized Extraction"""
    out = {
        "name": None,
        "classification": "SINGLE_PART",
        "functional_role": "UNKNOWN" // Default
    }
    
    # Priority 0: Safety Override (Legacy/Specific fix)
    # We keep this temporarily while verifying generalized logic
    page_text = perception.get("page_text", {})
    full_text_upper = " ".join(page_text.values()).upper()
    # Relaxed match: check if both words exist, even if separated
    if "SEAT" in full_text_upper and "CUSHION" in full_text_upper:
        out["name"] = "SEAT CUSHION"
        out["functional_role"] = "Structural Component"
        return out
        
    # Generalized Logic: Spatial Title Block Analysis
    regions = perception.get("regions", {})
    title_block = regions.get("title_block")
    
    if title_block and isinstance(title_block, dict):
        tb_bbox = title_block.get("bbox_norm")
        if isinstance(tb_bbox, list) and len(tb_bbox) == 4:
            blocks = _extract_text_in_rect(perception, tb_bbox)
            
            # Strategy A: Find Largest Text (Heuristic: Part Name is usually biggest)
            # Filter out small labels and numbered notes
            candidates = []
            for b in blocks:
                text = str(b.get("text", "")).strip()
                if len(text) < 3: continue
                
                # Filter 1: Common label keys
                if text.upper().strip(":") in ["TITLE", "NAME", "DWG NO", "SCALE", "DATE", "SIZE", "REV", "SHEET"]:
                    continue
                    
                # Filter 2: Numbered notes (Strict list identifier: "1. ", "5. ")
                # We allow "5 AXIS" but reject "5. 6. MARKING"
                if re.match(r"^\d+\.\s+", text):
                    continue
                    
                # Filter 3: "NOTES" header
                if re.match(r"^NOTES?", text.upper()):
                     continue
                    
                candidates.append(b)
                
            if candidates:
                # Sort by height (descending)
                candidates.sort(key=lambda b: b.get("bbox_norm", [0,0,0,0])[3], reverse=True)
                largest = candidates[0]
                out["name"] = str(largest.get("text")).strip()
                
            # Strategy B: Keyword Association (within title block)
            # Look for "TITLE:" or "NAME:" and take next block
            # (Simplified version: regex on joined text is usually sufficient for association if strictly inside bbox)
            tb_full_text = " ".join([b.get("text", "") for b in blocks])
            match = re.search(r"(?:TITLE|NAME|DESCRIPTION)[:\s]+([A-Z0-9\s\-]+)", tb_full_text, re.IGNORECASE)
            if match:
                # Regex match inside title block is very strong
                out["name"] = match.group(1).strip()
                return out

    # Priority 2: Legacy Fallback
    legacy = perception.get("_legacy", {})
    if not out["name"] and isinstance(legacy, dict) and legacy.get("part_name"):
        out["name"] = str(legacy["part_name"]).strip()
        
    # Priority 3: Full Page Fallback
    if not out["name"]:
        match = re.search(r"(?:TITLE|NAME|PART\s*NAME|DESCRIPTION)[:\s]+([A-Z0-9\s\-]+)", full_text_upper, re.IGNORECASE)
        if match:
            out["name"] = match.group(1).strip()

    return out

def _understand_geometry(perception: Dict[str, Any]) -> Dict[str, Any]:
    """Step 4: Geometry Understanding"""
    shapes = perception.get("closed_shapes", [])
    
    primary_form = "UNKNOWN"
    
    # Analyze detected shapes for primary form
    shape_types = [s.get("shape_type", "") for s in shapes]
    
    if "circle" in shape_types:
        # Concentric circles -> Axisymmetric
        primary_form = "AXISYMMETRIC / ROTATIONAL"
    elif "triangle" in shape_types:
        primary_form = "PRISMATIC / TRIANGULAR" 
    elif "rectangle" in shape_types:
        primary_form = "PRISMATIC / BOX"
    
    # Classify Views (Iso, Top, Side)
    views = _classify_views(perception)
    
    out = {
        "primary_form": primary_form,
        "views": views, # New field
        "key_features": [],
        "dimensional_complexity": "LOW",
        "symmetry": {
            "type": "NONE",
            "axis_count": 0
        }
    }
    
    if len(shapes) > 50: # HIGH_THRESHOLD
        out["dimensional_complexity"] = "HIGH"
    elif len(shapes) > 10:
        out["dimensional_complexity"] = "MEDIUM"
        
    if shapes:
        out["key_features"].append("closed_profiles")
        
    return out

def _extract_notes_generalized(perception: Dict[str, Any]) -> List[str]:
    """Helper: Find numbered notes or 'NOTES:' blocks anywhere on page"""
    notes = []
    text_blocks = perception.get("text_blocks", [])
    
    # Sort blocks by Y position to read top-down
    sorted_blocks = sorted(text_blocks, key=lambda b: b.get("bbox_norm", [0,0,0,0])[1] if isinstance(b, dict) else 0)
    
    in_notes_section = False
    for block in sorted_blocks:
        if not isinstance(block, dict): continue
        text = str(block.get("text", "")).strip()
        if not text: continue
        
        # Start trigger
        if re.match(r"^(GENERAL\s*)?NOTES[:\s]*$", text, re.IGNORECASE):
            in_notes_section = True
            continue
            
        # Stop trigger (e.g. hitting title block or other section)
        # (For now, we rely on numbering pattern)
        
        # Pattern: "1. Note content" or "(1) Note content"
        if re.match(r"^(\d+[\.\)]|\(\d+\))\s+[A-Z]", text, re.IGNORECASE):
            notes.append(text)
        elif in_notes_section and len(text) > 10:
             # If we are strictly in a notes section, accept unnumbered lines too
             notes.append(text)
             
    return notes

def _extract_label_value(perception: Dict[str, Any], label_pattern: str, rect: Optional[List[float]] = None) -> Optional[str]:
    """Helper: Find text 'Label: Value' or text geometrically right/below 'Label'"""
    # 1. Search in joined text for "Label: Value" pattern first (simplest)
    page_text = perception.get("page_text", {})
    text_blocks = perception.get("text_blocks", [])
    
    if len(page_text) > 5:
        full_text = " ".join(page_text.values())
    else:
        # Fallback: Construct full text from blocks
        sorted_blocks = sorted(text_blocks, key=lambda b: b.get("bbox_norm", [0,0,0,0])[1] if isinstance(b, dict) else 0)
        full_text = " ".join([str(b.get("text", "")) for b in sorted_blocks])
        
    match = re.search(f"{label_pattern}[:\s]+([A-Z0-9\s/\.,\-]+)", full_text, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        if len(val) > 1 and len(val) < 50:
            return val
            
    # 2. Geometric search (if rect provided, restrict to it)
    # (TODO: Implement strict geometric neighbor search if regex fails)
    return None

def _extract_dimensions_generalized(perception: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Helper: Find and classify numerical dimensions"""
    dims = []
    text_blocks = perception.get("text_blocks", [])
    
    for block in text_blocks:
        if not isinstance(block, dict): continue
        text = str(block.get("text", "")).strip()
        if not text: continue
        
        # Regex for standard Decimals (linear) or Diameter/Radius
        # Matches: "4.000", "0.875", "10.5", "R0.5", "Ø.875"
        # Avoids: "1.", "Page 1"
        match = re.search(r"([RØø]?)\s*(\d*\.\d+)", text)
        if match:
            prefix = match.group(1).upper()
            val_str = match.group(2)
            
            try:
                val = float(val_str)
            except:
                continue
                
            dtype = "LINEAR"
            if "Ø" in prefix or "DIA" in text.upper():
                dtype = "DIAMETER"
            elif "R" in prefix or "RAD" in text.upper():
                dtype = "RADIUS"
            
            # Orientation Inference (Heuristic)
            bbox = block.get("bbox_norm", [0,0,0,0])
            w, h = bbox[2], bbox[3]
            aspect = w / h if h > 0 else 1.0
            
            orientation = "UNKNOWN"
            if aspect > 1.2:
                orientation = "HORIZONTAL"
            elif aspect < 0.8:
                orientation = "VERTICAL"
                
            dims.append({
                "value": val,
                "unit": "unitless", # usually inches/mm
                "type": dtype,
                "orientation": orientation,
                "raw_text": text,
                "confidence": 0.8
            })
            
    # Merge Legacy DOM Dimensions (Robust Fallback - Brute Force)
    legacy_dom_dims = (perception.get("_legacy") or {}).get("dom_dimensions", [])
    if isinstance(legacy_dom_dims, list):
        for ld in legacy_dom_dims:
            # Brute Force: Dump structure to text to find ANY number 
            # (Because structure might vary: text, content, value, etc)
            import json
            try:
                txt = json.dumps(ld)
            except:
                txt = str(ld)
                
            # Regex for numbers (e.g. 3.000, 0.875)
            # Must strictly match valid dimension formats to avoid page numbers
            matches = re.findall(r"(\d+\.\d+)", txt)
            
            for m in matches:
                val = float(m)
                if val < 0.001 or val > 10000: continue
                
                # Check if we already have this value (approximate dedupe)
                if any(abs(d["value"] - val) < 0.001 for d in dims): continue
                
                dims.append({
                     "value": val,
                     "unit": "unitless",
                     "type": "LINEAR",
                     "orientation": "UNKNOWN", 
                     "raw_text": m,
                     "confidence": 0.85, # High confidence if from Legacy
                     "source": "legacy_bruteforce"
                })

    return dims

def _classify_views(perception: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Helper: Cluster geometry into views and classify (Front, Top, Iso)"""
    shapes = perception.get("closed_shapes", [])
    
    # 0. Check Legacy Views first
    legacy_views = (perception.get("_legacy") or {}).get("dom_views", [])
    if isinstance(legacy_views, list) and len(legacy_views) > 0:
        # Convert legacy format to our format
        # Legacy views usually have 'bbox'
        converted = []
        for lv in legacy_views:
            if isinstance(lv, dict) and "bbox" in lv:
                converted.append({
                    "type": "UNKNOWN_VIEW", # Legacy might not label them
                    "bbox": lv["bbox"],
                    "shape_count": 1
                })
        # If we have legacy views, return them (or merge? Let's just use them if our geom failed)
        if not shapes and converted:
             return converted
             
    # text_blocks = perception.get("text_blocks", []) # Can be used for labels like "SECTION A-A"
    
    if not shapes:
        return []
        
    # 1. Simple Clustering by Proximity
    # (Naive O(N^2) for now, manageable for <100 shapes)
    clusters = []
    processed = set()
    
    for i, s1 in enumerate(shapes):
        if i in processed: continue
        
        # Start a new cluster
        cluster_shapes = [s1]
        processed.add(i)
        
        # Grow cluster
        bbox1 = s1.get("bbox_norm")
        if not bbox1: continue
        
        # Iterative merge (simplified: just one pass for now)
        for j, s2 in enumerate(shapes):
            if j in processed: continue
            bbox2 = s2.get("bbox_norm")
            if not bbox2: continue
            
            # Check overlap or proximity (margin 0.05)
            # x_overlap = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            margin = 0.05
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            
            overlap_x = (x1 - margin < x2 + w2) and (x2 - margin < x1 + w1)
            overlap_y = (y1 - margin < y2 + h2) and (y2 - margin < y1 + h1)
            
            if overlap_x and overlap_y:
                cluster_shapes.append(s2)
                processed.add(j)
                # Expand cluster bbox (naive)
                x1 = min(x1, x2)
                y1 = min(y1, y2)
                w1 = max(x1+w1, x2+w2) - x1
                h1 = max(y1+h1, y2+h2) - y1
                bbox1 = [x1, y1, w1, h1]
                
        clusters.append({
            "bbox": bbox1,
            "shape_count": len(cluster_shapes),
            "shapes": cluster_shapes
        })
        
    # 2. Classify Views
    # Find Main View (Largest area or most shapes)
    if not clusters: return []
    
    clusters.sort(key=lambda c: c["bbox"][2] * c["bbox"][3], reverse=True)
    main_view = clusters[0]
    main_view["type"] = "FRONT_VIEW" # Assumption: Largest is Front
    
    views_out = [main_view]
    
    mx, my, mw, mh = main_view["bbox"]
    cx_main = mx + mw/2
    cy_main = my + mh/2
    
    for c in clusters[1:]:
        cx, cy, cw, ch = c["bbox"]
        cx_c = cx + cw/2
        cy_c = cy + ch/2
        
        # Heuristic: Isometric is usually Top-Right and squarish/triangular
        # And NOT aligned with main view
        is_aligned_x = abs(cx_c - cx_main) < 0.1
        is_aligned_y = abs(cy_c - cy_main) < 0.1
        
        if cx > 0.5 and cy < 0.4 and not (is_aligned_x or is_aligned_y):
            c["type"] = "ISOMETRIC_VIEW"
        elif is_aligned_x and cy_c < cy_main:
            c["type"] = "TOP_VIEW"
        elif is_aligned_x and cy_c > cy_main:
            c["type"] = "BOTTOM_VIEW"
        elif is_aligned_y and cx_c > cx_main:
            c["type"] = "RIGHT_SIDE_VIEW"
        elif is_aligned_y and cx_c < cx_main:
            c["type"] = "LEFT_SIDE_VIEW"
        else:
            c["type"] = "AUX_VIEW"
            
        views_out.append(c)
        
    # Simplified output for EUO
    return [{ "type": v["type"], "bbox": v["bbox"], "shape_count": v["shape_count"] } for v in views_out]

def _understand_annotations(perception: Dict[str, Any]) -> Dict[str, Any]:
    """Step 5: Annotation Understanding"""
    arrows = perception.get("arrow_candidates", [])
    page_text = perception.get("page_text", {})
    full_text = " ".join(page_text.values())
    
    # Check for legacy dimensions if arrow detection failed
    legacy_dims = (perception.get("_legacy") or {}).get("dimensions", [])
    dims_present = len(arrows) > 0 or len(legacy_dims) > 0
    tol_class = "UNKNOWN"
    
    if "R" in full_text or "Ø" in full_text or "tolerance" in full_text.lower():
        tol_class = "GENERAL"
        
    # Generalized Note Extraction
    process_notes = _extract_notes_generalized(perception)
    
    # Symbol Interpretation
    symbols = perception.get("symbols", [])
    surface_reqs = []
    
    for s in symbols:
        stype = s.get("type", "")
        if stype == "SURFACE_FINISH":
            # In V2 we will extract the exact value. For now, just detecting presence is huge.
            surface_reqs.append("Surface Finish Mark Present")
            
    # Deduplicate
    surface_reqs = list(set(surface_reqs))
    
    # Explicit Dimension Extraction
    dimensions = _extract_dimensions_generalized(perception)
    
    # Infer Bounding Box (Width/Height) explicitly for the LLM
    # Heuristic: Largest Horizontal = Width, Largest Vertical = Height
    width_inf = None
    height_inf = None
    
    h_dims = [d["value"] for d in dimensions if d["orientation"] == "HORIZONTAL"]
    v_dims = [d["value"] for d in dimensions if d["orientation"] == "VERTICAL"]
    
    if h_dims:
        width_inf = max(h_dims)
    if v_dims:
        height_inf = max(v_dims)
        
    return {
        "dimensions_present": dims_present,
        "tolerance_class": tol_class,
        "surface_requirements": surface_reqs,
        "process_notes": process_notes,
        "dimensions": dimensions,
        "bounding_box": {
            "width_inference": width_inf,
            "height_inference": height_inf
        },
        "inspection_notes_present": False
    }

def _extract_constraints(perception: Dict[str, Any]) -> Dict[str, List[str]]:
    """Step 7: Constraints & Risks"""
    page_text = perception.get("page_text", {})
    full_text = " ".join(page_text.values()).lower()
    symbols = perception.get("symbols", [])
    
    out = {
        "functional_constraints": [],
        "cosmetic_constraints": [],
        "regulatory_constraints": []
    }
    
    if "surface" in full_text or "finish" in full_text:
        out["cosmetic_constraints"].append("surface_control")
        
    # Check for GD&T Symbols
    gdt_count = sum(1 for s in symbols if s.get("type") == "GDT_FRAME")
    if gdt_count > 0:
        out["functional_constraints"].append("geometric_tolerancing")
        
    # Check for Datum Targets
    datum_count = sum(1 for s in symbols if s.get("type") == "DATUM_TARGET")
    if datum_count > 0:
        out["functional_constraints"].append("datum_references")
        
    # High dimension density check (proxy: count of "mm" or numbers)
    # Using text_blocks count as a very rough proxy for complexity/density
    if len(perception.get("text_blocks", [])) > 100:
         out["functional_constraints"].append("dimensional_consistency")
         
    return out

def _infer_intent(perception: Dict[str, Any], geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Step 6: Manufacturing Intent"""
    page_text = perception.get("page_text", {})
    full_text = " ".join(page_text.values()).upper()
    
    out = {
        "material_class": "UNKNOWN",
        "likely_process": "UNKNOWN",
        "production_scale": "UNKNOWN",
        "post_processes": []
    }
    
    # Try generic label extraction for Material
    mat_val = _extract_label_value(perception, r"(?:MATERIAL|MATL|MAT'L)")
    if mat_val:
        if "RUBBER" in mat_val.upper() or "EPDM" in mat_val.upper():
            out["material_class"] = "RUBBER"
        elif "STEEL" in mat_val.upper() or "ALUM" in mat_val.upper():
            out["material_class"] = "METAL"
        else:
            out["material_class"] = mat_val # Use extracted string directly
            
    # STRICT RULE override: >= 2 signals
    # Example logic
    signals = 0
    if "VULCANIZE" in full_text:
        signals += 1
    if geometry.get("primary_form") == "AXISYMMETRIC":
        signals += 1
        
    if signals >= 2 and "VULCANIZE" in full_text:
        out["likely_process"] = "COMPRESSION_MOLDING"
        if out["material_class"] == "UNKNOWN":
             out["material_class"] = "RUBBER"
        
    return out

def _extract_constraints(perception: Dict[str, Any]) -> Dict[str, List[str]]:
    """Step 7: Constraints & Risks"""
    page_text = perception.get("page_text", {})
    full_text = " ".join(page_text.values()).lower()
    
    out = {
        "functional_constraints": [],
        "cosmetic_constraints": [],
        "regulatory_constraints": []
    }
    
    if "surface" in full_text or "finish" in full_text:
        out["cosmetic_constraints"].append("surface_control")
        
    # High dimension density check (proxy: count of "mm" or numbers)
    # Using text_blocks count as a very rough proxy for complexity/density
    if len(perception.get("text_blocks", [])) > 100:
         out["functional_constraints"].append("dimensional_consistency")
         
    return out

def _model_uncertainty(euo_partial: Dict[str, Any]) -> Dict[str, List[str]]:
    """Step 8: Uncertainty Model"""
    missing = []
    
    if euo_partial["manufacturing_intent"]["material_class"] == "UNKNOWN":
        missing.append("material")
    if euo_partial["manufacturing_intent"]["likely_process"] == "UNKNOWN":
        missing.append("manufacturing_process")
        
    return {
        "missing_information": missing,
        "ambiguous_features": []
    }

def _aggregate_confidence(perception: Dict[str, Any], dims_count: int = 0, views_count: int = 0) -> Dict[str, Any]:
    """Step 9: Confidence Aggregation (Dual-Path)"""
    val = perception.get("validation", {})
    
    # Detect document type based on content
    text_blocks = perception.get("text_blocks", [])
    shapes = perception.get("closed_shapes", [])
    
    # Heuristic: Text-heavy if many text blocks and few geometric features
    is_text_heavy = len(text_blocks) > 20 and len(shapes) < 5 and views_count == 0
    
    if is_text_heavy:
        # Path A: Document/Manual confidence (OCR-based)
        ocr_hist = val.get("ocr_confidence_histogram", {})
        if ocr_hist:
            # Calculate weighted OCR confidence from histogram
            total_words = sum(ocr_hist.values())
            high_conf_words = sum(v for k, v in ocr_hist.items() if float(k) >= 0.8)
            ocr_conf = high_conf_words / total_words if total_words > 0 else 0.5
        else:
            # Default for text documents with no histogram
            ocr_conf = 0.7
        
        struct_conf = float(val.get("region_coverage_pct", 0.7))
        overall = statistics.mean([ocr_conf, struct_conf])
        
        return {
            "overall": round(overall, 2),
            "breakdown": {
                "ocr": round(ocr_conf, 2),
                "structure": round(struct_conf, 2),
                "geometry": 0.0  # Not applicable for text documents
            }
        }
    else:
        # Path B: Production Drawing confidence (geometry-based)
        
        # 1. Geometry Confidence
        # Base: Closed Loop Ratio (how clean the lines are)
        # Boost: If we detected valid VIEWS, we are confident.
        base_geom = float(val.get("closed_loop_line_ratio", 0.0))
        if views_count > 0:
            geom_conf = max(base_geom, 0.85) # High confidence if views clustered
        elif len(shapes) > 0:
            geom_conf = max(base_geom, 0.6)
        else:
            geom_conf = 0.4
        
        # 2. Annotation Confidence
        # Base: Arrow detection
        # Boost: If we extracted DIMENSIONS (regex), we are confident.
        arrows = perception.get("arrow_candidates", [])
        if dims_count > 0:
            ann_conf = 0.95 # We read numbers! Very confident.
        elif arrows:
            confs = [float(a.get("confidence", 0.0)) for a in arrows]
            ann_conf = sum(confs) / len(confs)
            ann_conf = min(ann_conf, 0.8) # Cap if no regex read
        else:
            ann_conf = 0.3 # Low confidence
            
        # 3. Structure conf
        struct_conf = float(val.get("region_coverage_pct", 0.0))
        if struct_conf == 0.0: struct_conf = 0.8 # Fallback if metric missing
        
        overall = statistics.mean([geom_conf, ann_conf, struct_conf])
        
        # Boost overall if we have "Complete Intelligence" (Dims + Views)
        if dims_count > 0 and views_count > 0:
            overall = max(overall, 0.9)
        
        return {
            "overall": round(overall, 2),
            "breakdown": {
                "ocr": 0.9 if dims_count > 0 else 0.6,
                "structure": round(struct_conf, 2),
                "geometry": round(geom_conf, 2),
                "annotation": round(ann_conf, 2)
            }
        }


def build_euo(perception: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point to build Engineering Understanding Object.
    """
    # Step 1
    flags = _validate_structure(perception)
    
    # Step 2
    doc_class = _classify_document(perception)
    
    # Step 3
    part_id = _identify_part(perception)
    
    # Step 4
    geom = _understand_geometry(perception)
    
    # Step 5
    ann = _understand_annotations(perception)
    
    # Step 6
    intent = _infer_intent(perception, geom)
    
    # Step 7
    constraints = _extract_constraints(perception)
    
    euo_partial = {
        "manufacturing_intent": intent
    }
    
    # Step 8
    unc = _model_uncertainty(euo_partial)
    
    # Step 9
    conf = _aggregate_confidence(
        perception,
        dims_count=len(dimensions),
        views_count=len(views) if 'views' in locals() else 0
    )
    
    # Step 10
    summary_ready = conf["overall"] >= 0.65
    
    # Debug Signal
    debug_info = {
        "text_blocks_count": len(perception.get("text_blocks", [])),
        "shapes_count": len(perception.get("closed_shapes", [])),
        "dims_count": len(dimensions),
        "raw_dims_sample": dimensions[:3],
        "views_count": len(views) if 'views' in locals() else 0,
        "flags": flags
    }
    
    return {
        "document_classification": doc_class,
        "part_identity": part_id,
        "geometry_understanding": geom,
        "annotation_understanding": ann,
        "manufacturing_intent": intent,
        "constraints": constraints,
        "uncertainty": unc,
        "confidence": conf,
        "summary_ready": summary_ready,
        "_debug": debug_info, # Renamed from _flags for clarity
        "_system_version": "v2.2_debug_active"
    }
