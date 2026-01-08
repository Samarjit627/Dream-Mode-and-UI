"""
Perception-only router for signals extraction from engineering drawings.
Returns structured JSON with NO semantic reasoning or intelligence.
"""
import os
import uuid
import tempfile
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

router = APIRouter(prefix="/api/v1/perception", tags=["perception"])


def _run_perception_pipeline(file_path: str) -> Dict[str, Any]:
    """
    Run the signals-only perception pipeline on a file.
    Returns structured JSON matching the user's strict schema.
    """
    from .processing import (
        _is_pdf,
        _render_pdf_to_images,
        _perception_paddleocr,
        _perception_opencv_geometry,
        _perception_regions_from_signals,
        _perception_dimension_candidates,
        _perception_text_blocks_to_page_text,
    )
    from .utils import read_image_from_path

    result: Dict[str, Any] = {
        "image_metadata": {"width_px": 0, "height_px": 0, "dpi": None, "pages": {}},
        "text_blocks": [],
        "lines": [],
        "closed_shapes": [],
        "arrow_candidates": [],
        "regions": {"title_block": None, "viewports": [], "notes": []},
        "dimension_candidates": [],
        "page_text": {},
        "validation": {
            "ocr_confidence_histogram": {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0},
            "closed_loop_line_ratio": 0.0,
            "region_coverage_pct": 0.0,
            "failure_reasons": [],
        },
    }

    try:
        is_pdf = _is_pdf(file_path)
        page_paths: List[str] = []

        if is_pdf:
            page_paths = _render_pdf_to_images(file_path, max_pages=10, dpi=300)
            if not page_paths:
                result["validation"]["failure_reasons"].append("pdf_render_failed")
                return result
        else:
            page_paths = [file_path]

        # Process each page
        for page_idx, page_path in enumerate(page_paths):
            try:
                import numpy as np
                from PIL import Image
                
                # Read image and convert to numpy array (HxWx3)
                pil_img = Image.open(page_path).convert("RGB")
                rgb_arr = np.array(pil_img)
                
                if rgb_arr is None or rgb_arr.size == 0:
                    result["validation"]["failure_reasons"].append(f"page_{page_idx}_read_failed")
                    continue

                h, w = rgb_arr.shape[:2]
                result["image_metadata"]["pages"][str(page_idx)] = {
                    "width_px": int(w),
                    "height_px": int(h),
                    "dpi": 300 if is_pdf else None,
                }

                # Set main dimensions from first page
                if page_idx == 0:
                    result["image_metadata"]["width_px"] = int(w)
                    result["image_metadata"]["height_px"] = int(h)
                    result["image_metadata"]["dpi"] = 300 if is_pdf else None

                # OCR with PaddleOCR
                ocr_result = _perception_paddleocr(rgb_arr, page_idx, dpi=300.0 if is_pdf else 0.0)
                text_blocks = ocr_result.get("text_blocks") or []
                result["text_blocks"].extend(text_blocks)

                # Update OCR confidence histogram
                metrics = ocr_result.get("_metrics") or {}
                hist = metrics.get("conf_hist") or {}
                for k, v in hist.items():
                    if k in result["validation"]["ocr_confidence_histogram"]:
                        result["validation"]["ocr_confidence_histogram"][k] += int(v)

                # Geometry detection with OpenCV
                geom_result = _perception_opencv_geometry(rgb_arr, page_idx)
                result["lines"].extend(geom_result.get("lines") or [])
                result["closed_shapes"].extend(geom_result.get("closed_shapes") or [])
                result["arrow_candidates"].extend(geom_result.get("arrow_candidates") or [])

                # Dimension candidates
                dim_cands = _perception_dimension_candidates(
                    page_idx=page_idx,
                    w_px=w,
                    h_px=h,
                    text_blocks=text_blocks,
                    arrow_candidates=geom_result.get("arrow_candidates") or [],
                )
                result["dimension_candidates"].extend(dim_cands)

                # Region detection (heuristic)
                regions = _perception_regions_from_signals(
                    page_idx=page_idx,
                    w_px=w,
                    h_px=h,
                    text_blocks=text_blocks,
                    lines=geom_result.get("lines") or [],
                )

                # Merge regions
                if regions.get("title_block") and not result["regions"]["title_block"]:
                    result["regions"]["title_block"] = regions["title_block"]
                if regions.get("viewports"):
                    result["regions"]["viewports"].extend(regions["viewports"])
                if regions.get("notes"):
                    result["regions"]["notes"].extend(regions["notes"])

                # Track coverage
                cov = regions.get("coverage", 0.0)
                result["validation"]["region_coverage_pct"] = max(
                    result["validation"]["region_coverage_pct"], float(cov)
                )

            except Exception as e:
                result["validation"]["failure_reasons"].append(f"page_{page_idx}_error:{str(e)}")

        # Calculate closed loop ratio
        line_count = len(result["lines"])
        closed_count = len(result["closed_shapes"])
        if line_count > 0:
            result["validation"]["closed_loop_line_ratio"] = min(1.0, closed_count / float(line_count))

        # Generate page text
        result["page_text"] = _perception_text_blocks_to_page_text(
            result["text_blocks"], max_chars_per_page=12000
        )

    except Exception as e:
        result["validation"]["failure_reasons"].append(f"pipeline_error:{str(e)}")

    return result


@router.post("/analyze")
async def analyze_perception(file: UploadFile = File(...)):
    """
    Analyze an engineering drawing and return signals-only perception data.
    
    NO semantic reasoning. NO summarization. NO CAD inference.
    Only structured signals: text blocks, lines, shapes, regions.
    """
    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename or "upload")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        perception_result = _run_perception_pipeline(tmp_path)
        
        # Build EUO from perception signals
        from .euo_builder import build_euo
        euo = build_euo(perception_result)
        
        # Structure final output
        final_output = {
            "engineering_understanding_object": euo,
            "perception_debug": perception_result 
        }
        
        return JSONResponse(final_output)
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
