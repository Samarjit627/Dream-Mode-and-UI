import uuid
import os
import json
import re
import math
from datetime import datetime
from threading import Thread
from typing import Dict, Any, List, Tuple

from .utils import read_image_from_path, dominant_color_hex
from .models_loader import ModelRegistry
from .label_taxonomy import get_labels


_CLIP_TAX_CACHE: Dict[int, Dict[str, Any]] = {}


def _is_pdf(path: str) -> bool:
    try:
        if path.lower().endswith('.pdf'):
            return True
        with open(path, 'rb') as f:
            head = f.read(5)
        return head == b'%PDF-'
    except Exception:
        return False


def _render_pdf_to_images(pdf_path: str, max_pages: int = 10, dpi: int = 250) -> List[str]:
    out_paths: List[str] = []
    used_pdfplumber = False
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages], start=1):
                try:
                    im = page.to_image(resolution=int(dpi))
                    out = f"{pdf_path}.page{i}.png"
                    im.save(out, format='PNG')
                    out_paths.append(out)
                    used_pdfplumber = True
                except Exception:
                    continue
    except Exception:
        used_pdfplumber = False

    if out_paths:
        return out_paths

    try:
        from pdf2image import convert_from_path  # type: ignore
        pages = convert_from_path(pdf_path, dpi=dpi, fmt='png', thread_count=2)
        for i, p in enumerate(pages[:max_pages], start=1):
            out = f"{pdf_path}.page{i}.png"
            p.save(out)
            out_paths.append(out)
        return out_paths
    except Exception:
        return out_paths


def _angle_from_quad(quad: List[List[float]]) -> float:
    try:
        if not (isinstance(quad, list) and len(quad) == 4):
            return 0.0
        p0 = quad[0]
        p1 = quad[1]
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        ang = math.degrees(math.atan2((y1 - y0), (x1 - x0)))
        while ang <= -180.0:
            ang += 360.0
        while ang > 180.0:
            ang -= 360.0
        return float(ang)
    except Exception:
        return 0.0


def _bbox_norm_from_px(x1: float, y1: float, x2: float, y2: float, w: float, h: float) -> List[float]:
    try:
        ww = float(max(1e-6, w))
        hh = float(max(1e-6, h))
        x1c = max(0.0, min(ww, float(x1)))
        y1c = max(0.0, min(hh, float(y1)))
        x2c = max(0.0, min(ww, float(x2)))
        y2c = max(0.0, min(hh, float(y2)))
        x = x1c / ww
        y = y1c / hh
        bw = max(0.0, (x2c - x1c) / ww)
        bh = max(0.0, (y2c - y1c) / hh)
        return [float(x), float(y), float(bw), float(bh)]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def _perception_opencv_geometry(rgb_arr, page_idx: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "lines": [],
        "closed_shapes": [],
        "arrow_candidates": [],
        "_metrics": {"line_count": 0, "closed_shape_count": 0, "arrow_count": 0, "closed_loop_line_ratio": 0.0},
    }
    try:
        import cv2  # type: ignore
        import numpy as np
        if rgb_arr is None:
            return out
        h, w = rgb_arr.shape[:2]
        gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 60, 180)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=90, minLineLength=max(25, w // 40), maxLineGap=10)
        line_items: List[Dict[str, Any]] = []
        if lines is not None:
            for ln in lines[:800]:
                x1, y1, x2, y2 = [float(v) for v in ln[0]]
                length = float(math.hypot(x2 - x1, y2 - y1))
                line_items.append({
                    "page": int(page_idx),
                    "start": [x1, y1],
                    "end": [x2, y2],
                    "length": length,
                })
        out["lines"] = line_items

        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 11)
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        closed_shapes: List[Dict[str, Any]] = []
        for i, c in enumerate(cnts[:800]):
            area = float(cv2.contourArea(c))
            if area < 120.0:
                continue
            peri = float(cv2.arcLength(c, True))
            if peri <= 0:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) < 3:
                continue
            closed_shapes.append({
                "page": int(page_idx),
                "contour_id": f"c{page_idx}_{i}",
                "area": float(area),
            })
        out["closed_shapes"] = closed_shapes

        arrow_candidates: List[Dict[str, Any]] = []
        for i, c in enumerate(cnts[:800]):
            area = float(cv2.contourArea(c))
            if area < 40.0 or area > 800.0:
                continue
            peri = float(cv2.arcLength(c, True))
            if peri <= 0:
                continue
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) != 3:
                continue
            x, y, bw, bh = cv2.boundingRect(approx)
            aspect = float(max(bw, bh)) / float(max(1, min(bw, bh)))
            conf = 0.0
            if aspect >= 1.2:
                conf += 0.4
            if area <= 300.0:
                conf += 0.4
            if conf >= 0.5:
                arrow_candidates.append({
                    "page": int(page_idx),
                    "bbox_px": [float(x), float(y), float(x + bw), float(y + bh)],
                    "bbox_norm": _bbox_norm_from_px(float(x), float(y), float(x + bw), float(y + bh), float(w), float(h)),
                    "confidence": float(min(1.0, conf)),
                })
        out["arrow_candidates"] = arrow_candidates

        out["_metrics"] = {
            "line_count": int(len(line_items)),
            "closed_shape_count": int(len(closed_shapes)),
            "arrow_count": int(len(arrow_candidates)),
            "closed_loop_line_ratio": float(min(1.0, len(closed_shapes) / float(max(1, len(line_items))))),
        }
        return out
    except Exception:
        return out


def _perception_regions_from_signals(page_idx: int, w_px: int, h_px: int, text_blocks: List[Dict[str, Any]], lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Deterministic, perception-only region proposal. No semantics beyond region types.
    # Uses OCR density + line density to propose a title block and main viewport.
    try:
        # Normalize to [x,y,w,h] in 0..1 for region geometry
        tb = []
        for b in text_blocks:
            if not (isinstance(b, dict) and int(b.get('page') or 0) == int(page_idx)):
                continue
            bn = b.get('bbox_norm')
            if isinstance(bn, list) and len(bn) == 4:
                tb.append([float(bn[0]), float(bn[1]), float(bn[2]), float(bn[3])])

        # Candidate title block: dense text in bottom band
        bottom_blocks = [bb for bb in tb if (bb[1] + bb[3] * 0.5) >= 0.72]
        tb_region = None
        tb_conf = 0.0
        if bottom_blocks:
            x1 = min(bb[0] for bb in bottom_blocks)
            y1 = min(bb[1] for bb in bottom_blocks)
            x2 = max(bb[0] + bb[2] for bb in bottom_blocks)
            y2 = max(bb[1] + bb[3] for bb in bottom_blocks)
            # clamp
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            area = max(0.0, (x2 - x1) * (y2 - y1))
            # confidence: density + compactness
            density = len(bottom_blocks) / max(1.0, area * 200.0)
            tb_conf = float(max(0.0, min(1.0, 0.35 + 0.45 * min(1.0, density) + 0.20 * min(1.0, len(bottom_blocks) / 40.0))))
            tb_region = {"page": int(page_idx), "bbox_norm": [x1, y1, x2 - x1, y2 - y1], "confidence": tb_conf}

        # Viewport: main region excluding title block and margins
        vp_bbox = [0.04, 0.04, 0.92, 0.92]
        if tb_region and isinstance(tb_region.get('bbox_norm'), list) and len(tb_region['bbox_norm']) == 4:
            tx, ty, tw, th = tb_region['bbox_norm']
            # reduce viewport height above title
            vp_bbox = [0.04, 0.04, 0.92, max(0.10, min(0.92, ty - 0.06))]

        # viewport confidence from line density
        vp_lines = []
        for ln in lines:
            if not (isinstance(ln, dict) and int(ln.get('page') or 0) == int(page_idx)):
                continue
            s = ln.get('start')
            e = ln.get('end')
            if not (isinstance(s, list) and isinstance(e, list) and len(s) == 2 and len(e) == 2):
                continue
            mx = (float(s[0]) + float(e[0])) * 0.5 / float(max(1, w_px))
            my = (float(s[1]) + float(e[1])) * 0.5 / float(max(1, h_px))
            if (mx >= vp_bbox[0]) and (mx <= vp_bbox[0] + vp_bbox[2]) and (my >= vp_bbox[1]) and (my <= vp_bbox[1] + vp_bbox[3]):
                vp_lines.append(ln)
        vp_conf = float(max(0.0, min(1.0, 0.25 + 0.55 * min(1.0, len(vp_lines) / 120.0) + 0.20 * min(1.0, len(tb) / 120.0))))
        viewports = [{"page": int(page_idx), "bbox_norm": vp_bbox, "confidence": vp_conf}]

        # Notes: OCR-dense blocks not in title block
        notes = []
        if tb_region:
            tx, ty, tw, th = tb_region['bbox_norm']
            note_blocks = [bb for bb in tb if not (bb[0] >= tx and bb[0] <= tx + tw and bb[1] >= ty and bb[1] <= ty + th)]
        else:
            note_blocks = tb
        # pick a notes bbox only if there is clear density
        if len(note_blocks) >= 25:
            x1 = min(bb[0] for bb in note_blocks)
            y1 = min(bb[1] for bb in note_blocks)
            x2 = max(bb[0] + bb[2] for bb in note_blocks)
            y2 = max(bb[1] + bb[3] for bb in note_blocks)
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            area = max(0.0, (x2 - x1) * (y2 - y1))
            density = len(note_blocks) / max(1.0, area * 240.0)
            conf = float(max(0.0, min(1.0, 0.20 + 0.55 * min(1.0, density) + 0.25 * min(1.0, len(note_blocks) / 120.0))))
            notes.append({"page": int(page_idx), "bbox_norm": [x1, y1, x2 - x1, y2 - y1], "confidence": conf})

        # coverage
        def _area(bb):
            try:
                return float(max(0.0, float(bb[2]) * float(bb[3])))
            except Exception:
                return 0.0
        cov = _area(vp_bbox) + (_area(tb_region['bbox_norm']) if tb_region else 0.0) + sum(_area(n['bbox_norm']) for n in notes)
        cov = float(max(0.0, min(1.0, cov)))

        return {"title_block": tb_region, "viewports": viewports, "notes": notes, "coverage": cov}
    except Exception:
        return {"title_block": None, "viewports": [], "notes": [], "coverage": 0.0}


def _perception_text_blocks_to_page_text(text_blocks: List[Dict[str, Any]], max_chars_per_page: int = 12000) -> Dict[str, str]:
    try:
        out: Dict[str, str] = {}
        for b in text_blocks:
            if not isinstance(b, dict):
                continue
            pn = int(b.get('page') or 0)
            if pn <= 0:
                continue
            t = str(b.get('text') or '').strip()
            if not t:
                continue
            k = str(pn)
            if k not in out:
                out[k] = ''
            out[k] = (out[k] + ' ' + t).replace('\n', ' ').replace('\r', ' ')
            out[k] = re.sub(r"\s+", " ", out[k]).strip()
            if len(out[k]) > int(max_chars_per_page):
                out[k] = out[k][:int(max_chars_per_page)]
        return out
    except Exception:
        return {}


def _perception_dimension_candidates(page_idx: int, w_px: int, h_px: int, text_blocks: List[Dict[str, Any]], arrow_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Signals-only: pair numeric-looking OCR blocks with nearby arrow candidates.
    try:
        num_pat = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:\s*(?:mm|cm|m|in|inch|\"|°|deg))?$", re.IGNORECASE)
        blocks = []
        for b in text_blocks:
            if not (isinstance(b, dict) and int(b.get('page') or 0) == int(page_idx)):
                continue
            txt = str(b.get('text') or '').strip()
            if not txt:
                continue
            # skip very long blocks
            if len(txt) > 24:
                continue
            if not num_pat.match(txt.replace(' ', '')) and not num_pat.match(txt):
                continue
            bn = b.get('bbox_norm')
            bp = b.get('bbox_px')
            if isinstance(bn, list) and len(bn) == 4:
                x, y, w, h = map(float, bn)
                cx, cy = x + w * 0.5, y + h * 0.5
            elif isinstance(bp, list) and len(bp) == 4:
                x1, y1, x2, y2 = map(float, bp)
                cx, cy = (x1 + x2) * 0.5 / float(max(1, w_px)), (y1 + y2) * 0.5 / float(max(1, h_px))
                bn = _bbox_norm_from_px(x1, y1, x2, y2, float(w_px), float(h_px))
            else:
                continue
            blocks.append({"text": txt, "confidence": float(b.get('confidence') or 0.0), "center": [cx, cy], "bbox_norm": bn})

        arrows = []
        for a in arrow_candidates:
            if not (isinstance(a, dict) and int(a.get('page') or 0) == int(page_idx)):
                continue
            bn = a.get('bbox_norm')
            if not (isinstance(bn, list) and len(bn) == 4):
                continue
            x, y, w, h = map(float, bn)
            arrows.append({"bbox_norm": bn, "center": [x + w * 0.5, y + h * 0.5], "confidence": float(a.get('confidence') or 0.0)})

        out: List[Dict[str, Any]] = []
        for b in blocks:
            best = None
            best_score = 0.0
            for a in arrows:
                dx = float(b['center'][0]) - float(a['center'][0])
                dy = float(b['center'][1]) - float(a['center'][1])
                dist = float((dx * dx + dy * dy) ** 0.5)
                # within ~15% of page diagonal (normalized)
                if dist > 0.22:
                    continue
                score = (1.0 - min(1.0, dist / 0.22)) * 0.55 + float(a['confidence']) * 0.25 + float(b['confidence']) * 0.20
                if score > best_score:
                    best_score = score
                    best = a
            if best and best_score >= 0.45:
                out.append({
                    "page": int(page_idx),
                    "text": b['text'],
                    "text_confidence": float(b['confidence']),
                    "text_bbox_norm": b['bbox_norm'],
                    "arrow_bbox_norm": best['bbox_norm'],
                    "confidence": float(max(0.0, min(1.0, best_score))),
                })
        return out[:120]
    except Exception:
        return []


def _paddleocr_blocks_on_crop(rgb_arr, page_idx: int, crop_norm: List[float], scale: float = 2.0) -> List[Dict[str, Any]]:
    # Run PaddleOCR on an upscaled crop; map boxes back to full-image coordinates.
    try:
        import numpy as np
        from paddleocr import PaddleOCR  # type: ignore
        h, w = rgb_arr.shape[:2]
        x, y, bw, bh = [float(v) for v in crop_norm]
        x1 = int(max(0, min(w - 1, round(x * w))))
        y1 = int(max(0, min(h - 1, round(y * h))))
        x2 = int(max(1, min(w, round((x + bw) * w))))
        y2 = int(max(1, min(h, round((y + bh) * h))))
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            return []
        crop = rgb_arr[y1:y2, x1:x2]
        if scale and scale > 1.01:
            import cv2  # type: ignore
            crop = cv2.resize(crop, (int((x2 - x1) * scale), int((y2 - y1) * scale)), interpolation=cv2.INTER_CUBIC)

        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        rr = ocr.ocr(crop, cls=True)
        blocks: List[Dict[str, Any]] = []
        if isinstance(rr, list) and rr:
            items = rr[0] if (len(rr) == 1 and isinstance(rr[0], list)) else rr
            for item in items:
                try:
                    quad = item[0]
                    txt = item[1][0]
                    conf = float(item[1][1])
                except Exception:
                    continue
                if not txt:
                    continue
                # quad points are in crop coordinates
                xs = [float(p[0]) for p in quad]
                ys = [float(p[1]) for p in quad]
                cx1, cy1, cx2, cy2 = min(xs), min(ys), max(xs), max(ys)
                # map back to original pixel coords
                ox1 = float(x1) + float(cx1) / float(scale or 1.0)
                oy1 = float(y1) + float(cy1) / float(scale or 1.0)
                ox2 = float(x1) + float(cx2) / float(scale or 1.0)
                oy2 = float(y1) + float(cy2) / float(scale or 1.0)
                blocks.append({
                    "page": int(page_idx),
                    "text": str(txt),
                    "bbox_px": [ox1, oy1, ox2, oy2],
                    "bbox_norm": _bbox_norm_from_px(ox1, oy1, ox2, oy2, float(w), float(h)),
                    "angle": float(_angle_from_quad(quad)),
                    "confidence": float(conf),
                })
        return blocks
    except Exception:
        return []


def _perception_paddleocr(rgb_arr, page_idx: int, dpi: float = 0.0) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "text_blocks": [],
        "_metrics": {"engine": "paddleocr", "count": 0, "conf_hist": {}},
    }
    try:
        import numpy as np
        from paddleocr import PaddleOCR  # type: ignore
        if rgb_arr is None:
            return out
        h, w = rgb_arr.shape[:2]
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        rr = ocr.ocr(rgb_arr, cls=True)
        blocks: List[Dict[str, Any]] = []
        confs: List[float] = []
        if isinstance(rr, list) and rr:
            for item in rr[0] if (len(rr) == 1 and isinstance(rr[0], list)) else rr:
                try:
                    quad = item[0]
                    txt = item[1][0]
                    conf = float(item[1][1])
                except Exception:
                    continue
                if not txt:
                    continue
                xs = [float(p[0]) for p in quad]
                ys = [float(p[1]) for p in quad]
                x1, y1, x2, y2 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
                blocks.append({
                    "page": int(page_idx),
                    "text": str(txt),
                    "bbox_px": [x1, y1, x2, y2],
                    "bbox_norm": _bbox_norm_from_px(x1, y1, x2, y2, float(w), float(h)),
                    "angle": float(_angle_from_quad(quad)),
                    "confidence": float(conf),
                })
                confs.append(float(conf))

        hist: Dict[str, int] = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for c in confs:
            if c < 0.2:
                hist["0.0-0.2"] += 1
            elif c < 0.4:
                hist["0.2-0.4"] += 1
            elif c < 0.6:
                hist["0.4-0.6"] += 1
            elif c < 0.8:
                hist["0.6-0.8"] += 1
            else:
                hist["0.8-1.0"] += 1

        out["text_blocks"] = blocks
        out["_metrics"] = {"engine": "paddleocr", "count": int(len(blocks)), "conf_hist": hist}
        return out
    except Exception:
        return out


def _extract_pdf_text(pdf_path: str, max_pages: int = 10) -> List[Dict[str, Any]]:
    """Best-effort embedded text extraction. If PDF is 'digital text', this is far more accurate than OCR."""
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return []

    try:
        reader = PdfReader(pdf_path)
        out: List[Dict[str, Any]] = []
        for i, page in enumerate(reader.pages[:max_pages], start=1):
            try:
                txt = page.extract_text() or ''
            except Exception:
                txt = ''
            # normalize whitespace
            txt = re.sub(r"\s+", " ", txt).strip()
            out.append({"page": i, "text": txt})
        return out
    except Exception:
        return []


def _preprocess_for_ocr(rgb_arr) -> Tuple[Any, Dict[str, Any]]:
    """Preprocess for drawings/scans: grayscale, deskew, binarize, suppress long lines.
    Returns an RGB array (H,W,3) plus debug stats.
    """
    info: Dict[str, Any] = {"deskew_deg": 0.0, "line_suppression": False}
    try:
        import cv2  # type: ignore
        import numpy as np

        img = rgb_arr
        if img is None:
            return rgb_arr, info

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)

        # deskew using foreground pixel min-area rect
        inv = 255 - thr
        coords = np.column_stack(np.where(inv > 0))
        if coords.shape[0] > 300:
            rect = cv2.minAreaRect(coords)
            angle = float(rect[-1])
            # convert OpenCV rect angle to a small rotation
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            # clamp to avoid crazy rotations
            if abs(angle) <= 12:
                (h, w) = thr.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                thr = cv2.warpAffine(thr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                info["deskew_deg"] = float(angle)

        # line suppression (engineering drawings often have heavy borders/grids)
        inv2 = 255 - thr
        h, w = inv2.shape[:2]
        kx = max(20, w // 30)
        ky = max(20, h // 30)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))
        horiz = cv2.morphologyEx(inv2, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
        vert = cv2.morphologyEx(inv2, cv2.MORPH_OPEN, vert_kernel, iterations=1)
        lines = cv2.bitwise_or(horiz, vert)
        if int(np.count_nonzero(lines)) > 0:
            inv2 = cv2.subtract(inv2, lines)
            info["line_suppression"] = True

        cleaned = 255 - inv2
        out_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        return out_rgb, info
    except Exception as e:
        info["error"] = str(e)
        return rgb_arr, info


def _drawing_likeness(rgb_arr) -> Dict[str, Any]:
    """Heuristic drawing-vs-photo score based on line/edge density and low color.
    Returns a dict with score in [0,1] and supporting stats.
    """
    out: Dict[str, Any] = {"score": 0.0}
    try:
        import cv2  # type: ignore
        import numpy as np

        if rgb_arr is None:
            return out
        img = rgb_arr
        if len(getattr(img, 'shape', [])) != 3:
            return out

        h, w = img.shape[:2]
        if h < 10 or w < 10:
            return out

        # Downscale for speed
        scale = 900.0 / float(max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray_blur, 80, 200)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size or 1)

        # Line density via Hough lines (drawing pages have many long straight lines)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=90, minLineLength=max(30, img.shape[1] // 12), maxLineGap=8)
        line_count = int(0 if lines is None else len(lines))
        line_density = float(line_count) / 200.0  # normalize rough scale

        # Low colorfulness heuristic (drawings often near B/W)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        mean_sat = float(np.mean(sat)) / 255.0
        low_color = 1.0 - min(1.0, mean_sat * 1.8)

        # B/W-ish ratio (near-white background)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)
        white_ratio = float(np.count_nonzero(bw > 200)) / float(bw.size or 1)

        # Combine: drawings have moderate edges, lots of lines, low color, and high white background
        score = 0.0
        score += min(1.0, edge_density * 8.0) * 0.30
        score += min(1.0, line_density) * 0.35
        score += min(1.0, low_color) * 0.20
        score += min(1.0, max(0.0, (white_ratio - 0.35) / 0.55)) * 0.15
        score = float(max(0.0, min(1.0, score)))

        out.update({
            "score": score,
            "edge_density": float(edge_density),
            "line_count": int(line_count),
            "mean_saturation": float(mean_sat),
            "white_ratio": float(white_ratio),
        })
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def _group_words_into_lines(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    if not words:
        return lines
    try:
        def _center_y(wd: Dict[str, Any]) -> float:
            bb = wd.get('bbox') if isinstance(wd, dict) else None
            if not (isinstance(bb, list) and len(bb) == 4):
                return 0.0
            return float(bb[1]) + float(bb[3]) * 0.5

        wds = [w for w in words if isinstance(w, dict) and str(w.get('text') or '').strip()]
        wds.sort(key=lambda w: (_center_y(w), float((w.get('bbox') or [0, 0, 0, 0])[0])))

        y_eps = 0.018
        cur: List[Dict[str, Any]] = []
        cur_y = None
        for w in wds:
            cy = _center_y(w)
            if cur_y is None:
                cur_y = cy
                cur = [w]
                continue
            if abs(cy - float(cur_y)) <= y_eps:
                cur.append(w)
                cur_y = (float(cur_y) * 0.8) + (cy * 0.2)
            else:
                cur.sort(key=lambda ww: float((ww.get('bbox') or [0, 0, 0, 0])[0]))
                txt = re.sub(r"\s+", " ", " ".join([str(ww.get('text') or '').strip() for ww in cur if str(ww.get('text') or '').strip()])).strip()
                lines.append({"y": float(cur_y), "text": txt, "words": cur})
                cur = [w]
                cur_y = cy
        if cur:
            cur.sort(key=lambda ww: float((ww.get('bbox') or [0, 0, 0, 0])[0]))
            txt = re.sub(r"\s+", " ", " ".join([str(ww.get('text') or '').strip() for ww in cur if str(ww.get('text') or '').strip()])).strip()
            lines.append({"y": float(cur_y or 0.0), "text": txt, "words": cur})
        return [ln for ln in lines if str(ln.get('text') or '').strip()]
    except Exception:
        return lines


def _extract_titleblock_fields(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    kv: List[Dict[str, str]] = []
    if not lines:
        return {"fields": fields, "kv": kv}

    joined = "\n".join([str(ln.get('text') or '') for ln in lines if isinstance(ln, dict)])

    def _pick(pats: List[str]) -> str:
        for pat in pats:
            m = re.search(pat, joined, flags=re.IGNORECASE)
            if m:
                v = str((m.group(1) or '')).strip().rstrip('.,;')
                if v:
                    return v
        return ''

    scale = _pick([
        r"\bscale\b\s*[:=]?\s*([0-9]+\s*[:/]\s*[0-9]+(?:\s*\(.*?\))?)",
        r"\bscale\b\s*[:=]?\s*(n\s*ts|nts|not\s*to\s*scale)",
        r"\bscale\b\s*[:=]?\s*([0-9]+\.?[0-9]*\s*(?:mm|cm|m|in|inch|\"))",
    ])
    if scale:
        fields["scale"] = scale

    sheet = _pick([
        r"\bsheet\b\s*[:#=]?\s*([0-9]+\s*(?:of\s*[0-9]+)?)",
    ])
    if sheet:
        fields["sheet"] = sheet

    rev = _pick([
        r"\brev(?:ision)?\b\s*[:#=]?\s*([a-z0-9\-\.]{1,10})",
    ])
    if rev:
        fields["revision"] = rev

    mat = _pick([
        r"\bmaterial\b\s*[:=]?\s*([a-z0-9\-\.,\s]{3,60})",
        r"\bmatl\b\s*[:=]?\s*([a-z0-9\-\.,\s]{3,60})",
    ])
    if mat:
        fields["material"] = re.sub(r"\s+", " ", mat).strip()

    finish = _pick([
        r"\bfinish\b\s*[:=]?\s*([a-z0-9\-\.,\s]{3,60})",
        r"\bsurface\s*finish\b\s*[:=]?\s*([a-z0-9\-\.,\s]{3,60})",
    ])
    if finish:
        fields["finish"] = re.sub(r"\s+", " ", finish).strip()

    title = _pick([
        r"\btitle\b\s*[:=]?\s*([a-z0-9\-\.,\s]{3,80})",
        r"\bdescription\b\s*[:=]?\s*([a-z0-9\-\.,\s]{3,120})",
        r"\bpart\s*name\b\s*[:=]?\s*([a-z0-9\-\.,\s]{3,120})",
        r"\bitem\s*name\b\s*[:=]?\s*([a-z0-9\-\.,\s]{3,120})",
    ])
    if title:
        fields["title"] = re.sub(r"\s+", " ", title).strip()

    # Heuristic: many drawings don't label TITLE explicitly; infer a likely part title
    # from the title block lines by picking a long alpha-heavy line that isn't a key/value row.
    if not str(fields.get("title") or '').strip():
        try:
            bad_keys = [
                'scale', 'sheet', 'rev', 'revision', 'date', 'material', 'finish', 'dwg', 'drawing',
                'doc', 'document', 'drawn', 'checked', 'approved', 'projection', 'tolerance',
            ]
            best = ''
            best_score = 0.0
            for ln in lines[:28]:
                txt = str(ln.get('text') or '').strip()
                if not txt:
                    continue
                low = txt.lower()
                if ':' in txt or '=' in txt:
                    continue
                if any(k in low for k in bad_keys):
                    continue
                # reject rows that are mostly numeric/units
                letters = sum(1 for ch in txt if ch.isalpha())
                digits = sum(1 for ch in txt if ch.isdigit())
                if letters < 4 or letters < digits:
                    continue
                if len(txt) < 4 or len(txt) > 60:
                    continue
                score = float(letters) + (0.35 * float(len(txt)))
                # prefer ALLCAPS-like titles
                if txt.upper() == txt and letters >= 6:
                    score += 6.0
                if score > best_score:
                    best_score = score
                    best = txt
            if best:
                fields["title"] = re.sub(r"\s+", " ", best).strip()
        except Exception:
            pass

    drawing_no = _pick([
        r"\b(?:drawing\s*(?:no\.?|number)|dwg\s*(?:no\.?|number)|dwg\s*no)\b\s*[:#=]?\s*([a-z0-9\-\./]{3,40})",
        r"\bdoc(?:ument)?\s*ref(?:erence)?\b\s*[:#=]?\s*([a-z0-9\-\./]{3,40})",
    ])
    if drawing_no:
        fields["drawing_no"] = drawing_no

    date_s = _pick([
        r"\bdate\b\s*[:=]?\s*([0-9]{1,2}[\-/][0-9]{1,2}[\-/][0-9]{2,4})",
        r"\bdate\b\s*[:=]?\s*([0-9]{4}[\-/][0-9]{1,2}[\-/][0-9]{1,2})",
    ])
    if date_s:
        fields["date"] = date_s

    maker = _pick([
        r"\b(?:drawn\s*by|drawn)\b\s*[:=]?\s*([a-z0-9\-\.,\s]{2,50})",
    ])
    if maker:
        fields["drawn_by"] = re.sub(r"\s+", " ", maker).strip()

    checked = _pick([
        r"\b(?:checked\s*by|checked)\b\s*[:=]?\s*([a-z0-9\-\.,\s]{2,50})",
    ])
    if checked:
        fields["checked_by"] = re.sub(r"\s+", " ", checked).strip()

    approved = _pick([
        r"\b(?:approved\s*by|approved)\b\s*[:=]?\s*([a-z0-9\-\.,\s]{2,50})",
    ])
    if approved:
        fields["approved_by"] = re.sub(r"\s+", " ", approved).strip()

    if not fields:
        for ln in lines[:18]:
            txt = str(ln.get('text') or '').strip()
            if ':' in txt:
                a, b = txt.split(':', 1)
                ka = re.sub(r"\s+", " ", a.strip()).strip()
                vb = re.sub(r"\s+", " ", b.strip()).strip()
                if 1 <= len(ka) <= 22 and 1 <= len(vb) <= 80:
                    kv.append({"k": ka, "v": vb})

    return {"fields": fields, "kv": kv}


def _detect_border_from_image(rgb_arr) -> Dict[str, Any]:
    """Detect sheet border as largest closed contour. Returns bbox in normalized coords and confidence.
    Hard gate happens outside this function.
    """
    out: Dict[str, Any] = {
        "confidence": 0.0,
        "bbox": None,
        "polyline": None,
        "warnings": [],
    }
    try:
        import cv2  # type: ignore
        import numpy as np

        if rgb_arr is None:
            out["warnings"].append("no_image")
            return out
        img = rgb_arr
        h, w = img.shape[:2]
        if h < 20 or w < 20:
            out["warnings"].append("too_small")
            return out

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            out["warnings"].append("no_contours")
            return out

        # pick largest contour by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        best = contours[0]
        area = float(cv2.contourArea(best))
        img_area = float(h * w)
        area_ratio = (area / img_area) if img_area > 0 else 0.0

        peri = float(cv2.arcLength(best, True))
        approx = cv2.approxPolyDP(best, 0.01 * peri, True)
        approx_pts = approx.reshape(-1, 2).astype(float).tolist() if approx is not None else []

        x, y, ww, hh = cv2.boundingRect(best)
        bbox = [float(x), float(y), float(x + ww), float(y + hh)]
        bbox_norm = [bbox[0] / float(w), bbox[1] / float(h), bbox[2] / float(w), bbox[3] / float(h)]
        bbox_norm_contour = list(bbox_norm)

        # confidence: big enough + near-rectangular + perimeter-to-bbox consistency
        rect_area = float(ww * hh)
        fill = (area / rect_area) if rect_area > 0 else 0.0
        # rectangularity prefers 4-6 vertices
        vcount = len(approx_pts)
        vscore = 1.0 if (4 <= vcount <= 6) else (0.7 if (3 <= vcount <= 8) else 0.4)
        aspr = float(ww) / float(hh or 1)
        # allow common sheet ratios; be forgiving but penalize extreme
        ascore = 1.0 if (0.55 <= aspr <= 2.0) else (0.6 if (0.35 <= aspr <= 3.0) else 0.2)
        ascore = float(max(0.0, min(1.0, ascore)))

        # area ratio should be large (border encloses most of page)
        a_score = float(max(0.0, min(1.0, (area_ratio - 0.30) / 0.55)))
        fill_score = float(max(0.0, min(1.0, (fill - 0.50) / 0.45)))

        conf = 0.0
        conf += a_score * 0.45
        conf += fill_score * 0.25
        conf += vscore * 0.20
        conf += ascore * 0.10
        conf = float(max(0.0, min(1.0, conf)))

        # Fallback: detect long horizontal/vertical line structure (common for drawing borders).
        # This helps when the largest contour is influenced by dense internal geometry.
        try:
            kx = int(max(25, min(w // 18, 160)))
            ky = int(max(25, min(h // 18, 160)))
            horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, kx), np.uint8), iterations=1)
            vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((ky, 1), np.uint8), iterations=1)
            lines_img = cv2.bitwise_or(horiz, vert)

            # Compute extents using direction-specific masks.
            yh, xh = np.where(horiz > 0)
            yv, xv = np.where(vert > 0)
            yu, xu = np.where(lines_img > 0)

            n_h = int(len(xh) if xh is not None else 0)
            n_v = int(len(xv) if xv is not None else 0)

            # Require sufficient line evidence.
            n_all = int(len(xu) if xu is not None else 0)
            if yu is not None and xu is not None and n_all > 200:
                # For cropped borders, horizontal mask may only find the top line.
                # Vertical border lines usually still span most of the page height.
                x_src = xv if (xv is not None and len(xv) > 120) else xu
                y_src = yv if (yv is not None and len(yv) > 120) else yu

                x_min = float(np.min(x_src))
                x_max = float(np.max(x_src))
                y_min = float(np.min(y_src))
                y_max = float(np.max(y_src))

                # If the line-based bbox collapses in one dimension (common when only one border side is detected),
                # borrow that dimension from the contour bbox.
                try:
                    span_x0 = (x_max - x_min) / float(w or 1)
                    span_y0 = (y_max - y_min) / float(h or 1)
                    if span_y0 < 0.45 and isinstance(bbox_norm_contour, list) and len(bbox_norm_contour) == 4:
                        y_min = float(bbox_norm_contour[1]) * float(h)
                        y_max = float(bbox_norm_contour[3]) * float(h)
                    if span_x0 < 0.45 and isinstance(bbox_norm_contour, list) and len(bbox_norm_contour) == 4:
                        x_min = float(bbox_norm_contour[0]) * float(w)
                        x_max = float(bbox_norm_contour[2]) * float(w)
                except Exception:
                    pass

                # If page OCR is weak (very few blocks), do targeted OCR on likely title and viewport crops.
                try:
                    page_blocks = [b for b in (perception_v1.get('text_blocks') or []) if isinstance(b, dict) and int(b.get('page') or 0) == int(page_idx)]
                    if len(page_blocks) < 25:
                        crops = [
                            [0.00, 0.72, 1.00, 0.28],  # bottom band (title block / footer)
                            [0.00, 0.00, 1.00, 0.72],  # main viewport band
                        ]
                        extra: List[Dict[str, Any]] = []
                        for c in crops:
                            extra.extend(_paddleocr_blocks_on_crop(arr, page_idx=int(page_idx), crop_norm=c, scale=2.0) or [])
                        # merge extra blocks (no dedupe; downstream compaction handles it)
                        if extra:
                            perception_v1["text_blocks"].extend(extra)
                except Exception:
                    pass

                # Margin proximity: border should be near the image edges.
                m = float(max(4.0, min(w, h) * 0.03))
                near_left = 1.0 if x_min <= m else max(0.0, 1.0 - (x_min - m) / (m * 3.0))
                near_right = 1.0 if (w - x_max) <= m else max(0.0, 1.0 - ((w - x_max) - m) / (m * 3.0))
                near_top = 1.0 if y_min <= m else max(0.0, 1.0 - (y_min - m) / (m * 3.0))
                near_bottom = 1.0 if (h - y_max) <= m else max(0.0, 1.0 - ((h - y_max) - m) / (m * 3.0))

                span_x = (x_max - x_min) / float(w or 1)
                span_y = (y_max - y_min) / float(h or 1)

                # If vertical border evidence exists but y-span is collapsed, treat as cropped/missed horizontal borders
                # and expand to the page bounds (this preserves strictness by requiring strong left/right evidence).
                try:
                    if span_y < 0.45 and n_v > 900 and float(near_left) >= 0.80 and float(near_right) >= 0.80 and span_x >= 0.85:
                        y_min = 0.0
                        y_max = float(h - 1)
                        near_top = 1.0
                        near_bottom = 1.0
                        span_y = 1.0
                except Exception:
                    pass

                # Initial scores.
                span_score = float(max(0.0, min(1.0, (min(span_x, span_y) - 0.65) / 0.30)))
                # Cropped pages may miss one side of the printed border.
                # Use the strongest 3 edges as the primary signal.
                edge_components = [float(near_left), float(near_right), float(near_top), float(near_bottom)]
                edge_components.sort(reverse=True)
                edge_score = float(max(0.0, min(1.0, sum(edge_components[:3]) / 3.0)))

                # If bbox is still collapsed vertically (e.g. only the top border line detected), but x-span is huge
                # and edge proximity is strong, expand vertically to full page.
                try:
                    if span_y < 0.20 and span_x >= 0.88 and edge_score >= 0.88 and n_all > 1200:
                        y_min = 0.0
                        y_max = float(h - 1)
                        near_top = 1.0
                        near_bottom = 1.0
                        span_y = 1.0
                        span_score = float(max(0.0, min(1.0, (min(span_x, span_y) - 0.65) / 0.30)))
                        edge_components = [float(near_left), float(near_right), float(near_top), float(near_bottom)]
                        edge_components.sort(reverse=True)
                        edge_score = float(max(0.0, min(1.0, sum(edge_components[:3]) / 3.0)))
                except Exception:
                    pass

                # If one side is missing but the other three are very strong, clamp bbox to the image edge.
                strong3 = bool(edge_components[2] >= 0.88)
                if strong3 and span_score >= 0.75:
                    if float(near_top) < 0.25:
                        y_min = 0.0
                    if float(near_left) < 0.25:
                        x_min = 0.0
                    if float(near_right) < 0.25:
                        x_max = float(w - 1)
                    if float(near_bottom) < 0.25:
                        y_max = float(h - 1)

                line_conf = float(max(0.0, min(1.0, (edge_score * 0.70) + (span_score * 0.30))))

                # Promote confidence in the common "cropped border" scenario: 3 strong sides + big span.
                try:
                    strong_edges = int(sum([1 for v in [near_left, near_right, near_top, near_bottom] if float(v) >= 0.78]))
                    if strong_edges >= 3 and span_score >= 0.70 and edge_score >= 0.75:
                        line_conf = float(max(line_conf, 0.92))
                    elif strong_edges >= 2 and span_score >= 0.88 and edge_score >= 0.80:
                        line_conf = float(max(line_conf, 0.90))
                except Exception:
                    pass

                if line_conf > conf:
                    conf = line_conf
                    bbox_norm = [x_min / float(w), y_min / float(h), x_max / float(w), y_max / float(h)]
                    out["warnings"].append("border_detected_by_lines")
                    out["stats"] = {
                        **(out.get("stats") or {}),
                        "line_edge_score": edge_score,
                        "line_span_score": span_score,
                        "line_pixels_all": int(n_all),
                        "line_pixels_h": int(n_h),
                        "line_pixels_v": int(n_v),
                    }
        except Exception as _e:
            pass

        # Fallback 2: Hough line frame detection (more robust when morphology misses vertical borders).
        try:
            edges = cv2.Canny(gray, 60, 160)
            min_len_v = int(max(60.0, float(h) * 0.35))
            min_len_h = int(max(60.0, float(w) * 0.35))
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=120, minLineLength=int(min(min_len_v, min_len_h)), maxLineGap=int(max(6.0, min(w, h) * 0.02)))

            if lines is not None and len(lines) > 0:
                v_xs: List[float] = []
                h_ys: List[float] = []
                for l in lines:
                    x1, y1, x2, y2 = [float(v) for v in l[0].tolist()]
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    length = float((dx * dx + dy * dy) ** 0.5)
                    # near-vertical
                    if dy > 0 and (dx / (dy + 1e-6)) < 0.15 and length >= float(min_len_v):
                        v_xs.append((x1 + x2) * 0.5)
                        continue
                    # near-horizontal
                    if dx > 0 and (dy / (dx + 1e-6)) < 0.15 and length >= float(min_len_h):
                        h_ys.append((y1 + y2) * 0.5)

                if len(v_xs) >= 2:
                    x_min = float(max(0.0, min(v_xs)))
                    x_max = float(min(float(w - 1), max(v_xs)))
                    y_min = float(max(0.0, min(h_ys))) if len(h_ys) >= 1 else 0.0
                    y_max = float(min(float(h - 1), max(h_ys))) if len(h_ys) >= 2 else float(h - 1)

                    # Score: strong left/right + large x-span. Allow cropped/missing top line.
                    m = float(max(4.0, min(w, h) * 0.03))
                    near_left = 1.0 if x_min <= m else max(0.0, 1.0 - (x_min - m) / (m * 3.0))
                    near_right = 1.0 if (w - x_max) <= m else max(0.0, 1.0 - ((w - x_max) - m) / (m * 3.0))
                    near_top = 1.0 if y_min <= m else max(0.0, 1.0 - (y_min - m) / (m * 3.0))
                    near_bottom = 1.0 if (h - y_max) <= m else max(0.0, 1.0 - ((h - y_max) - m) / (m * 3.0))

                    span_x = (x_max - x_min) / float(w or 1)
                    span_y = (y_max - y_min) / float(h or 1)

                    # If top border is missing but sides are strong, clamp to the image top.
                    if near_top < 0.25 and near_left >= 0.80 and near_right >= 0.80 and span_x >= 0.85:
                        y_min = 0.0
                        near_top = 1.0
                        span_y = (y_max - y_min) / float(h or 1)

                    edge_components = [float(near_left), float(near_right), float(near_top), float(near_bottom)]
                    edge_components.sort(reverse=True)
                    edge_score = float(max(0.0, min(1.0, sum(edge_components[:3]) / 3.0)))
                    span_score = float(max(0.0, min(1.0, (min(span_x, span_y) - 0.65) / 0.30)))

                    line_conf = float(max(0.0, min(1.0, (edge_score * 0.70) + (span_score * 0.30))))
                    # Promote when we have very strong left/right edges and span.
                    if near_left >= 0.85 and near_right >= 0.85 and span_x >= 0.90 and span_score >= 0.70:
                        line_conf = float(max(line_conf, 0.92))

                    if line_conf > conf:
                        conf = float(line_conf)
                        bbox_norm = [x_min / float(w), y_min / float(h), x_max / float(w), y_max / float(h)]
                        out["warnings"].append("border_detected_by_hough")
                        out["stats"] = {
                            **(out.get("stats") or {}),
                            "hough_v_lines": int(len(v_xs)),
                            "hough_h_lines": int(len(h_ys)),
                            "hough_edge_score": float(edge_score),
                            "hough_span_score": float(span_score),
                        }
        except Exception:
            pass

        stats_out = {
            "area_ratio": area_ratio,
            "fill": fill,
            "vertices": vcount,
            "aspect_ratio": aspr,
        }
        try:
            if isinstance(out.get("stats"), dict):
                stats_out = {**stats_out, **(out.get("stats") or {})}
        except Exception:
            pass

        out.update({
            "confidence": conf,
            "bbox": bbox_norm,
            "polyline": approx_pts[:200],
            "stats": stats_out,
        })
        return out
    except Exception as e:
        out["warnings"].append(str(e))
        return out


def _edge_density_grid(rgb_arr, grid_x: int = 6, grid_y: int = 4) -> List[List[float]]:
    try:
        import cv2  # type: ignore
        import numpy as np

        img = rgb_arr
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray_blur, 80, 200)

        out: List[List[float]] = []
        for gy in range(grid_y):
            row: List[float] = []
            y0 = int((gy / grid_y) * h)
            y1 = int(((gy + 1) / grid_y) * h)
            for gx in range(grid_x):
                x0 = int((gx / grid_x) * w)
                x1 = int(((gx + 1) / grid_x) * w)
                tile = edges[y0:y1, x0:x1]
                den = float(np.count_nonzero(tile)) / float(tile.size or 1)
                row.append(den)
            out.append(row)
        return out
    except Exception:
        return [[0.0 for _ in range(grid_x)] for _ in range(grid_y)]


def _build_engineering_drawing_canonical(
    engineering_dom: Dict[str, Any],
    extracted_fields: Dict[str, Any],
    text_full: str,
    is_pdf: bool,
) -> Dict[str, Any]:
    # Canonical object (Option A) based on the user's mapping.
    # This builder is deterministic and only uses structured + OCR/PDF text signals.
    dom = engineering_dom if isinstance(engineering_dom, dict) else {}
    sheet = dom.get('sheet') if isinstance(dom.get('sheet'), dict) else {}
    border = (sheet.get('border') if isinstance(sheet, dict) else None) or {}
    border_conf = float(border.get('confidence_score') or 0.0)

    regions = dom.get('regions') if isinstance(dom.get('regions'), list) else []
    views = dom.get('views') if isinstance(dom.get('views'), list) else []
    annotations_dom = dom.get('annotations') if isinstance(dom.get('annotations'), dict) else {}
    metadata_dom = dom.get('metadata') if isinstance(dom.get('metadata'), dict) else {}

    # ---- Confidence per layer ----
    sheet_conf = float(max(0.0, min(1.0, border_conf)))

    # Regions confidence: title block confidence if present, else mild.
    title_conf = 0.0
    try:
        for r in regions:
            if isinstance(r, dict) and str(r.get('type') or '').upper() == 'TITLE_BLOCK':
                title_conf = float(r.get('confidence') or 0.0)
                break
    except Exception:
        title_conf = 0.0
    regions_conf = float(max(0.0, min(1.0, 0.3 + 0.7 * max(0.0, min(1.0, title_conf)))))

    # Views confidence: depends on number of clusters (non-zero) and border gate.
    views_count = len(views)
    views_conf = float(max(0.0, min(1.0, (0.2 if views_count else 0.0) + (0.6 if views_count >= 2 else 0.0) + (0.2 * sheet_conf))))

    # Geometry confidence (Phase 1): loop presence not yet reliable; use drawing-likeness and view clusters.
    geom_conf = float(max(0.0, min(1.0, (0.25 if views_count else 0.0) + (0.25 if sheet_conf >= 0.9 else 0.0))))

    # Annotation confidence: dimensions extracted + dim candidates.
    dims = extracted_fields.get('dimensions') if isinstance(extracted_fields, dict) else None
    dims_count = len(dims) if isinstance(dims, list) else 0
    dim_cands = annotations_dom.get('dimensions') if isinstance(annotations_dom, dict) else None
    dim_cand_count = len(dim_cands) if isinstance(dim_cands, list) else 0
    ann_conf = float(max(0.0, min(1.0, (0.15 if (dims_count or dim_cand_count) else 0.0) + (0.35 if dim_cand_count >= 5 else 0.0) + (0.25 if dims_count >= 2 else 0.0) + 0.25 * sheet_conf)))

    # Intent inference (rule-based, requires >=2 agreeing signals).
    t = str(text_full or '')
    t_l = t.lower()
    intent_signals: List[str] = []
    if 'vulcanize' in t_l or 'vulcanise' in t_l:
        intent_signals.append('vulcanize')
    if 'grain' in t_l:
        intent_signals.append('grain')
    if 'flammability' in t_l or 'fmvss' in t_l:
        intent_signals.append('flammability')
    # OEM/tier1 heuristic
    if 'toyota' in t_l or 'boshoku' in t_l:
        intent_signals.append('oem_tier1')

    inferred_intent: Dict[str, Any] = {
        "material_class": None,
        "process": None,
        "industry": None,
        "part_family": None,
        "use_case": None,
        "production_scale": None,
        "evidence": [],
        "confidence": 0.0,
    }

    if len(set(intent_signals)) >= 2:
        ev = [{"signal": s} for s in sorted(set(intent_signals))]
        inferred_intent["evidence"] = ev
        if 'vulcanize' in intent_signals:
            inferred_intent["material_class"] = "Rubber / Elastomer"
            inferred_intent["process"] = "Molded + Vulcanized"
        if 'flammability' in intent_signals:
            inferred_intent["industry"] = "Automotive Interior"
        if 'grain' in intent_signals:
            # cosmetic surface requirement
            pass
        if inferred_intent.get('process'):
            inferred_intent["production_scale"] = "Mass Production"
        inferred_intent["confidence"] = float(min(1.0, len(set(intent_signals)) / 4.0))
    else:
        inferred_intent["confidence"] = float(min(1.0, len(set(intent_signals)) / 4.0))

    intent_conf = float(max(0.0, min(1.0, float(inferred_intent.get('confidence') or 0.0))))

    # Global confidence (0.2 each) as specified.
    global_conf = float(max(0.0, min(1.0, 0.2 * sheet_conf + 0.2 * regions_conf + 0.2 * geom_conf + 0.2 * ann_conf + 0.2 * intent_conf)))

    # Metadata: prefer extracted title block fields if present.
    md_fields = metadata_dom.get('fields') if isinstance(metadata_dom.get('fields'), dict) else {}

    def _looks_like_garbage_text(s: str) -> bool:
        try:
            txt = str(s or '').strip()
            if not txt:
                return True
            if len(txt) < 3:
                return True
            if len(txt) > 90:
                return True
            # Very low alphabetic signal or extreme repetition.
            letters = sum(1 for ch in txt if ch.isalpha())
            digits = sum(1 for ch in txt if ch.isdigit())
            if letters < 4:
                return True
            if digits > 0 and letters < digits:
                return True

            tokens = [t for t in re.split(r"\s+", txt.upper()) if t]
            if not tokens:
                return True
            # Too many short tokens (typical OCR noise).
            short = sum(1 for t in tokens if len(t) <= 2)
            if short / max(1.0, float(len(tokens))) > 0.55:
                return True
            # Repetition: single token dominating.
            from collections import Counter

            c = Counter(tokens)
            top = c.most_common(1)[0][1]
            if top >= 4 and top / max(1.0, float(len(tokens))) > 0.45:
                return True
            return False
        except Exception:
            return True

    def _looks_like_plausible_part_name(s: str) -> bool:
        try:
            txt = str(s or '').strip()
            if _looks_like_garbage_text(txt):
                return False
            # Part names typically have at least one strong keyword.
            if re.search(r"\b(SEAT|CUSHION|PAD|COVER|TRIM|BRACKET|FRAME|HOUSING|PANEL|ASSY|ASSEMBLY)\b", txt, flags=re.IGNORECASE):
                return True
            # Otherwise accept only if it looks like a clean title (few tokens, alpha-heavy).
            tokens = [t for t in re.split(r"\s+", txt) if t]
            if len(tokens) > 12:
                return False
            letters = sum(1 for ch in txt if ch.isalpha())
            digits = sum(1 for ch in txt if ch.isdigit())
            return bool(letters >= 8 and letters >= digits)
        except Exception:
            return False

    def _looks_like_plausible_material(s: str) -> bool:
        try:
            txt = str(s or '').strip()
            if _looks_like_garbage_text(txt):
                return False
            return bool(
                re.search(
                    r"\b(RUBBER|EPDM|SILICONE|NBR|TPU|TPE|PP|PE|ABS|PC|PA\d{1,2}|POM|PVC|EVA|FOAM|LEATHER|FABRIC|STEEL|ALUMIN(?:UM|IUM))\b",
                    txt,
                    flags=re.IGNORECASE,
                )
            )
        except Exception:
            return False

    part_name_src = None
    part_name_conf = 0.0
    part_name_val = None
    try:
        v = md_fields.get('title')
        if isinstance(v, str) and v.strip() and _looks_like_plausible_part_name(v):
            part_name_val = v.strip()
            part_name_src = 'title_block'
            part_name_conf = 0.9
    except Exception:
        pass
    if not part_name_val:
        try:
            v = extracted_fields.get('title') if isinstance(extracted_fields, dict) else None
            if isinstance(v, str) and v.strip() and _looks_like_plausible_part_name(v):
                part_name_val = v.strip()
                part_name_src = 'title_block'
                part_name_conf = 0.7
        except Exception:
            pass
    if not part_name_val:
        if 'seat cushion' in t_l:
            part_name_val = 'Seat Cushion'
            part_name_src = 'ocr'
            part_name_conf = 0.6

    material_src = None
    material_conf = 0.0
    material_val = None
    try:
        v = md_fields.get('material')
        if isinstance(v, str) and v.strip() and _looks_like_plausible_material(v):
            material_val = v.strip()
            material_src = 'title_block'
            material_conf = 0.9
    except Exception:
        pass
    if not material_val and 'rubber' in t_l:
        material_val = 'Rubber'
        material_src = 'ocr'
        material_conf = 0.6
    metadata = {
        "part_name": part_name_val or None,
        "part_number": md_fields.get('drawing_no') or None,
        "reference_drawing": None,
        "company": None,
        "drawing_type": None,
        "size": None,
        "units": "mm" if ('mm' in t_l or 'millimeter' in t_l) else None,
        "standard_hint": None,
        "material": material_val or None,
        "evidence": {
            "part_name": {"source": part_name_src, "confidence": float(part_name_conf)},
            "material": {"source": material_src, "confidence": float(material_conf)},
        },
    }

    if part_name_val:
        p_l = str(part_name_val).lower()
        if 'seat' in p_l and 'cushion' in p_l:
            inferred_intent["part_family"] = "Seat Component"
            inferred_intent["use_case"] = "Seating / comfort component"
            inferred_intent["evidence"] = (inferred_intent.get('evidence') or []) + [
                {"signal": "part_name:seat_cushion", "source": part_name_src, "confidence": float(part_name_conf)}
            ]

    if material_val and not inferred_intent.get('material_class'):
        m_l = str(material_val).lower()
        if 'rubber' in m_l or 'elastomer' in m_l:
            inferred_intent["material_class"] = "Rubber / Elastomer"
            inferred_intent["evidence"] = (inferred_intent.get('evidence') or []) + [
                {"signal": "material:rubber", "source": material_src, "confidence": float(material_conf)}
            ]

    # Summary: fixed abstraction template.
    summary_obj: Dict[str, Any] = {
        "description": "",
        "confidence": global_conf,
        "uncertain": bool(global_conf < 0.75),
    }
    if global_conf < 0.75:
        summary_obj["description"] = "Uncertain: I could not reliably parse the drawing structure with high confidence. Please provide a higher-resolution export or ensure the full sheet border is visible."
    else:
        part_desc = metadata.get('part_name') or 'a part'
        proc_desc = inferred_intent.get('process') or 'a manufacturing process'
        ind_desc = inferred_intent.get('industry') or inferred_intent.get('use_case') or 'an application domain'
        summary_obj["description"] = (
            f"This drawing defines {part_desc}. The part appears intended for {proc_desc}. "
            f"Notes and title block hints suggest {ind_desc}. "
            f"The sheet contains structured regions (title block + viewports) and dimension/annotation content."
        )

    def _pick_evidence_lines(src_text: str, pats: List[str], limit: int = 6) -> List[str]:
        try:
            src = str(src_text or '').replace('\r', '\n')
            if not src.strip():
                return []
            lines_in = [ln.strip() for ln in src.split('\n') if ln.strip()]
            out: List[str] = []
            seen = set()
            for ln in lines_in:
                if len(ln) > 180:
                    continue
                ok = False
                for pat in pats:
                    if re.search(pat, ln, flags=re.IGNORECASE):
                        ok = True
                        break
                if not ok:
                    continue
                k = ln.lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(ln)
                if len(out) >= limit:
                    break
            return out
        except Exception:
            return []

    euo_conf = {
        "global": float(global_conf),
        "sheet": float(sheet_conf),
        "regions": float(regions_conf),
        "views": float(views_conf),
        "geometry": float(geom_conf),
        "annotations": float(ann_conf),
        "intent": float(intent_conf),
    }

    euo_evidence: List[Dict[str, Any]] = []
    try:
        if metadata.get('part_name'):
            ev_lines = _pick_evidence_lines(text_full, [r"\bseat\s*cushion\b", r"\bpart\s*name\b", r"\bname\b", r"\bdescription\b"], limit=6)
            euo_evidence.append({
                "claim": "part_name",
                "value": metadata.get('part_name'),
                "source": (metadata.get('evidence') or {}).get('part_name', {}).get('source'),
                "confidence": float((metadata.get('evidence') or {}).get('part_name', {}).get('confidence') or 0.0),
                "snippets": ev_lines,
            })
    except Exception:
        pass
    try:
        if metadata.get('material'):
            ev_lines = _pick_evidence_lines(text_full, [r"\brubber\b", r"\bmaterial\b", r"\bepdm\b", r"\bnbr\b", r"\btpu\b"], limit=6)
            euo_evidence.append({
                "claim": "material",
                "value": metadata.get('material'),
                "source": (metadata.get('evidence') or {}).get('material', {}).get('source'),
                "confidence": float((metadata.get('evidence') or {}).get('material', {}).get('confidence') or 0.0),
                "snippets": ev_lines,
            })
    except Exception:
        pass
    try:
        if isinstance(inferred_intent.get('evidence'), list) and inferred_intent.get('evidence'):
            sigs = [str(e.get('signal') or '') for e in inferred_intent.get('evidence') if isinstance(e, dict)]
            if any('flammability' in s for s in sigs):
                euo_evidence.append({
                    "claim": "constraint.flammability",
                    "value": "Flammability requirement present",
                    "source": "ocr",
                    "confidence": float(intent_conf),
                    "snippets": _pick_evidence_lines(text_full, [r"\bfmvss\b", r"\bflammability\b"], limit=6),
                })
            if any('grain' in s for s in sigs):
                euo_evidence.append({
                    "claim": "constraint.surface_grain",
                    "value": "Surface grain specification present",
                    "source": "ocr",
                    "confidence": float(intent_conf),
                    "snippets": _pick_evidence_lines(text_full, [r"\bgrain\b", r"\bgr\d{3,4}\b"], limit=6),
                })
    except Exception:
        pass

    constraints: List[Dict[str, Any]] = []
    try:
        if 'fmvss' in t_l or 'flammability' in t_l:
            constraints.append({"kind": "standard", "value": "FMVSS / flammability", "confidence": float(intent_conf)})
        if 'grain' in t_l:
            constraints.append({"kind": "surface", "value": "Grain / texture requirement", "confidence": float(intent_conf)})
        if 'vulcanize' in t_l or 'vulcanise' in t_l:
            constraints.append({"kind": "process", "value": "Vulcanization mentioned", "confidence": float(intent_conf)})
    except Exception:
        constraints = []

    uncertainties: List[Dict[str, Any]] = []
    try:
        if float(global_conf) < 0.8:
            uncertainties.append({"kind": "global", "reason": "Global understanding confidence is below 0.80", "confidence": float(global_conf)})
        if not metadata.get('part_number'):
            uncertainties.append({"kind": "metadata", "reason": "Part/drawing number not confidently extracted", "confidence": 0.0})
        if float(geom_conf) < 0.7:
            uncertainties.append({"kind": "geometry", "reason": "Geometry understanding is not implemented beyond view detection", "confidence": float(geom_conf)})
    except Exception:
        uncertainties = []

    def _to_enum(val: Any, allowed: List[str], default: str = "UNKNOWN") -> str:
        try:
            s = str(val or '').strip().upper()
            if s in allowed:
                return s
        except Exception:
            pass
        return default

    def _industry_enum(v: Any) -> str:
        s = str(v or '').strip().upper()
        allowed = ["AUTOMOTIVE", "AEROSPACE", "INDUSTRIAL", "CONSUMER", "MEDICAL", "UNKNOWN"]
        if s in allowed:
            return s
        if 'auto' in s:
            return "AUTOMOTIVE"
        if 'aero' in s:
            return "AEROSPACE"
        if 'medical' in s:
            return "MEDICAL"
        if 'consumer' in s:
            return "CONSUMER"
        if 'industrial' in s or 'industry' in s:
            return "INDUSTRIAL"
        return "UNKNOWN"

    def _standard_enum(v: Any) -> str:
        s = str(v or '').strip().upper()
        allowed = ["ISO", "JIS", "ASME", "MIXED", "UNKNOWN"]
        if s in allowed:
            return s
        if 'ANSI' in s or 'ASME' in s:
            return "ASME"
        return "UNKNOWN"

    def _material_class_enum(v: Any) -> str:
        s = str(v or '').strip().upper()
        allowed = ["METAL", "PLASTIC", "RUBBER", "COMPOSITE", "UNKNOWN"]
        if s in allowed:
            return s
        t = str(text_full or '').lower()
        if 'rubber' in t or 'epdm' in t or 'nbr' in t:
            return "RUBBER"
        if 'plastic' in t or 'pp' in t or 'abs' in t or 'tpu' in t:
            return "PLASTIC"
        if 'steel' in t or 'aluminum' in t or 'aluminium' in t:
            return "METAL"
        return "UNKNOWN"

    def _process_enum(v: Any) -> str:
        s = str(v or '').strip().upper()
        allowed = ["MACHINING", "INJECTION_MOLDING", "COMPRESSION_MOLDING", "CASTING", "ADDITIVE", "UNKNOWN"]
        if s in allowed:
            return s
        t = str(text_full or '').lower()
        if 'inject' in t and 'mold' in t:
            return "INJECTION_MOLDING"
        if 'compression' in t and 'mold' in t:
            return "COMPRESSION_MOLDING"
        if 'machine' in t or 'machining' in t or 'cnc' in t:
            return "MACHINING"
        if 'cast' in t or 'casting' in t:
            return "CASTING"
        if 'print' in t and ('3d' in t or 'additive' in t):
            return "ADDITIVE"
        return "UNKNOWN"

    def _scale_enum(v: Any) -> str:
        s = str(v or '').strip().upper()
        allowed = ["PROTOTYPE", "LOW_VOLUME", "MASS_PRODUCTION", "UNKNOWN"]
        if s in allowed:
            return s
        return "UNKNOWN"

    def _count_intent_signals(intent_obj: Dict[str, Any], key_pat: str) -> int:
        try:
            ev = intent_obj.get('evidence')
            if not isinstance(ev, list):
                return 0
            c = 0
            for e in ev:
                if not isinstance(e, dict):
                    continue
                sig = str(e.get('signal') or '').lower()
                if re.search(key_pat, sig):
                    c += 1
            return c
        except Exception:
            return 0

    # Enforce: manufacturing_intent fields should be supported by >=2 signals.
    mat_class = _material_class_enum(inferred_intent.get('material_class'))
    proc = _process_enum(inferred_intent.get('process'))
    prod_scale = _scale_enum(inferred_intent.get('production_scale'))

    mat_sig = _count_intent_signals(inferred_intent, r"material|rubber|plastic|metal")
    proc_sig = _count_intent_signals(inferred_intent, r"process|mold|machin|cast|additive|vulcan")
    scale_sig = _count_intent_signals(inferred_intent, r"volume|production|mass|prototype")

    if mat_sig < 2:
        mat_class = "UNKNOWN"
    if proc_sig < 2:
        proc = "UNKNOWN"
    if scale_sig < 2:
        prod_scale = "UNKNOWN"

    post_processes: List[str] = []
    try:
        if 'deburr' in t_l:
            post_processes.append('deburring')
        if 'texture' in t_l or 'grain' in t_l:
            post_processes.append('surface_texturing')
    except Exception:
        post_processes = []

    # Constraints buckets (schema)
    functional_constraints: List[str] = []
    cosmetic_constraints: List[str] = []
    regulatory_constraints: List[str] = []
    try:
        if 'grain' in t_l:
            cosmetic_constraints.append('surface_grain_uniformity')
        if 'fmvss' in t_l or 'flammability' in t_l:
            regulatory_constraints.append('flammability_limit')
    except Exception:
        pass

    # Standards referenced (schema)
    referenced_standards: List[str] = []
    try:
        for m in re.finditer(r"\b(?:BSDM|BSDL|FMVSS|ISO|JIS|ASME)\s*[-_]?\s*\d+[A-Z0-9-]*\b", str(text_full or ''), flags=re.IGNORECASE):
            st = re.sub(r"\s+", "", str(m.group(0) or '')).upper()
            if st and st not in referenced_standards:
                referenced_standards.append(st)
            if len(referenced_standards) >= 10:
                break
    except Exception:
        referenced_standards = []

    compliance_domain = "UNKNOWN"
    try:
        if 'fmvss' in t_l or 'flammability' in t_l:
            compliance_domain = "FLAMMABILITY"
    except Exception:
        compliance_domain = "UNKNOWN"

    # Risks (schema)
    quality_risks: List[Dict[str, Any]] = []
    try:
        if mat_class == 'RUBBER':
            quality_risks.append({
                "risk_type": "MATERIAL_DEGRADATION",
                "description": "Rubber parts can be sensitive to aging/ozone/temperature; material grade and test requirements matter.",
                "severity": "MEDIUM",
            })
        if 'grain' in t_l:
            quality_risks.append({
                "risk_type": "COSMETIC_DEFECT",
                "description": "Surface grain/texture requirements can drive cosmetic rejection if tooling or process varies.",
                "severity": "MEDIUM",
            })
        if float(intent_conf) < 0.75:
            quality_risks.append({
                "risk_type": "PROCESS_VARIABILITY",
                "description": "Manufacturing intent is not yet strongly grounded; process choice may be uncertain.",
                "severity": "LOW",
            })
    except Exception:
        quality_risks = []

    # Uncertainty model (schema)
    missing_info: List[str] = []
    ambiguous_features: List[str] = []
    try:
        if not metadata.get('material'):
            missing_info.append('exact material grade')
        if proc == 'UNKNOWN':
            missing_info.append('tooling type / manufacturing process')
        if float(geom_conf) < 0.7:
            ambiguous_features.append('key geometric features not reliably extracted')
    except Exception:
        missing_info = []
        ambiguous_features = []

    # Document classification (schema) - conservative: do not guess unless strong.
    doc_type = "UNKNOWN"
    try:
        if float(global_conf) >= 0.80 and int(dims_count) >= 2 and int(views_count) >= 1:
            doc_type = "PRODUCTION_DRAWING"
        elif float(global_conf) >= 0.80 and int(views_count) >= 1:
            doc_type = "CONCEPT_DRAWING"
    except Exception:
        doc_type = "UNKNOWN"

    # Geometry understanding (schema) - conservative placeholders.
    primary_form = "UNKNOWN"
    dimensional_complexity = "UNKNOWN"
    try:
        if int(dims_count) >= 10:
            dimensional_complexity = "HIGH"
        elif int(dims_count) >= 4:
            dimensional_complexity = "MEDIUM"
        elif int(dims_count) >= 1:
            dimensional_complexity = "LOW"
        if 'axisymmetric' in str(inferred_intent.get('part_family') or '').lower():
            primary_form = "AXISYMMETRIC"
    except Exception:
        pass

    # Annotation understanding (schema)
    tolerance_class = "UNKNOWN"
    try:
        if re.search(r"\btolerance\b|\btolerances\b", t_l):
            tolerance_class = "GENERAL"
    except Exception:
        tolerance_class = "UNKNOWN"

    surface_requirements: List[str] = []
    process_notes: List[str] = []
    try:
        if 'grain' in t_l:
            surface_requirements.append('grain_control')
        if 'deburr' in t_l:
            surface_requirements.append('no_burrs')
        if 'vulcanize' in t_l or 'vulcanise' in t_l:
            process_notes.append('vulcanize')
    except Exception:
        surface_requirements = []
        process_notes = []

    # Part identity (schema)
    part_name = metadata.get('part_name')
    internal_id = metadata.get('part_number')
    part_classification = "UNKNOWN"
    functional_role = "UNKNOWN"
    try:
        pn_l = str(part_name or '').lower()
        if 'assy' in pn_l or 'assembly' in pn_l:
            part_classification = "ASSEMBLY"
        elif part_name:
            part_classification = "SINGLE_PART"

        if 'cushion' in pn_l or 'damp' in pn_l or 'vibration' in pn_l:
            functional_role = "DAMPING"
        elif 'seal' in pn_l or 'gasket' in pn_l:
            functional_role = "SEALING"
        elif 'cover' in pn_l or 'trim' in pn_l:
            functional_role = "COSMETIC"
        elif part_name:
            functional_role = "UNKNOWN"
    except Exception:
        part_classification = "UNKNOWN"
        functional_role = "UNKNOWN"

    # Summary ready (schema)
    summary_ready = bool(float(global_conf) >= 0.75)
    if summary_ready and not (part_name or int(dims_count) >= 1 or int(views_count) >= 1):
        summary_ready = False

    engineering_understanding: Dict[str, Any] = {
        "object_id": str(uuid.uuid4()),
        "input_type": "PDF" if bool(is_pdf) else "IMAGE",
        "document_classification": {
            "document_type": doc_type,
            "industry_hint": _industry_enum(inferred_intent.get('industry')),
            "drawing_standard_hint": _standard_enum(metadata.get('standard_hint')),
        },
        "part_identity": {
            "name": part_name,
            "internal_id": internal_id,
            "classification": _to_enum(part_classification, ["SINGLE_PART", "ASSEMBLY", "SUB_COMPONENT", "UNKNOWN"], default="UNKNOWN"),
            "functional_role": _to_enum(functional_role, ["STRUCTURAL", "COSMETIC", "SEALING", "DAMPING", "SUPPORT", "UNKNOWN"], default="UNKNOWN"),
        },
        "geometry_understanding": {
            "primary_form": _to_enum(primary_form, ["AXISYMMETRIC", "PRISMATIC", "FREEFORM", "SHEET_LIKE", "UNKNOWN"], default="UNKNOWN"),
            "key_features": [],
            "dimensional_complexity": _to_enum(dimensional_complexity, ["LOW", "MEDIUM", "HIGH", "UNKNOWN"], default="UNKNOWN"),
            "symmetry": {"type": "UNKNOWN", "axis_count": None},
        },
        "annotation_understanding": {
            "dimensions_present": bool(int(dims_count) >= 1),
            "tolerance_class": _to_enum(tolerance_class, ["GENERAL", "TIGHT", "LOOSE", "UNKNOWN"], default="UNKNOWN"),
            "surface_requirements": surface_requirements,
            "process_notes": process_notes,
            "inspection_notes_present": bool(re.search(r"\binspect\b|\binspection\b", t_l) is not None),
        },
        "manufacturing_intent": {
            "material_class": _to_enum(mat_class, ["METAL", "PLASTIC", "RUBBER", "COMPOSITE", "UNKNOWN"], default="UNKNOWN"),
            "likely_process": _to_enum(proc, ["MACHINING", "INJECTION_MOLDING", "COMPRESSION_MOLDING", "CASTING", "ADDITIVE", "UNKNOWN"], default="UNKNOWN"),
            "production_scale": _to_enum(prod_scale, ["PROTOTYPE", "LOW_VOLUME", "MASS_PRODUCTION", "UNKNOWN"], default="UNKNOWN"),
            "post_processes": post_processes,
        },
        "constraints": {
            "functional_constraints": functional_constraints,
            "cosmetic_constraints": cosmetic_constraints,
            "regulatory_constraints": regulatory_constraints,
        },
        "quality_risks": quality_risks,
        "standards_and_compliance": {
            "referenced_standards": referenced_standards,
            "compliance_domain": _to_enum(compliance_domain, ["FLAMMABILITY", "SAFETY", "DIMENSIONAL", "SURFACE", "UNKNOWN"], default="UNKNOWN"),
        },
        "uncertainty": {
            "missing_information": missing_info,
            "ambiguous_features": ambiguous_features,
        },
        "confidence": {
            "overall": float(global_conf),
            "breakdown": {
                "geometry": float(geom_conf),
                "annotations": float(ann_conf),
                "intent": float(intent_conf),
            },
        },
        "summary_ready": bool(summary_ready),
        # Keep evidence for audit/debugging (non-schema extension)
        "_evidence": euo_evidence,
        "_legacy": {
            "constraints_flat": constraints,
            "uncertainties_flat": uncertainties,
            "confidence_layers": euo_conf,
            "geometry_counts": {
                "views_detected": int(views_count),
                "regions_detected": int(len(regions) if isinstance(regions, list) else 0),
                "dimension_candidates": int(dim_cand_count),
                "dimensions_extracted": int(dims_count),
            },
        },
    }

    return {
        "sheet": {
            "size": sheet.get('size') if isinstance(sheet, dict) else None,
            "units": metadata.get('units') or None,
            "standard_hint": metadata.get('standard_hint') or None,
            "border": {
                "confidence": sheet_conf,
                "bbox": border.get('bbox'),
            },
        },
        "regions": regions,
        "views": views,
        "geometry": {
            "by_view": {},
            "confidence": geom_conf,
        },
        "annotations": {
            "dimensions": dim_cands if isinstance(dim_cands, list) else [],
            "notes": annotations_dom.get('notes') if isinstance(annotations_dom.get('notes'), list) else [],
            "symbols": annotations_dom.get('symbols') if isinstance(annotations_dom.get('symbols'), list) else [],
            "confidence": ann_conf,
            "counts": {
                "dimension_candidates": int(dim_cand_count),
                "dimensions_extracted": int(dims_count),
            },
        },
        "metadata": metadata,
        "inferred_intent": inferred_intent,
        "engineering_understanding": engineering_understanding,
        "summary": summary_obj,
        "confidence": {
            "sheet": sheet_conf,
            "regions": regions_conf,
            "views": views_conf,
            "geometry": geom_conf,
            "annotations": ann_conf,
            "intent": intent_conf,
            "global": global_conf,
        },
        "diagnostics": dom.get('diagnostics') if isinstance(dom.get('diagnostics'), dict) else {},
        "is_pdf": bool(is_pdf),
    }


def _locate_title_block_anywhere(
    words: List[Dict[str, Any]],
    border_bbox: List[float],
    rgb_arr,
    grid_x: int = 6,
    grid_y: int = 4,
) -> Dict[str, Any]:
    """Choose title block region anywhere: high text density + low edge density near border interior."""
    out: Dict[str, Any] = {"bbox": None, "confidence": 0.0, "method": "grid"}
    if not (isinstance(border_bbox, list) and len(border_bbox) == 4):
        return out
    if not words:
        return out

    bx1, by1, bx2, by2 = [float(x) for x in border_bbox]
    bw = max(1e-6, bx2 - bx1)
    bh = max(1e-6, by2 - by1)

    # Count words per tile inside border.
    counts = [[0 for _ in range(grid_x)] for _ in range(grid_y)]
    for wd in words:
        bb = wd.get('bbox') if isinstance(wd, dict) else None
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        x, y, w, h = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        cx = x + w * 0.5
        cy = y + h * 0.5
        if cx < bx1 or cx > bx2 or cy < by1 or cy > by2:
            continue
        # normalize to border local coords
        lx = (cx - bx1) / bw
        ly = (cy - by1) / bh
        gx = int(min(grid_x - 1, max(0, int(lx * grid_x))))
        gy = int(min(grid_y - 1, max(0, int(ly * grid_y))))
        counts[gy][gx] += 1

    # Edge density grid of whole page and then sample corresponding tiles.
    ed = _edge_density_grid(rgb_arr, grid_x=grid_x, grid_y=grid_y)

    # Score tiles: want text density high AND edge density low.
    max_c = max([c for row in counts for c in row] or [0])
    if max_c <= 0:
        return out

    best = None
    best_score = -1e9
    for gy in range(grid_y):
        for gx in range(grid_x):
            c = counts[gy][gx]
            if c <= 0:
                continue
            text_s = float(c) / float(max_c or 1)
            edge_s = float(ed[gy][gx])
            # prefer low edges, but allow some gridlines
            edge_good = 1.0 - min(1.0, edge_s * 7.0)

            # title block often near edges: boost tiles near border sides
            near_edge = 0.0
            if gx == 0 or gx == grid_x - 1:
                near_edge += 0.25
            if gy == 0 or gy == grid_y - 1:
                near_edge += 0.25
            score = (text_s * 0.65) + (edge_good * 0.25) + near_edge

            if score > best_score:
                best_score = score
                best = (gx, gy, c, edge_s)

    if best is None:
        return out

    gx, gy, c, edge_s = best
    # convert tile to normalized bbox in global coords
    x1 = bx1 + (gx / grid_x) * bw
    x2 = bx1 + ((gx + 1) / grid_x) * bw
    y1 = by1 + (gy / grid_y) * bh
    y2 = by1 + ((gy + 1) / grid_y) * bh

    conf = float(max(0.0, min(1.0, best_score / 1.5)))
    out.update({
        "bbox": [x1, y1, x2, y2],
        "confidence": conf,
        "stats": {"tile": [gx, gy], "word_count": int(c), "edge_density": float(edge_s)},
    })
    return out


def _cluster_view_segments(
    segments: List[List[float]],
    img_w: int,
    img_h: int,
    dist_px: float,
) -> List[Dict[str, Any]]:
    if not segments:
        return []

    parent = list(range(len(segments)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def seg_points(seg: List[float]) -> List[List[float]]:
        # Endpoints only. Midpoints tend to create transitive "bridges" via leaders/centerlines.
        x1, y1, x2, y2 = seg
        return [[x1, y1], [x2, y2]]

    pts = [seg_points(s) for s in segments]
    d2 = float(dist_px) * float(dist_px)

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            close = False
            for p in pts[i]:
                for q in pts[j]:
                    dx = float(p[0]) - float(q[0])
                    dy = float(p[1]) - float(q[1])
                    if (dx * dx + dy * dy) <= d2:
                        close = True
                        break
                if close:
                    break
            if close:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(len(segments)):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    out: List[Dict[str, Any]] = []
    for _, idxs in clusters.items():
        xs: List[float] = []
        ys: List[float] = []
        total_len = 0.0
        for k in idxs:
            x1, y1, x2, y2 = segments[k]
            xs.extend([float(x1), float(x2)])
            ys.extend([float(y1), float(y2)])
            dx = float(x1) - float(x2)
            dy = float(y1) - float(y2)
            total_len += float((dx * dx + dy * dy) ** 0.5)
        if not xs or not ys:
            continue
        x1 = float(max(0.0, min(float(img_w), min(xs))))
        y1 = float(max(0.0, min(float(img_h), min(ys))))
        x2 = float(max(0.0, min(float(img_w), max(xs))))
        y2 = float(max(0.0, min(float(img_h), max(ys))))
        area = float(max(1.0, (x2 - x1) * (y2 - y1)))
        density = float(total_len / (area ** 0.5))
        out.append({
            "bbox": [x1, y1, x2, y2],
            "segment_count": int(len(idxs)),
            "total_length_px": float(total_len),
            "density": float(density),
        })

    def _bbox_width(c: Dict[str, Any]) -> float:
        bb = c.get('bbox')
        if not (isinstance(bb, list) and len(bb) == 4):
            return 0.0
        return float(bb[2]) - float(bb[0])

    out.sort(key=_bbox_width, reverse=True)
    return out


def _detect_view_clusters_in_viewport(
    rgb_arr,
    viewport_bbox: List[float],
    title_bbox,
) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception:
        return {"views": [], "segments": 0, "status": "cv2_unavailable"}

    if not (isinstance(viewport_bbox, list) and len(viewport_bbox) == 4):
        return {"views": [], "segments": 0, "status": "no_viewport_bbox"}

    h, w = rgb_arr.shape[:2]

    bb = [float(v) for v in viewport_bbox]
    if max(bb) <= 1.5:
        bb = [bb[0] * float(w), bb[1] * float(h), bb[2] * float(w), bb[3] * float(h)]

    vx1 = int(max(0, min(int(bb[0]), w)))
    vy1 = int(max(0, min(int(bb[1]), h)))
    vx2 = int(max(0, min(int(bb[2]), w)))
    vy2 = int(max(0, min(int(bb[3]), h)))
    if vx2 <= vx1 or vy2 <= vy1:
        return {"views": [], "segments": 0, "status": "invalid_viewport_bbox"}

    roi = rgb_arr[vy1:vy2, vx1:vx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    min_dim = float(min(vx2 - vx1, vy2 - vy1))
    # Slightly longer minimum keeps dimension leaders from dominating clustering.
    min_len = int(max(18.0, min_dim * 0.055))
    max_gap = int(max(4.0, min_dim * 0.015))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=min_len, maxLineGap=max_gap)

    segments: List[List[float]] = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = [float(v) for v in l[0].tolist()]
            x1 += float(vx1)
            x2 += float(vx1)
            y1 += float(vy1)
            y2 += float(vy1)

            if isinstance(title_bbox, list) and len(title_bbox) == 4:
                tx1, ty1, tx2, ty2 = [float(v) for v in title_bbox]
                mx = (x1 + x2) * 0.5
                my = (y1 + y2) * 0.5
                if tx1 <= mx <= tx2 and ty1 <= my <= ty2:
                    continue

            # Drop very short segments (often leaders/connector noise).
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            seg_len = float((dx * dx + dy * dy) ** 0.5)
            if seg_len < float(max(20.0, min_dim * 0.03)):
                continue
            segments.append([x1, y1, x2, y2])

    # Tighter clustering distance to prevent bridging across separate view islands.
    dist_px = float(max(12.0, min_dim * 0.035))
    clusters = _cluster_view_segments(segments, img_w=w, img_h=h, dist_px=dist_px)

    # Expose raw clusters for downstream regioning/view typing.
    raw_clusters: List[Dict[str, Any]] = []
    for c in clusters:
        bb = c.get('bbox')
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in bb]
        raw_clusters.append({
            "bbox": [x1, y1, x2, y2],
            "segment_count": int(c.get('segment_count') or 0),
            "total_length_px": float(c.get('total_length_px') or 0.0),
            "density": float(c.get('density') or 0.0),
        })

    # Filter tiny clusters for "views" list (stricter). Keep raw_clusters always.
    views: List[Dict[str, Any]] = []
    for i, c in enumerate(clusters, start=1):
        bb = c.get('bbox')
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in bb]
        area = float(max(0.0, (x2 - x1) * (y2 - y1)))
        if area < float((min_dim * min_dim) * 0.004):
            continue
        views.append({
            "id": f"v{i}",
            "region_id": "r_view",
            "view_type": "UNKNOWN",
            "bounding_box": [x1, y1, x2, y2],
            "confidence": float(max(0.2, min(0.9, 0.25 + 0.03 * float(c.get('segment_count') or 0)))),
            "stats": {
                "segment_count": int(c.get('segment_count') or 0),
                "total_length_px": float(c.get('total_length_px') or 0.0),
                "density": float(c.get('density') or 0.0),
            },
            "geometry": {"nodes": [], "edges": [], "faces_2d": [], "open_profiles": []},
            "local_coordinate_frame": None,
            "inferred_alignment": [],
        })

    return {"views": views, "clusters": raw_clusters, "segments": int(len(segments)), "status": "ok"}


def _extract_dimension_candidates_from_words(
    words: List[Dict[str, Any]],
    viewport_bbox: List[float],
    title_bbox,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not (isinstance(viewport_bbox, list) and len(viewport_bbox) == 4):
        return out

    vx1, vy1, vx2, vy2 = [float(v) for v in viewport_bbox]

    tx1 = ty1 = tx2 = ty2 = None
    if isinstance(title_bbox, list) and len(title_bbox) == 4:
        tx1, ty1, tx2, ty2 = [float(v) for v in title_bbox]

    for wd in words:
        if not isinstance(wd, dict):
            continue
        txt = str(wd.get('text') or '').strip()
        if not txt:
            continue
        bb = wd.get('bbox')
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        x, y, ww, hh = [float(v) for v in bb]
        cx = x + ww * 0.5
        cy = y + hh * 0.5
        if not (vx1 <= cx <= vx2 and vy1 <= cy <= vy2):
            continue
        if tx1 is not None and tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
            continue

        t = txt.replace(' ', '')
        looks_numeric = False
        if any(ch.isdigit() for ch in t) and len(t) <= 12:
            looks_numeric = True
        if not looks_numeric:
            continue

        out.append({
            "id": f"d{len(out) + 1}",
            "dimension_type": "UNKNOWN",
            "arrowheads": [],
            "text": txt,
            "reference_edges": [],
            "confidence": 0.35,
            "bbox": [x, y, x + ww, y + hh],
        })

    return out[:200]


def _extract_viewport_clusters(
    rgb_arr,
    border_bbox: List[float],
    title_bbox,
    words: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception:
        return []

    if rgb_arr is None:
        return []
    if not (isinstance(border_bbox, list) and len(border_bbox) == 4):
        return []

    h, w = rgb_arr.shape[:2]

    # border_bbox is typically normalized (0..1). Convert to pixel coords for image ops.
    bb = [float(v) for v in border_bbox]
    if max(bb) <= 1.5:
        bb = [bb[0] * float(w), bb[1] * float(h), bb[2] * float(w), bb[3] * float(h)]

    bx1, by1, bx2, by2 = [int(max(0, min(v, lim))) for v, lim in zip(bb, [w, h, w, h])]
    if bx2 <= bx1 or by2 <= by1:
        return []

    # Work on a slightly inset interior to avoid the sheet frame (which often connects components).
    inset = int(max(2.0, min(float(min(bx2 - bx1, by2 - by1)) * 0.02, 28.0)))
    ix1 = int(min(bx2 - 1, bx1 + inset))
    iy1 = int(min(by2 - 1, by1 + inset))
    ix2 = int(max(ix1 + 2, bx2 - inset))
    iy2 = int(max(iy1 + 2, by2 - inset))

    roi = rgb_arr[iy1:iy2, ix1:ix2]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 60, 160)

    # Strengthen linework lightly; avoid merging distinct views.
    k = int(max(1, min(min(roi.shape[:2]) // 320, 3)))
    if k > 1:
        edges = cv2.dilate(edges, np.ones((k, k), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # Mask text boxes to prevent leader lines/text from bridging separate view islands.
    try:
        if isinstance(words, list) and words:
            for wd in words:
                if not isinstance(wd, dict):
                    continue
                bb0 = wd.get('bbox')
                if not (isinstance(bb0, list) and len(bb0) == 4):
                    continue
                x0, y0, ww0, hh0 = [float(v) for v in bb0]
                # OCR words are stored as normalized [x,y,w,h]
                if max([x0, y0, ww0, hh0]) <= 1.5:
                    x0 = x0 * float(w)
                    y0 = y0 * float(h)
                    ww0 = ww0 * float(w)
                    hh0 = hh0 * float(h)

                # Map to ROI coords
                rx1 = int(max(0, min(float(ix2 - ix1), (x0 - float(ix1)) - 2.0)))
                ry1 = int(max(0, min(float(iy2 - iy1), (y0 - float(iy1)) - 2.0)))
                rx2 = int(max(0, min(float(ix2 - ix1), (x0 + ww0 - float(ix1)) + 2.0)))
                ry2 = int(max(0, min(float(iy2 - iy1), (y0 + hh0 - float(iy1)) + 2.0)))
                if rx2 > rx1 and ry2 > ry1:
                    edges[ry1:ry2, rx1:rx2] = 0
    except Exception:
        pass

    # Break thin connectors (dimension leaders/centerlines) that join islands.
    try:
        edges = cv2.erode(edges, np.ones((2, 2), np.uint8), iterations=1)
    except Exception:
        pass

    # Mask out title block if present.
    if isinstance(title_bbox, list) and len(title_bbox) == 4:
        tb = [float(v) for v in title_bbox]
        if max(tb) <= 1.5:
            tb = [tb[0] * float(w), tb[1] * float(h), tb[2] * float(w), tb[3] * float(h)]
        tx1, ty1, tx2, ty2 = tb
        # title bbox is in full-image coords; map to roi coords
        rx1 = int(max(0, min(w, tx1) - ix1))
        ry1 = int(max(0, min(h, ty1) - iy1))
        rx2 = int(max(0, min(w, tx2) - ix1))
        ry2 = int(max(0, min(h, ty2) - iy1))
        if rx2 > rx1 and ry2 > ry1:
            edges[ry1:ry2, rx1:rx2] = 0

    # ---- Viewport clustering strategy ----
    # Connected components can merge distinct views due to leader/extension lines.
    # Instead, split by large whitespace gutters (low edge-density bands) recursively.
    ebin = (edges > 0).astype(np.uint8)

    debug: Dict[str, Any] = {"tiles": [], "cuts": []}

    def _find_best_gutter_cut(proj: "np.ndarray", min_run: int) -> int | None:
        # proj is 1D counts; find the longest run below a low threshold.
        if proj is None or proj.size < (min_run * 2):
            return None
        # More sensitive than before: drawings often have dense annotations.
        thr = float(max(1.0, float(proj.mean()) * 0.08))
        best_len = 0
        best_mid = None
        run = 0
        start = 0
        for i, v in enumerate(proj.tolist()):
            if float(v) <= thr:
                if run == 0:
                    start = i
                run += 1
            else:
                if run >= min_run and run > best_len:
                    best_len = run
                    best_mid = start + (run // 2)
                run = 0
        if run >= min_run and run > best_len:
            best_mid = start + (run // 2)
        # allow slightly shorter gutters when nothing else exists
        if best_mid is None and min_run >= 12:
            return _find_best_gutter_cut(proj, min_run=int(max(8, min_run * 0.6)))
        return int(best_mid) if best_mid is not None else None

    def _split_tiles(bin_img: "np.ndarray", x0: int, y0: int, x1: int, y1: int, depth: int) -> List[List[int]]:
        if depth <= 0:
            return [[x0, y0, x1, y1]]
        w0 = x1 - x0
        h0 = y1 - y0
        if w0 < 140 or h0 < 140:
            return [[x0, y0, x1, y1]]

        tile = bin_img[y0:y1, x0:x1]
        if tile.size <= 0:
            return [[x0, y0, x1, y1]]

        # If too sparse, keep as-is.
        nz = int(tile.sum())
        if nz < 250:
            return [[x0, y0, x1, y1]]

        # Prefer splitting along the longer dimension.
        min_run = int(max(12, min(w0, h0) * 0.04))
        if w0 >= h0:
            proj = tile.sum(axis=0)
            cut = _find_best_gutter_cut(proj, min_run=min_run)
            if cut is None:
                proj2 = tile.sum(axis=1)
                cut2 = _find_best_gutter_cut(proj2, min_run=min_run)
                if cut2 is None:
                    # fallback: split at midpoint if still dense (prevents one mega-viewport)
                    if nz > 6000:
                        cx = x0 + (w0 // 2)
                        return _split_tiles(bin_img, x0, y0, cx, y1, depth - 1) + _split_tiles(bin_img, cx, y0, x1, y1, depth - 1)
                    return [[x0, y0, x1, y1]]
                # horizontal split
                cy = y0 + int(cut2)
                return _split_tiles(bin_img, x0, y0, x1, cy, depth - 1) + _split_tiles(bin_img, x0, cy, x1, y1, depth - 1)
            cx = x0 + int(cut)
            return _split_tiles(bin_img, x0, y0, cx, y1, depth - 1) + _split_tiles(bin_img, cx, y0, x1, y1, depth - 1)
        else:
            proj = tile.sum(axis=1)
            cut = _find_best_gutter_cut(proj, min_run=min_run)
            if cut is None:
                proj2 = tile.sum(axis=0)
                cut2 = _find_best_gutter_cut(proj2, min_run=min_run)
                if cut2 is None:
                    if nz > 6000:
                        cy = y0 + (h0 // 2)
                        return _split_tiles(bin_img, x0, y0, x1, cy, depth - 1) + _split_tiles(bin_img, x0, cy, x1, y1, depth - 1)
                    return [[x0, y0, x1, y1]]
                cx = x0 + int(cut2)
                return _split_tiles(bin_img, x0, y0, cx, y1, depth - 1) + _split_tiles(bin_img, cx, y0, x1, y1, depth - 1)
            cy = y0 + int(cut)
            return _split_tiles(bin_img, x0, y0, x1, cy, depth - 1) + _split_tiles(bin_img, x0, cy, x1, y1, depth - 1)

    tiles = _split_tiles(ebin, 0, 0, int(ebin.shape[1]), int(ebin.shape[0]), depth=4)

    clusters: List[Dict[str, Any]] = []
    for x0, y0, x1, y1 in tiles:
        if x1 <= x0 or y1 <= y0:
            continue
        tile = ebin[y0:y1, x0:x1]
        if tile.size <= 0:
            continue
        nz = int(tile.sum())
        debug["tiles"].append({"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1), "nz": int(nz)})
        if nz < 700:
            continue
        # Tight bbox around actual edges inside the tile.
        ys, xs = np.where(tile > 0)
        if xs is None or ys is None or len(xs) < 200:
            continue
        tx0 = int(x0 + int(xs.min()))
        tx1 = int(x0 + int(xs.max()))
        ty0 = int(y0 + int(ys.min()))
        ty1 = int(y0 + int(ys.max()))
        ww = tx1 - tx0
        hh = ty1 - ty0
        if ww < 70 or hh < 70:
            continue

        fx1 = float(ix1 + tx0)
        fy1 = float(iy1 + ty0)
        fx2 = float(ix1 + tx1)
        fy2 = float(iy1 + ty1)
        clusters.append({"bbox": [fx1, fy1, fx2, fy2], "area": float(nz)})

    clusters.sort(key=lambda c: float(c.get('area') or 0.0), reverse=True)
    out = clusters[:8]
    try:
        if not out:
            debug["status"] = "no_clusters"
        else:
            debug["status"] = "ok"
            debug["clusters"] = [{"bbox": c.get('bbox'), "area": c.get('area')} for c in out[:6]]
    except Exception:
        pass

    # Attach debug to function for caller via attribute pattern.
    try:
        setattr(_extract_viewport_clusters, "_last_debug", debug)
    except Exception:
        pass
    return out


def _split_viewport_children_v2(
    rgb_arr,
    viewport_bbox: List[float],
    title_bbox,
    words: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception:
        return {"children": [], "status": "cv2_unavailable", "debug": {}}

    if rgb_arr is None:
        return {"children": [], "status": "no_image", "debug": {}}
    if not (isinstance(viewport_bbox, list) and len(viewport_bbox) == 4):
        return {"children": [], "status": "no_viewport_bbox", "debug": {}}

    h, w = rgb_arr.shape[:2]
    bb = [float(v) for v in viewport_bbox]
    if max(bb) <= 1.5:
        bb = [bb[0] * float(w), bb[1] * float(h), bb[2] * float(w), bb[3] * float(h)]

    vx1 = int(max(0, min(int(bb[0]), w)))
    vy1 = int(max(0, min(int(bb[1]), h)))
    vx2 = int(max(0, min(int(bb[2]), w)))
    vy2 = int(max(0, min(int(bb[3]), h)))
    if vx2 <= vx1 or vy2 <= vy1:
        return {"children": [], "status": "invalid_viewport_bbox", "debug": {}}

    inset = int(max(2.0, min(float(min(vx2 - vx1, vy2 - vy1)) * 0.02, 24.0)))
    ix1 = int(min(vx2 - 1, vx1 + inset))
    iy1 = int(min(vy2 - 1, vy1 + inset))
    ix2 = int(max(ix1 + 2, vx2 - inset))
    iy2 = int(max(iy1 + 2, vy2 - inset))
    if ix2 <= ix1 or iy2 <= iy1:
        ix1, iy1, ix2, iy2 = vx1, vy1, vx2, vy2

    roi = rgb_arr[iy1:iy2, ix1:ix2]
    if roi.size <= 0:
        return {"children": [], "status": "empty_roi", "debug": {}}

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 60, 160)

    k = int(max(1, min(min(roi.shape[:2]) // 320, 3)))
    if k > 1:
        edges = cv2.dilate(edges, np.ones((k, k), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    try:
        if isinstance(words, list) and words:
            for wd in words:
                if not isinstance(wd, dict):
                    continue
                bb0 = wd.get('bbox')
                if not (isinstance(bb0, list) and len(bb0) == 4):
                    continue
                x0, y0, ww0, hh0 = [float(v) for v in bb0]
                if max([x0, y0, ww0, hh0]) <= 1.5:
                    x0 = x0 * float(w)
                    y0 = y0 * float(h)
                    ww0 = ww0 * float(w)
                    hh0 = hh0 * float(h)
                cx = x0 + ww0 * 0.5
                cy = y0 + hh0 * 0.5
                if not (float(vx1) <= cx <= float(vx2) and float(vy1) <= cy <= float(vy2)):
                    continue
                rx1 = int(max(0, min(float(ix2 - ix1), (x0 - float(ix1)) - 2.0)))
                ry1 = int(max(0, min(float(iy2 - iy1), (y0 - float(iy1)) - 2.0)))
                rx2 = int(max(0, min(float(ix2 - ix1), (x0 + ww0 - float(ix1)) + 2.0)))
                ry2 = int(max(0, min(float(iy2 - iy1), (y0 + hh0 - float(iy1)) + 2.0)))
                if rx2 > rx1 and ry2 > ry1:
                    edges[ry1:ry2, rx1:rx2] = 0
    except Exception:
        pass

    try:
        edges = cv2.erode(edges, np.ones((2, 2), np.uint8), iterations=1)
    except Exception:
        pass

    if isinstance(title_bbox, list) and len(title_bbox) == 4:
        try:
            tb = [float(v) for v in title_bbox]
            if max(tb) <= 1.5:
                tb = [tb[0] * float(w), tb[1] * float(h), tb[2] * float(w), tb[3] * float(h)]
            tx1, ty1, tx2, ty2 = tb
            rx1 = int(max(0, min(float(ix2 - ix1), tx1 - float(ix1))))
            ry1 = int(max(0, min(float(iy2 - iy1), ty1 - float(iy1))))
            rx2 = int(max(0, min(float(ix2 - ix1), tx2 - float(ix1))))
            ry2 = int(max(0, min(float(iy2 - iy1), ty2 - float(iy1))))
            if rx2 > rx1 and ry2 > ry1:
                edges[ry1:ry2, rx1:rx2] = 0
        except Exception:
            pass

    ebin = (edges > 0).astype(np.uint8)
    debug: Dict[str, Any] = {"tiles": [], "status": None}

    def _find_best_gutter_cut(proj: "np.ndarray", min_run: int) -> int | None:
        if proj is None or proj.size < (min_run * 2):
            return None
        thr = float(max(1.0, float(proj.mean()) * 0.08))
        best_len = 0
        best_mid = None
        run = 0
        start = 0
        for i, v in enumerate(proj.tolist()):
            if float(v) <= thr:
                if run == 0:
                    start = i
                run += 1
            else:
                if run >= min_run and run > best_len:
                    best_len = run
                    best_mid = start + (run // 2)
                run = 0
        if run >= min_run and run > best_len:
            best_mid = start + (run // 2)
        if best_mid is None and min_run >= 12:
            return _find_best_gutter_cut(proj, min_run=int(max(8, min_run * 0.6)))
        return int(best_mid) if best_mid is not None else None

    def _split_tiles(bin_img: "np.ndarray", x0: int, y0: int, x1: int, y1: int, depth: int) -> List[List[int]]:
        if depth <= 0:
            return [[x0, y0, x1, y1]]
        w0 = x1 - x0
        h0 = y1 - y0
        if w0 < 120 or h0 < 120:
            return [[x0, y0, x1, y1]]
        tile = bin_img[y0:y1, x0:x1]
        if tile.size <= 0:
            return [[x0, y0, x1, y1]]
        nz = int(tile.sum())
        if nz < 220:
            return [[x0, y0, x1, y1]]
        min_run = int(max(10, min(w0, h0) * 0.05))
        if w0 >= h0:
            proj = tile.sum(axis=0)
            cut = _find_best_gutter_cut(proj, min_run=min_run)
            if cut is None:
                proj2 = tile.sum(axis=1)
                cut2 = _find_best_gutter_cut(proj2, min_run=min_run)
                if cut2 is None:
                    if nz > 5200:
                        cx = x0 + (w0 // 2)
                        return _split_tiles(bin_img, x0, y0, cx, y1, depth - 1) + _split_tiles(bin_img, cx, y0, x1, y1, depth - 1)
                    return [[x0, y0, x1, y1]]
                cy = y0 + int(cut2)
                return _split_tiles(bin_img, x0, y0, x1, cy, depth - 1) + _split_tiles(bin_img, x0, cy, x1, y1, depth - 1)
            cx = x0 + int(cut)
            return _split_tiles(bin_img, x0, y0, cx, y1, depth - 1) + _split_tiles(bin_img, cx, y0, x1, y1, depth - 1)
        proj = tile.sum(axis=1)
        cut = _find_best_gutter_cut(proj, min_run=min_run)
        if cut is None:
            proj2 = tile.sum(axis=0)
            cut2 = _find_best_gutter_cut(proj2, min_run=min_run)
            if cut2 is None:
                if nz > 5200:
                    cy = y0 + (h0 // 2)
                    return _split_tiles(bin_img, x0, y0, x1, cy, depth - 1) + _split_tiles(bin_img, x0, cy, x1, y1, depth - 1)
                return [[x0, y0, x1, y1]]
            cx = x0 + int(cut2)
            return _split_tiles(bin_img, x0, y0, cx, y1, depth - 1) + _split_tiles(bin_img, cx, y0, x1, y1, depth - 1)
        cy = y0 + int(cut)
        return _split_tiles(bin_img, x0, y0, x1, cy, depth - 1) + _split_tiles(bin_img, x0, cy, x1, y1, depth - 1)

    tiles = _split_tiles(ebin, 0, 0, int(ebin.shape[1]), int(ebin.shape[0]), depth=4)

    children: List[Dict[str, Any]] = []
    roi_area = float(max(1.0, float((ix2 - ix1) * (iy2 - iy1))))
    for x0, y0, x1, y1 in tiles:
        if x1 <= x0 or y1 <= y0:
            continue
        tile = ebin[y0:y1, x0:x1]
        if tile.size <= 0:
            continue
        nz = int(tile.sum())
        debug["tiles"].append({"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1), "nz": int(nz)})
        if nz < 650:
            continue
        ys, xs = np.where(tile > 0)
        if xs is None or ys is None or len(xs) < 180:
            continue
        tx0 = int(x0 + int(xs.min()))
        tx1 = int(x0 + int(xs.max()))
        ty0 = int(y0 + int(ys.min()))
        ty1 = int(y0 + int(ys.max()))
        ww = int(tx1 - tx0)
        hh = int(ty1 - ty0)
        if ww < 60 or hh < 60:
            continue
        fx1 = float(ix1 + tx0)
        fy1 = float(iy1 + ty0)
        fx2 = float(ix1 + tx1)
        fy2 = float(iy1 + ty1)
        child_area = float(max(1.0, (fx2 - fx1) * (fy2 - fy1)))
        rel = float(min(1.0, max(0.0, child_area / roi_area)))
        dens = float(nz / float(max(1, (x1 - x0) * (y1 - y0))))
        conf = float(max(0.35, min(0.85, 0.35 + 0.25 * rel + 0.25 * min(1.0, dens * 10.0))))
        children.append({"bbox": [fx1, fy1, fx2, fy2], "area": float(nz), "confidence": conf})

    children.sort(key=lambda c: float(c.get('area') or 0.0), reverse=True)
    children = children[:12]
    debug["status"] = "ok" if children else "no_children"
    return {"children": children, "status": str(debug.get('status') or 'ok'), "debug": debug}


def _clip_taxonomy_rank(model: Any, image_tensor: Any, labels: list[str], topk: int = 7) -> Dict[str, Any]:
    import clip  # type: ignore
    import torch

    cache_key = id(model)
    cached = _CLIP_TAX_CACHE.get(cache_key)
    if cached is None or cached.get("labels") != labels:
        prompts = [f"a photo of a {l}" for l in labels]
        tokens = clip.tokenize(prompts)
        with torch.no_grad():
            tf = model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        cached = {"labels": labels, "tf": tf}
        _CLIP_TAX_CACHE[cache_key] = cached

    tf = cached["tf"]
    with torch.no_grad():
        imf = model.encode_image(image_tensor)
        imf = imf / imf.norm(dim=-1, keepdim=True)
        sims = (imf @ tf.T).squeeze(0)
        probs = torch.softmax(sims, dim=0)

        k = int(min(topk, probs.shape[0]))
        vals, idxs = torch.topk(probs, k=k)
        candidates = [
            {"label": str(labels[int(i)]), "score": float(v.item())}
            for v, i in zip(vals, idxs)
        ]

        top1 = float(candidates[0]["score"]) if candidates else 0.0
        top2 = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        margin = top1 - top2
        entropy = float((-(probs * torch.log(probs + 1e-12)).sum()).item())
        n = float(probs.shape[0])
        entropy_norm = float(entropy / (torch.log(torch.tensor(n)).item() if n > 1 else 1.0))

    return {
        "top_label": candidates[0]["label"] if candidates else None,
        "top_score": top1,
        "candidates": candidates,
        "uncertainty": {
            "margin": margin,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
        },
        "source": "clip-taxonomy",
    }

# Base data dir: default to local ./data next to this app; override with DREAM_DATA_DIR (e.g., /app/data in Docker)
BASE_DATA = os.environ.get(
    "DREAM_DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
)
RESULT_DIR = os.path.join(BASE_DATA, "results")
IMAGE_DIR = os.path.join(BASE_DATA, "images")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


def process_upload_async(file_bytes: bytes, filename: str, models: ModelRegistry) -> str:
    job_id = str(uuid.uuid4())
    path = os.path.join(IMAGE_DIR, f"{job_id}_{filename}")
    with open(path, "wb") as f:
        f.write(file_bytes)
    t = Thread(target=_worker, args=(job_id, path, models), daemon=True)
    t.start()
    return job_id


def process_upload_sync(file_bytes: bytes, filename: str, models: ModelRegistry) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    path = os.path.join(IMAGE_DIR, f"{job_id}_{filename}")
    with open(path, "wb") as f:
        f.write(file_bytes)
    _worker(job_id, path, models)
    out_path = os.path.join(RESULT_DIR, f"{job_id}.json")
    with open(out_path, "r") as f:
        return json.load(f)


def _worker(job_id: str, file_path: str, models: ModelRegistry) -> str:
    timestamp = datetime.utcnow().isoformat()
    result: Dict[str, Any] = {
        "request_id": job_id,
        "source": os.path.basename(file_path),
        "timestamp": timestamp,
        "confidence_overall": None,
        "objects": [],
        "text": {},
        "sketch": {},
        "depth": {},
        "logs": []
    }

    # 0) PDF handling (rasterize pages; analyze first page for objects)
    is_pdf = _is_pdf(file_path)
    page_paths: List[str] = [file_path]
    pdf_text_pages: List[Dict[str, Any]] = []
    if is_pdf:
        page_paths = _render_pdf_to_images(file_path, max_pages=10, dpi=350)
        pdf_text_pages = _extract_pdf_text(file_path, max_pages=10)
        if not page_paths:
            # fallback to processing the PDF itself (some downstream components may still work)
            page_paths = [file_path]
    else:
        page_paths = [file_path]

    # 1) Detection (YOLOv8 tiny)
    yolo = models.get_yolo()
    if yolo is not None:
        try:
            from PIL import Image
            iw, ih = Image.open(primary_path).size
            preds = yolo(primary_path)
            r = preds[0]
            names = getattr(yolo.model, 'names', {}) or {}
            dets = []
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().numpy())
                cls = int(b.cls[0].cpu().numpy())
                label = names.get(cls, str(cls))
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                # clamp to image bounds (defensive)
                x1 = max(0.0, min(float(iw), x1))
                x2 = max(0.0, min(float(iw), x2))
                y1 = max(0.0, min(float(ih), y1))
                y2 = max(0.0, min(float(ih), y2))
                dets.append({"id": f"o{len(dets)+1}", "class": label, "bbox": [x1, y1, x2, y2], "confidence": conf})
            result["objects"] = dets
            result["logs"].append({"stage": "detection", "model": "yolov8n", "count": len(dets)})
        except Exception as e:
            result["logs"].append({"stage": "detection", "model": "yolov8n", "error": str(e)})
    else:
        result["logs"].append({"stage": "detection", "model": "yolov8n", "status": "unavailable"})

    # 2) Segmentation (SAM optional): if not configured, skip gracefully
    sam = models.get_sam()
    if sam is not None and result["objects"]:
        try:
            import numpy as np
            from PIL import Image
            image = Image.open(primary_path).convert("RGB")
            image_np = np.array(image)
            sam.set_image(image_np)
            masks_count = 0
            for o in result["objects"]:
                x1, y1, x2, y2 = map(int, o["bbox"])
                box = np.array([[x1, y1, x2, y2]])
                masks, scores, _ = sam.predict(box=box, multimask_output=True)
                # pick best mask
                idx = int(np.argmax(scores))
                mask = (masks[idx].astype('uint8') * 255)
                mask_path = f"{primary_path}.mask.{o['id']}.png"
                Image.fromarray(mask).save(mask_path)
                o["mask_url"] = mask_path
                # dominant color within mask
                o["color"] = [{"name": None, "hex": dominant_color_hex(image, mask), "confidence": 0.6}]
                masks_count += 1
            result["logs"].append({"stage": "segmentation", "model": "sam", "count": masks_count})
        except Exception as e:
            result["logs"].append({"stage": "segmentation", "model": "sam", "error": str(e)})
    else:
        result["logs"].append({"stage": "segmentation", "model": "sam", "status": "skipped"})

    # 3) Text extraction / OCR
    def _extract_dimensions(text_in: str):
        t = (text_in or '').replace('\u00d8', '⌀')
        t = re.sub(r"\s+", " ", t)
        dims = []
        patterns = [
            (r"(?P<prefix>⌀|Ø|DIA)\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|in|inch|\")?", "diameter"),
            (r"(?P<prefix>R)\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|in|inch|\")?", "radius"),
            (r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|in|inch|\")\b", "linear"),
            (r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>deg|°)", "angle"),
            (r"±\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|in|inch|\")?", "tolerance"),
        ]
        for pat, kind in patterns:
            for m in re.finditer(pat, t, flags=re.IGNORECASE):
                gd = m.groupdict()
                val = gd.get('val')
                unit = gd.get('unit')
                prefix = gd.get('prefix')
                if val:
                    dims.append({
                        "kind": kind,
                        "value": float(val),
                        "unit": (unit or '').lower() or None,
                        "prefix": prefix or None,
                        "raw": m.group(0),
                    })
        return dims

    def _extract_contacts(text_in: str) -> Dict[str, Any]:
        t = str(text_in or '')
        t_norm = re.sub(r"\s+", " ", t).strip()

        emails = sorted(set([m.group(0) for m in re.finditer(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", t_norm, flags=re.IGNORECASE)]))
        urls = []
        for m in re.finditer(r"\b(?:https?://)?(?:www\.)?[A-Z0-9.-]+\.[A-Z]{2,}(?:/[^\s]*)?\b", t_norm, flags=re.IGNORECASE):
            u = m.group(0).strip().rstrip('.,;')
            if '@' in u:
                continue
            if len(u) < 6:
                continue
            urls.append(u)
        urls = sorted(set(urls))

        phones = []
        for m in re.finditer(r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}", t_norm):
            p = re.sub(r"\s+", " ", m.group(0)).strip()
            digits = re.sub(r"\D", "", p)
            if len(digits) >= 9 and len(digits) <= 15:
                phones.append(p)
        phones = sorted(set(phones))

        address_like = []
        addr_keys = ["street", "st.", "st ", "road", "rd.", "rd ", "avenue", "ave", "sector", "block", "building", "floor", "suite", "city", "state", "zip", "pin", "pincode", "india", "usa"]
        if any(k in t_norm.lower() for k in addr_keys):
            address_like.append(t_norm)

        has_location_icon_hint = bool(address_like)
        has_phone_icon_hint = bool(phones)
        has_email_icon_hint = bool(emails)
        has_web_icon_hint = bool(urls)

        return {
            "emails": emails,
            "phones": phones,
            "urls": urls,
            "address_like": address_like,
            "icon_hints": {
                "location": has_location_icon_hint,
                "phone": has_phone_icon_hint,
                "email": has_email_icon_hint,
                "website": has_web_icon_hint,
            },
        }

    text_full = ''
    ocr_words = []
    extracted_fields: Dict[str, Any] = {}
    ocr_pages: List[Dict[str, Any]] = []

    drawing_stats_overall: Dict[str, Any] = {"score": 0.0}

    engineering_dom: Dict[str, Any] = {
        "sheet": None,
        "regions": [],
        "views": [],
        "annotations": {"dimensions": [], "symbols": [], "notes": []},
        "metadata": None,
        "diagnostics": {"warnings": [], "errors": [], "ambiguity_flags": []},
    }

    pdf_text_by_page: Dict[int, str] = {}
    if is_pdf and pdf_text_pages:
        for p in pdf_text_pages:
            try:
                pn = int(p.get('page') or 0)
            except Exception:
                pn = 0
            if pn > 0:
                pdf_text_by_page[pn] = str(p.get('text') or '')

    ocr_engine_ok = False
    abort_engineering_parse = False

    perception_v1: Dict[str, Any] = {
        "image_metadata": {"pages": {}},
        "text_blocks": [],
        "lines": [],
        "closed_shapes": [],
        "arrow_candidates": [],
        "dimension_candidates": [],
        "page_text": {},
        "regions": {"title_block": None, "viewports": [], "notes": []},
        "confidence": {},
        "validation": {"failure_reasons": [], "ocr_confidence_histogram": {}, "closed_loop_line_ratio": 0.0, "region_coverage_pct": 0.0},
    }
    if page_paths:
        try:
            import numpy as np
            from PIL import Image
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore
            except Exception:
                from rapidocr_onnxruntime.main import RapidOCR  # type: ignore

            ocr = RapidOCR()
            ocr_engine_ok = True
            merged_parts: List[str] = []

            # Border detection gatekeeper uses the first page image
            try:
                import cv2  # type: ignore
                arr0 = np.array(Image.open(primary_path).convert('RGB'))
                h0, w0 = arr0.shape[:2]
                border = _detect_border_from_image(arr0)
                # drawing-likeness early signal for gating
                try:
                    ds0 = _drawing_likeness(arr0)
                    if isinstance(ds0, dict):
                        drawing_stats_overall = ds0 if float(ds0.get('score') or 0.0) >= float(drawing_stats_overall.get('score') or 0.0) else drawing_stats_overall
                except Exception:
                    pass

                engineering_dom["sheet"] = {
                    "width_px": int(w0),
                    "height_px": int(h0),
                    "aspect_ratio": float(w0) / float(h0 or 1),
                    "border": {
                        "polyline_id": "border0" if border.get('bbox') else None,
                        "margin_mm": None,
                        "confidence_score": float(border.get('confidence') or 0.0),
                        "bbox": border.get('bbox'),
                    }
                }
                engineering_dom["border"] = border
                border_conf = float(border.get('confidence') or 0.0)
                # Strict rule applies when it looks like an engineering drawing (avoid breaking photos/text PDFs).
                if border_conf < 0.9 and float(drawing_stats_overall.get('score') or 0.0) >= 0.55:
                    abort_engineering_parse = True
                    engineering_dom["diagnostics"]["errors"].append("border_confidence_below_threshold")
                    engineering_dom["diagnostics"]["warnings"].append("parsing_aborted_due_to_border")
                    result["logs"].append({"stage": "border-gate", "status": "aborted", "border_confidence": border_conf, "drawing_likeness": drawing_stats_overall})
                else:
                    result["logs"].append({"stage": "border-gate", "status": "ok", "border_confidence": border_conf, "drawing_likeness": drawing_stats_overall})
            except Exception as e:
                engineering_dom["diagnostics"]["errors"].append(f"border_detection_failed: {str(e)}")

            if abort_engineering_parse:
                # Do not proceed with engineering layout parsing when border gate fails.
                # Still run OCR so non-engineering documents (tables/reports) are readable.
                engineering_dom["diagnostics"]["warnings"].append("engineering_layout_skipped_due_to_border_gate")

            for page_idx, pth in enumerate(page_paths, start=1):
                im = Image.open(pth).convert('RGB')
                w, h = im.size
                arr = np.array(im)
                arr2, pre_info = _preprocess_for_ocr(arr)

                try:
                    perception_v1["image_metadata"]["pages"][str(int(page_idx))] = {"width_px": int(w), "height_px": int(h), "dpi": 350 if is_pdf else None}
                except Exception:
                    pass

                try:
                    pocr = _perception_paddleocr(arr, page_idx)
                    if isinstance(pocr, dict) and isinstance(pocr.get('text_blocks'), list):
                        perception_v1["text_blocks"].extend(pocr.get('text_blocks') or [])
                        try:
                            hh = (pocr.get('_metrics') or {}).get('conf_hist')
                            if isinstance(hh, dict):
                                for k, v in hh.items():
                                    perception_v1["validation"]["ocr_confidence_histogram"][k] = int(perception_v1["validation"]["ocr_confidence_histogram"].get(k, 0)) + int(v)
                        except Exception:
                            pass
                except Exception:
                    pass

                try:
                    geom = _perception_opencv_geometry(arr, page_idx)
                    if isinstance(geom, dict):
                        perception_v1["lines"].extend(geom.get('lines') or [])
                        perception_v1["closed_shapes"].extend(geom.get('closed_shapes') or [])
                        perception_v1["arrow_candidates"].extend(geom.get('arrow_candidates') or [])
                except Exception:
                    pass

                try:
                    dcs = _perception_dimension_candidates(
                        page_idx=int(page_idx),
                        w_px=int(w),
                        h_px=int(h),
                        text_blocks=perception_v1.get('text_blocks') or [],
                        arrow_candidates=perception_v1.get('arrow_candidates') or [],
                    )
                    if dcs:
                        perception_v1["dimension_candidates"].extend(dcs)
                except Exception:
                    pass

                try:
                    regs = _perception_regions_from_signals(
                        page_idx=int(page_idx),
                        w_px=int(w),
                        h_px=int(h),
                        text_blocks=perception_v1.get('text_blocks') or [],
                        lines=perception_v1.get('lines') or [],
                    )
                    if isinstance(regs, dict):
                        if regs.get('title_block') and not perception_v1.get('regions', {}).get('title_block'):
                            perception_v1.setdefault('regions', {})['title_block'] = regs.get('title_block')
                        if isinstance(regs.get('viewports'), list):
                            perception_v1.setdefault('regions', {}).setdefault('viewports', []).extend(regs.get('viewports') or [])
                        if isinstance(regs.get('notes'), list):
                            perception_v1.setdefault('regions', {}).setdefault('notes', []).extend(regs.get('notes') or [])
                        cov = regs.get('coverage')
                        if isinstance(cov, (int, float)):
                            perception_v1.setdefault('validation', {}).setdefault('region_coverage_pct_by_page', {})[str(int(page_idx))] = float(max(0.0, min(1.0, float(cov))))
                except Exception:
                    pass

                # drawing-likeness score (helpful for engineering drawing PDFs)
                try:
                    ds = _drawing_likeness(arr)
                    if isinstance(ds, dict) and float(ds.get('score') or 0.0) > float(drawing_stats_overall.get('score') or 0.0):
                        drawing_stats_overall = ds
                    pre_info = {**pre_info, "drawing": ds}
                except Exception:
                    pass

                def _run_ocr(arr_in):
                    rr = ocr(arr_in)
                    if isinstance(rr, tuple) and len(rr) >= 1:
                        rr = rr[0]
                    if rr is None:
                        rr = []
                    parts_local: List[str] = []
                    words_local: List[Dict[str, Any]] = []
                    confs: List[float] = []
                    for item in rr:
                        try:
                            box, txt, conf = item
                        except Exception:
                            continue
                        if not txt:
                            continue
                        parts_local.append(str(txt))
                        confs.append(float(conf))
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        x1, y1, x2, y2 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
                        nx = max(0.0, min(1.0, x1 / float(w or 1)))
                        ny = max(0.0, min(1.0, y1 / float(h or 1)))
                        nw = max(0.0, min(1.0, (x2 - x1) / float(w or 1)))
                        nh = max(0.0, min(1.0, (y2 - y1) / float(h or 1)))
                        words_local.append({"text": str(txt), "confidence": float(conf), "bbox": [nx, ny, nw, nh], "page": page_idx})
                    page_txt = re.sub(r"\s+", " ", " ".join(parts_local)).strip()
                    avg_conf = (sum(confs) / len(confs)) if confs else 0.0
                    score = (len(page_txt) * 1.0) + (avg_conf * 50.0)
                    return page_txt, words_local, avg_conf, score

                page_text_a, words_a, avg_a, score_a = _run_ocr(arr)
                page_text_b, words_b, avg_b, score_b = _run_ocr(arr2)
                if score_b >= score_a:
                    page_text = page_text_b
                    words_this = words_b
                    chosen = {"variant": "preprocessed", "avg_conf": avg_b}
                else:
                    page_text = page_text_a
                    words_this = words_a
                    chosen = {"variant": "original", "avg_conf": avg_a}

                for wd in words_this:
                    ocr_words.append({"text": wd["text"], "confidence": wd["confidence"], "bbox": wd["bbox"]})

                if page_text:
                    merged_parts.append(page_text)
                ocr_pages.append({
                    "page": page_idx,
                    "image": os.path.basename(pth),
                    "text": page_text,
                    "words": words_this,
                    "preprocess": pre_info,
                    "choice": chosen,
                    "extracted_fields": {"dimensions": _extract_dimensions(page_text)}
                })

            text_full = "\n\n".join(merged_parts).strip()
            extracted_fields = {"dimensions": _extract_dimensions(text_full)}
            result["logs"].append({"stage": "ocr", "engine": "rapidocr-onnx", "words": len(ocr_words), "pages": len(ocr_pages)})
        except Exception as e:
            result["logs"].append({"stage": "ocr", "engine": "rapidocr-onnx", "error": str(e)})
            try:
                perception_v1["validation"]["failure_reasons"].append(f"rapidocr_failed:{str(e)}")
            except Exception:
                pass

    if not ocr_engine_ok and not text_full:
        try:
            import pytesseract
            text_full = (pytesseract.image_to_string(primary_path) or '').strip()
            extracted_fields = {"dimensions": _extract_dimensions(text_full)}
            result["logs"].append({"stage": "ocr", "engine": "tesseract"})
        except Exception as e2:
            result["logs"].append({"stage": "ocr", "engine": "tesseract", "error": str(e2)})
            text_full = ''

    if is_pdf and (pdf_text_by_page or ocr_pages):
        ocr_by_page: Dict[int, Dict[str, Any]] = {int(p.get('page') or 0): p for p in ocr_pages if int(p.get('page') or 0) > 0}
        combined_pages: List[Dict[str, Any]] = []
        merged_display_parts: List[str] = []
        max_page = 0
        if page_paths:
            max_page = len(page_paths)
        if pdf_text_by_page:
            max_page = max(max_page, max(pdf_text_by_page.keys()))
        if ocr_by_page:
            max_page = max(max_page, max(ocr_by_page.keys()))

        for page_idx in range(1, max_page + 1):
            p_pdf = str(pdf_text_by_page.get(page_idx) or '').strip()
            p_ocr_obj = ocr_by_page.get(page_idx) or {}
            p_ocr = str(p_ocr_obj.get('text') or '').strip()
            display = p_pdf if len(p_pdf) >= 40 else p_ocr
            if display:
                merged_display_parts.append(display)
            combined_pages.append({
                "page": page_idx,
                "image": os.path.basename(page_paths[page_idx - 1]) if page_paths and (page_idx - 1) < len(page_paths) else None,
                "text": display,
                "pdf_text": p_pdf,
                "ocr_text": p_ocr,
                "words": p_ocr_obj.get('words') or [],
                "preprocess": p_ocr_obj.get('preprocess') or {"deskew_deg": 0.0, "line_suppression": False},
                "choice": p_ocr_obj.get('choice') or {"variant": "pdf-text" if p_pdf else "ocr", "avg_conf": 1.0 if p_pdf else 0.0},
                "extracted_fields": {"dimensions": _extract_dimensions(display), "contacts": _extract_contacts(display)},
            })

        text_full = "\n\n".join(merged_display_parts).strip()
        extracted_fields = {"dimensions": _extract_dimensions(text_full), "contacts": _extract_contacts(text_full)}
        ocr_pages = combined_pages
        result["logs"].append({"stage": "multimodal-text", "pages": len(ocr_pages), "has_pdf_text": bool(pdf_text_by_page), "has_ocr": bool(ocr_by_page)})

    try:
        # basic validation metric: closed shapes vs lines ratio (aggregate)
        lc = int(len(perception_v1.get('lines') or []))
        cc = int(len(perception_v1.get('closed_shapes') or []))
        perception_v1["validation"]["closed_loop_line_ratio"] = float(min(1.0, cc / float(max(1, lc))))
    except Exception:
        pass

    try:
        perception_v1["page_text"] = _perception_text_blocks_to_page_text(perception_v1.get('text_blocks') or [], max_chars_per_page=12000)
    except Exception:
        pass

    try:
        covs = perception_v1.get('validation', {}).get('region_coverage_pct_by_page')
        if isinstance(covs, dict) and covs:
            vals = [float(v) for v in covs.values() if isinstance(v, (int, float))]
            if vals:
                perception_v1["validation"]["region_coverage_pct"] = float(sum(vals) / float(len(vals)))
    except Exception:
        pass

    engineering_by_page: List[Dict[str, Any]] = []
    try:
        if ocr_pages and isinstance(ocr_pages, list):
            for p in ocr_pages:
                if not isinstance(p, dict):
                    continue
                words = p.get('words')
                if not isinstance(words, list) or not words:
                    continue
                title_words = []
                border_words = []
                main_words = []
                for wd in words:
                    bb = wd.get('bbox') if isinstance(wd, dict) else None
                    if not (isinstance(bb, list) and len(bb) == 4):
                        continue
                    x, y, w, h = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                    cx = x + w * 0.5
                    cy = y + h * 0.5
                    in_bottom = cy >= 0.78
                    in_right = cx >= 0.58
                    in_left = cx <= 0.42
                    in_border = (cy <= 0.12) or (cy >= 0.88) or (cx <= 0.12) or (cx >= 0.88)
                    in_title = in_bottom and (in_right or in_left) and (x >= 0.35 or x <= 0.65)

                    if in_title:
                        title_words.append(wd)
                    elif in_border:
                        border_words.append(wd)
                    else:
                        main_words.append(wd)

                title_lines = _group_words_into_lines(title_words)
                border_lines = _group_words_into_lines(border_words)
                main_lines = _group_words_into_lines(main_words)

                tb = _extract_titleblock_fields(title_lines)

                border_text = "\n".join([str(ln.get('text') or '') for ln in border_lines if isinstance(ln, dict)])
                border_scale = ''
                try:
                    m = re.search(r"\bscale\b\s*[:=]?\s*([^\n]{1,40})", border_text, flags=re.IGNORECASE)
                    if m:
                        border_scale = str(m.group(1) or '').strip().rstrip('.,;')
                except Exception:
                    border_scale = ''

                main_text = "\n".join([str(ln.get('text') or '') for ln in main_lines if isinstance(ln, dict)])
                engineering_by_page.append({
                    "page": int(p.get('page') or 0) or None,
                    "title_block": {
                        "text": "\n".join([str(ln.get('text') or '') for ln in title_lines if isinstance(ln, dict)]),
                        "fields": tb.get('fields') if isinstance(tb, dict) else {},
                        "kv": tb.get('kv') if isinstance(tb, dict) else [],
                    },
                    "border_notes": {
                        "text": border_text,
                        "scale_hint": border_scale,
                    },
                    "main_area": {
                        "text": main_text,
                    },
                })

            if engineering_by_page:
                engineering_by_page.sort(key=lambda e: int(e.get('page') or 0))
    except Exception as e:
        result["logs"].append({"stage": "engineering-layout", "error": str(e)})

    if not extracted_fields:
        extracted_fields = {"dimensions": _extract_dimensions(text_full), "contacts": _extract_contacts(text_full)}
    elif isinstance(extracted_fields, dict) and "contacts" not in extracted_fields:
        extracted_fields["contacts"] = _extract_contacts(text_full)

    if ocr_pages and isinstance(ocr_pages, list):
        for p in ocr_pages:
            try:
                ef = p.get("extracted_fields") if isinstance(p, dict) else None
                if not isinstance(ef, dict):
                    ef = {}
                if "contacts" not in ef:
                    ef["contacts"] = _extract_contacts(str(p.get("text") or ''))
                p["extracted_fields"] = ef
            except Exception:
                continue

    summary = text_full[:400] + ('...' if len(text_full) > 400 else '')
    result["text"] = {
        "full_text": text_full,
        "summary": summary,
        "words": ocr_words,
        "extracted_fields": extracted_fields,
        "pages": ocr_pages,
    }

    try:
        result["perception_v1"] = perception_v1
    except Exception:
        pass

    if engineering_by_page:
        try:
            result["text"]["engineering"] = {
                "pages": engineering_by_page,
            }
        except Exception:
            pass

    # Engineering Drawing DOM (grammar-first). Only populate when border gate passes.
    try:
        border_conf = None
        border_bbox = None
        if isinstance(engineering_dom.get('sheet'), dict) and isinstance(engineering_dom.get('sheet', {}).get('border'), dict):
            border_conf = float(engineering_dom.get('sheet', {}).get('border', {}).get('confidence_score') or 0.0)
            border_bbox = engineering_dom.get('sheet', {}).get('border', {}).get('bbox')

        if border_conf is not None and border_conf >= 0.9 and ocr_pages:
            # Use page 1 OCR words for regions.
            p1 = None
            for pp in ocr_pages:
                if isinstance(pp, dict) and int(pp.get('page') or 0) == 1:
                    p1 = pp
                    break
            if p1 is None and isinstance(ocr_pages[0], dict):
                p1 = ocr_pages[0]

            words1 = p1.get('words') if isinstance(p1, dict) else []
            if not isinstance(words1, list):
                words1 = []

            # Re-open first page image to compute edge grid.
            rgb0 = None
            try:
                from PIL import Image
                import numpy as np
                rgb0 = np.array(Image.open(primary_path).convert('RGB'))
            except Exception:
                rgb0 = None

            # Locate title block anywhere.
            tb_loc = _locate_title_block_anywhere(words1, border_bbox, rgb0)
            tb_bbox = tb_loc.get('bbox')
            tb_conf = float(tb_loc.get('confidence') or 0.0)

            # Secondary viewport proposals using Hough-segment clustering (helps when leader lines connect everything).
            hough_view_bboxes: List[List[float]] = []
            try:
                if rgb0 is not None and isinstance(border_bbox, list) and len(border_bbox) == 4:
                    view_info0 = _detect_view_clusters_in_viewport(rgb0, border_bbox, tb_bbox)
                    if isinstance(view_info0, dict):
                        # Prefer raw clusters (always present) over filtered views.
                        cl = view_info0.get('clusters')
                        if not isinstance(cl, list):
                            cl = view_info0.get('views')
                        if isinstance(cl, list):
                            for vv in cl[:12]:
                                bb = None
                                segs = 0
                                dens = 0.0
                                if isinstance(vv, dict):
                                    bb = vv.get('bbox') if vv.get('bbox') is not None else vv.get('bounding_box')
                                    try:
                                        segs = int(vv.get('segment_count') or vv.get('stats', {}).get('segment_count') or 0)
                                    except Exception:
                                        segs = 0
                                    try:
                                        dens = float(vv.get('density') or vv.get('stats', {}).get('density') or 0.0)
                                    except Exception:
                                        dens = 0.0
                                if isinstance(bb, list) and len(bb) == 4:
                                    try:
                                        x1, y1, x2, y2 = [float(x) for x in bb]
                                        ww = float(x2 - x1)
                                        hh = float(y2 - y1)
                                        # Accept either sufficiently large clusters OR small but line-dense clusters.
                                        ok = (ww >= 70.0 and hh >= 70.0) or (segs >= 6) or (dens >= 2.0)
                                        if ok:
                                            pad = 18.0
                                            x1p = max(0.0, x1 - pad)
                                            y1p = max(0.0, y1 - pad)
                                            x2p = min(float((rgb0.shape[1] if rgb0 is not None else (x2 + pad)) ), x2 + pad)
                                            y2p = min(float((rgb0.shape[0] if rgb0 is not None else (y2 + pad)) ), y2 + pad)
                                            hough_view_bboxes.append([x1p, y1p, x2p, y2p])
                                    except Exception:
                                        continue

                        try:
                            engineering_dom["diagnostics"]["warnings"].append(
                                f"hough_clusters={len(hough_view_bboxes)}"
                            )
                        except Exception:
                            pass
            except Exception:
                hough_view_bboxes = []

            # Deterministic regions: TITLE_BLOCK (anywhere) + multiple VIEWPORT clusters.
            regions: List[Dict[str, Any]] = []
            regions.append({
                "id": "r_title",
                "type": "TITLE_BLOCK",
                "bounding_box": tb_bbox,
                "confidence": tb_conf,
                "method": tb_loc.get('method'),
                "stats": tb_loc.get('stats') or {},
            })

            vps: List[Dict[str, Any]] = []
            try:
                if rgb0 is not None and isinstance(border_bbox, list) and len(border_bbox) == 4:
                    vps = _extract_viewport_clusters(rgb0, border_bbox, tb_bbox, words=words1 if isinstance(words1, list) else None)
            except Exception:
                vps = []

            # If edge/gutter clustering yields only 0-1 viewports but Hough clustering found multiple, prefer Hough.
            try:
                if (not isinstance(vps, list) or len(vps) <= 1) and len(hough_view_bboxes) >= 2:
                    vps = []
                    for bb in hough_view_bboxes[:8]:
                        try:
                            x1, y1, x2, y2 = [float(x) for x in bb]
                            vps.append({"bbox": [x1, y1, x2, y2], "area": float(max(1.0, (x2 - x1) * (y2 - y1)))})
                        except Exception:
                            continue
            except Exception:
                pass

            if not vps and isinstance(border_bbox, list) and len(border_bbox) == 4:
                bx1, by1, bx2, by2 = [float(x) for x in border_bbox]
                vps = [{"bbox": [bx1, by1, bx2, by2], "area": float(max(1.0, (bx2 - bx1) * (by2 - by1)))}]

            viewport_leaf_ids: List[str] = []
            vp_child_debug: List[Dict[str, Any]] = []
            for i, vp in enumerate(vps, start=1):
                bb = vp.get('bbox')
                parent_id = f"r_view_{i}"

                children_info = None
                try:
                    if rgb0 is not None and isinstance(bb, list) and len(bb) == 4:
                        children_info = _split_viewport_children_v2(
                            rgb0,
                            bb,
                            tb_bbox,
                            words=words1 if isinstance(words1, list) else None,
                        )
                except Exception:
                    children_info = None

                children = []
                if isinstance(children_info, dict):
                    children = children_info.get('children')
                    if not isinstance(children, list):
                        children = []
                    try:
                        vp_child_debug.append({
                            "parent": parent_id,
                            "status": children_info.get('status'),
                            "children": len(children),
                        })
                    except Exception:
                        pass

                has_children = bool(children and len(children) >= 2)
                regions.append({
                    "id": parent_id,
                    "type": "VIEWPORT",
                    "bounding_box": bb,
                    "confidence": 0.65,
                    "method": "edge_components",
                    "stats": {
                        "area": float(vp.get('area') or 0.0),
                        "level": 1,
                        "is_leaf": (not has_children),
                    },
                })

                if has_children:
                    for j, ch in enumerate(children, start=1):
                        ch_bb = ch.get('bbox') if isinstance(ch, dict) else None
                        if not (isinstance(ch_bb, list) and len(ch_bb) == 4):
                            continue
                        ch_id = f"{parent_id}_c{j}"
                        viewport_leaf_ids.append(ch_id)
                        regions.append({
                            "id": ch_id,
                            "type": "VIEWPORT",
                            "bounding_box": ch_bb,
                            "confidence": float(ch.get('confidence') or 0.0),
                            "method": "viewport_child_v2",
                            "stats": {
                                "area": float(ch.get('area') or 0.0),
                                "parent_id": parent_id,
                                "level": 2,
                                "is_leaf": True,
                            },
                        })
                else:
                    viewport_leaf_ids.append(parent_id)

            try:
                engineering_dom["diagnostics"]["viewport_child_v2"] = vp_child_debug
            except Exception:
                pass

            engineering_dom["regions"] = regions

            # Views stubs (Layer 1->2 handoff). View typing happens later.
            views_out: List[Dict[str, Any]] = []
            leaf_regions: List[Dict[str, Any]] = []
            try:
                for r in regions:
                    if not isinstance(r, dict):
                        continue
                    if str(r.get('type') or '').upper() != 'VIEWPORT':
                        continue
                    rid = str(r.get('id') or '')
                    if rid and rid in viewport_leaf_ids:
                        leaf_regions.append(r)
            except Exception:
                leaf_regions = []

            if not leaf_regions:
                leaf_regions = [r for r in regions if isinstance(r, dict) and str(r.get('type') or '').upper() == 'VIEWPORT']

            for i, vp in enumerate(leaf_regions, start=1):
                views_out.append({
                    "id": f"V{i}",
                    "region_id": str(vp.get('id') or f"r_view_{i}"),
                    "type": "UNKNOWN",
                    "bounding_box": vp.get('bounding_box'),
                    "confidence": float(vp.get('confidence') or 0.0),
                    "evidence": {"method": vp.get('method')},
                })
            engineering_dom["views"] = views_out

            # Title block metadata extraction using words inside detected title bbox.
            tb_words = []
            if isinstance(tb_bbox, list) and len(tb_bbox) == 4:
                x1, y1, x2, y2 = [float(x) for x in tb_bbox]
                for wd in words1:
                    bb = wd.get('bbox') if isinstance(wd, dict) else None
                    if not (isinstance(bb, list) and len(bb) == 4):
                        continue
                    xx, yy, ww, hh = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                    cx = xx + ww * 0.5
                    cy = yy + hh * 0.5
                    if (x1 <= cx <= x2) and (y1 <= cy <= y2):
                        tb_words.append(wd)
            tb_lines = _group_words_into_lines(tb_words)
            tb = _extract_titleblock_fields(tb_lines)
            raw_tb = "\n".join([str(ln.get('text') or '') for ln in tb_lines if isinstance(ln, dict)])
            fields = tb.get('fields') if isinstance(tb, dict) else {}
            engineering_dom["metadata"] = {
                "region_id": "r_title",
                "fields": fields if isinstance(fields, dict) else {},
                "raw_text": raw_tb,
            }

            # Minimal viewport intelligence: view clustering + dimension candidates.
            try:
                if rgb0 is not None and isinstance(border_bbox, list) and len(border_bbox) == 4:
                    view_info = _detect_view_clusters_in_viewport(rgb0, border_bbox, tb_bbox)
                    if isinstance(view_info, dict):
                        vws = view_info.get('views')
                        if isinstance(vws, list):
                            # Do NOT overwrite deterministic view stubs derived from region clustering.
                            engineering_dom["view_cluster_details"] = vws
                        try:
                            engineering_dom["diagnostics"]["warnings"].append(
                                f"viewport_segments={int(view_info.get('segments') or 0)}"
                            )
                        except Exception:
                            pass
            except Exception as e:
                engineering_dom["diagnostics"]["warnings"].append(f"view_clustering_failed: {str(e)}")

            try:
                if isinstance(words1, list) and isinstance(border_bbox, list) and len(border_bbox) == 4:
                    dim_cands = _extract_dimension_candidates_from_words(words1, border_bbox, tb_bbox)
                    if isinstance(engineering_dom.get("annotations"), dict):
                        engineering_dom["annotations"]["dimensions"] = dim_cands
            except Exception as e:
                engineering_dom["diagnostics"]["warnings"].append(f"dimension_candidates_failed: {str(e)}")

        else:
            # Gate failed; keep diagnostics and stop at DOM level.
            if border_conf is not None and border_conf < 0.9:
                engineering_dom["diagnostics"]["warnings"].append("parsing_aborted_due_to_border")
    except Exception as e:
        engineering_dom["diagnostics"]["errors"].append(f"dom_build_failed: {str(e)}")

    try:
        if isinstance(engineering_dom, dict) and engineering_dom.get('sheet'):
            result["text"]["engineering_drawing_dom"] = engineering_dom
    except Exception:
        pass

    # Canonical EngineeringDrawing object (Option A): built from DOM + extracted fields.
    try:
        if isinstance(engineering_dom, dict) and isinstance(engineering_dom.get('sheet'), dict):
            result["engineering_drawing"] = _build_engineering_drawing_canonical(
                engineering_dom=engineering_dom,
                extracted_fields=extracted_fields if isinstance(extracted_fields, dict) else {},
                text_full=str(text_full or ''),
                is_pdf=bool(is_pdf),
            )
    except Exception as e:
        result["logs"].append({"stage": "engineering_drawing_canonical", "error": str(e)})

    try:
        dims = extracted_fields.get("dimensions") if isinstance(extracted_fields, dict) else None
        dims_count = len(dims) if isinstance(dims, list) else 0
        contacts = extracted_fields.get("contacts") if isinstance(extracted_fields, dict) else None
        has_email = bool(isinstance(contacts, dict) and contacts.get("emails"))
        has_phone = bool(isinstance(contacts, dict) and contacts.get("phones"))
        has_url = bool(isinstance(contacts, dict) and contacts.get("urls"))

        # Engineering DOM signals (grammar-first). If border gate passes, it overrides generic visual labels.
        dom_border_conf = 0.0
        dom_has_metadata_fields = False
        try:
            eng_dom = None
            if isinstance(result.get("text"), dict):
                eng_dom = result.get("text", {}).get("engineering_drawing_dom")
            if isinstance(eng_dom, dict):
                dom_border_conf = float(((eng_dom.get("sheet") or {}).get("border") or {}).get("confidence_score") or 0.0)
                dom_fields = ((eng_dom.get("metadata") or {}).get("fields") or {})
                # Tighten: do not treat arbitrary table text as "metadata". Require >=2 canonical title-block keys.
                if isinstance(dom_fields, dict):
                    key_map = {
                        'title': ['title', 'part_name', 'name'],
                        'drawing_no': ['drawing_no', 'dwg_no', 'dwg', 'drg_no', 'drg'],
                        'scale': ['scale'],
                        'sheet': ['sheet', 'sheet_no'],
                        'revision': ['revision', 'rev'],
                        'date': ['date'],
                        'material': ['material', 'matl'],
                        'finish': ['finish'],
                    }
                    hits = 0
                    for _, keys in key_map.items():
                        for k in keys:
                            v = dom_fields.get(k)
                            if v is None:
                                continue
                            if str(v).strip():
                                hits += 1
                                break
                    dom_has_metadata_fields = bool(hits >= 2)
        except Exception:
            dom_border_conf = 0.0
            dom_has_metadata_fields = False

        # Engineering/title-block keyword hits
        txt_l = str(text_full or '').lower()
        kw_list = [
            'drawing', 'dwg', 'rev', 'revision', 'scale', 'sheet', 'title', 'projection',
            'tolerance', 'tolerances', 'material', 'finish', 'mm', 'inch', 'iso', 'ansi',
            'typ.', 'typical', 'deburr', 'machining', 'surface finish'
        ]
        kw_hits = [k for k in kw_list if k in txt_l]
        kw_score = float(min(1.0, len(kw_hits) / 6.0))

        draw_score = float(drawing_stats_overall.get('score') or 0.0)
        has_pdf_text_any = bool(is_pdf and pdf_text_by_page and any(v.strip() for v in pdf_text_by_page.values()))
        has_ocr_any = bool(ocr_pages and any(str(p.get('ocr_text') or p.get('text') or '').strip() for p in ocr_pages))
        # A rendered PDF page or an uploaded image is always a "visual" artifact even if object detection returns 0.
        has_visual = bool(page_paths)

        # Negative override: if the content is strongly photo-like (people/vehicles/etc), do not classify as engineering drawing.
        top_label = None
        top_score = 0.0
        try:
            if isinstance(result.get('scene'), dict):
                top_label = str((result.get('scene') or {}).get('top_label') or '').strip().lower() or None
                top_score = float((result.get('scene') or {}).get('top_score') or 0.0)
        except Exception:
            top_label = None
            top_score = 0.0

        obj_names: List[str] = []
        try:
            if isinstance(result.get('objects'), list):
                for o in result.get('objects')[:12]:
                    if not isinstance(o, dict):
                        continue
                    nm = str(o.get('class') or o.get('label') or o.get('className') or '').strip().lower()
                    if nm:
                        obj_names.append(nm)
        except Exception:
            obj_names = []

        photo_like = False
        try:
            photo_labels = [
                'a person', 'a portrait photo', 'a selfie', 'a group of people', 'a man', 'a woman', 'a human face',
                'a motorcycle', 'a car', 'a bicycle', 'a scooter', 'a vehicle',
            ]
            if top_label and top_score >= 0.35:
                if any(lbl in top_label for lbl in photo_labels):
                    photo_like = True
            if any(n in obj_names for n in ['person', 'motorcycle', 'car', 'bicycle', 'truck', 'bus']):
                photo_like = True
        except Exception:
            photo_like = False

        # Stronger engineering-drawing gating: border alone is not enough.
        # Require border + at least one additional, drawing-specific signal.
        # Note: PDFs often contain many numbers (page counts, totals, IDs) that look like "dimensions".
        # Only treat dimensions_count as a drawing signal if it co-occurs with other drawing-like evidence.
        dims_count_is_signal = bool(dims_count >= 2)
        if is_pdf and dims_count_is_signal:
            dims_count_is_signal = bool((draw_score >= 0.50) or (kw_score >= 0.50) or bool(dom_has_metadata_fields))

        has_dims_signal = bool(dims_count_is_signal)
        has_kw_signal = bool(kw_score >= 0.55)
        has_draw_signal = bool(draw_score >= 0.60)
        has_dom_signal = bool(dom_has_metadata_fields)
        eng_signal_count = int(has_dims_signal) + int(has_kw_signal) + int(has_draw_signal) + int(has_dom_signal)
        dom_border_strong = bool(dom_border_conf >= 0.9)
        # Tighten: border must be paired with metadata/title-block OR strong drawing/keyword evidence.
        strong_engineering_evidence = (
            bool(dom_border_strong and dom_has_metadata_fields and (has_draw_signal or has_kw_signal or has_dims_signal))
            or bool(eng_signal_count >= 2 and (has_draw_signal or has_kw_signal))
        )
        kind = "unknown"
        if is_pdf:
            # PDFs with selectable text are frequently resumes/forms/docs. Only call it an engineering drawing
            # when drawing evidence is strong.
            if photo_like:
                kind = "image" if (has_visual and not has_ocr_any) else ("document_with_illustration" if (has_visual and has_ocr_any) else "document")
            elif strong_engineering_evidence and (draw_score >= 0.55 or kw_score >= 0.55 or dims_count >= 2 or dom_border_strong):
                kind = "engineering_drawing"
            elif (has_email or has_phone) and len(ocr_pages or []) <= 2:
                kind = "business_card"
            elif has_pdf_text_any and has_ocr_any:
                kind = "mixed_pdf"
            elif has_pdf_text_any:
                kind = "text_document"
            elif has_ocr_any:
                kind = "scanned_document"
            else:
                kind = "pdf"
        else:
            # For non-PDF images, default to image when OCR is empty. Do not label these as document.
            if (not has_ocr_any) and has_visual and (draw_score < 0.55):
                kind = "image"
            elif photo_like:
                kind = "image" if (has_visual and not has_ocr_any) else ("document_with_illustration" if (has_visual and has_ocr_any) else "document")
            elif strong_engineering_evidence:
                kind = "engineering_drawing"
            elif has_email or has_phone or has_url:
                kind = "business_card"
            elif has_visual and not has_ocr_any:
                kind = "image"
            elif has_ocr_any and not has_visual:
                kind = "document"
            elif has_visual and has_ocr_any:
                kind = "document_with_illustration"

        score = 0.15
        score += 0.35 if (has_pdf_text_any or has_ocr_any) else 0.0
        score += 0.25 if has_visual else 0.0
        score += 0.25 if dims_count >= 2 else 0.0
        score += 0.20 * max(kw_score, draw_score)
        score += 0.20 if dom_border_conf >= 0.9 else 0.0
        score += 0.10 if dom_has_metadata_fields else 0.0
        score = float(max(0.0, min(1.0, score)))

        result["judgement"] = {
            "kind": kind,
            "confidence": score,
            "evidence": {
                "is_pdf": bool(is_pdf),
                "has_pdf_text": bool(has_pdf_text_any),
                "has_ocr_text": bool(has_ocr_any),
                "has_objects": bool(has_visual),
                "dimensions_count": int(dims_count),
                "drawing_likeness": drawing_stats_overall,
                "engineering_keywords": kw_hits[:25],
                "engineering_dom_border_confidence": float(dom_border_conf),
                "engineering_dom_has_metadata": bool(dom_has_metadata_fields),
                "engineering_signal_count": int(eng_signal_count),
                "engineering_strong_evidence": bool(strong_engineering_evidence),
                "photo_like_override": bool(photo_like),
                "photo_like_top_label": top_label,
                "photo_like_top_score": float(top_score),
                "top_label": (result.get("scene") or {}).get("top_label") if isinstance(result.get("scene"), dict) else None,
            },
        }
    except Exception as e:
        result["logs"].append({"stage": "judgement", "error": str(e)})

    # 4) CLIP coarse guess (optional)
    clip_pair = models.get_clip()
    if clip_pair is not None:
        try:
            import clip  # type: ignore
            import torch
            from PIL import Image
            model, preprocess = clip_pair
            image = preprocess(Image.open(primary_path)).unsqueeze(0)

            # Prefer broad label taxonomy scoring (COCO + products + engineering parts)
            try:
                labels = get_labels()
                result["scene"] = _clip_taxonomy_rank(model, image, labels, topk=7)
                result["logs"].append({"stage": "clip", "match": result["scene"].get("top_label"), "score": result["scene"].get("top_score"), "source": "taxonomy"})
            except Exception as e:
                result["logs"].append({"stage": "clip", "error": str(e), "source": "taxonomy"})

            # Fallback: small curated prompt list
            text_prompts = [
                "a person",
                "a human face",
                "a portrait photo",
                "a selfie",
                "a group of people",
                "a man",
                "a woman",
                "a human head",
                "a pair of glasses",
                "a document",
                "a printed document",
                "a form",
                "a filled form",
                "a sheet of paper",
                "a notebook",
                "a notebook page",
                "a book",
                "a book cover",
                "a page of text",
                "a qr code",
                "a barcode",
                "a motorcycle",
                "a motorbike",
                "a scooter",
                "a moped",
                "a bicycle",
                "a car",
                "a truck",
                "a helmet",
                "a mobile phone holder",
                "a phone holder",
                "a phone mount",
                "a car phone mount",
                "a bike phone mount",
                "a tripod",
                "a camera mount",
                "a clamp",
                "headphones",
                "a mug",
                "a chair",
                "a water purifier",
                "a smartphone",
                "a mobile phone",
            ]
            if "scene" not in result or not result.get("scene", {}).get("candidates"):
                tokens = clip.tokenize(text_prompts)
                with torch.no_grad():
                    imf = model.encode_image(image)
                    tf = model.encode_text(tokens)
                    sims = (imf @ tf.T).squeeze(0)
                    probs = torch.softmax(sims, dim=0)
                    best = int(torch.argmax(probs).item())
                    score = float(probs[best].item())

                    topk = int(min(5, probs.shape[0]))
                    vals, idxs = torch.topk(probs, k=topk)
                    candidates = [
                        {"label": str(text_prompts[int(i)]), "score": float(v.item())}
                        for v, i in zip(vals, idxs)
                    ]

                result["scene"] = {
                    "top_label": text_prompts[best],
                    "top_score": score,
                    "candidates": candidates,
                    "source": "clip-prompts",
                }
                result["logs"].append({"stage": "clip", "match": text_prompts[best], "score": score, "source": "prompts"})

            # Attach as scene tag if no objects
            if not result["objects"] and result.get("scene", {}).get("top_label"):
                result["objects"].append({"id": "cx1", "class_clip_guess": result["scene"]["top_label"], "clip_score": float(result["scene"].get("top_score") or 0.0)})
        except Exception as e:
            result["logs"].append({"stage": "clip", "error": str(e)})
    else:
        result["logs"].append({"stage": "clip", "status": "skipped"})

    try:
        j = result.get("judgement") if isinstance(result.get("judgement"), dict) else None
        if j is None:
            j = {"kind": "unknown", "confidence": 0.15, "evidence": {"is_pdf": bool(is_pdf)}}
            result["judgement"] = j

        kind = str(j.get("kind") or "unknown")
        conf = j.get("confidence")
        conf_s = ''
        try:
            if isinstance(conf, (int, float)):
                conf_s = f" ({int(max(0.0, min(1.0, float(conf))) * 100)}%)"
        except Exception:
            conf_s = ''

        top_scene = None
        try:
            if isinstance(result.get("scene"), dict):
                top_scene = result.get("scene", {}).get("top_label")
        except Exception:
            top_scene = None

        objs = result.get("objects") if isinstance(result.get("objects"), list) else []
        obj_names = []
        for o in objs[:5]:
            if isinstance(o, dict):
                nm = o.get("class") or o.get("class_clip_guess")
                if nm and str(nm) not in obj_names:
                    obj_names.append(str(nm))

        dims = None
        try:
            if isinstance(extracted_fields, dict):
                dims = extracted_fields.get("dimensions")
        except Exception:
            dims = None
        dims_count = len(dims) if isinstance(dims, list) else 0

        contacts = extracted_fields.get("contacts") if isinstance(extracted_fields, dict) else None
        emails = contacts.get("emails") if isinstance(contacts, dict) else None
        phones = contacts.get("phones") if isinstance(contacts, dict) else None
        urls = contacts.get("urls") if isinstance(contacts, dict) else None
        address_like = contacts.get("address_like") if isinstance(contacts, dict) else None
        emails = [str(x) for x in emails] if isinstance(emails, list) else []
        phones = [str(x) for x in phones] if isinstance(phones, list) else []
        urls = [str(x) for x in urls] if isinstance(urls, list) else []
        address_like = [str(x) for x in address_like] if isinstance(address_like, list) else []

        preview = ''
        try:
            preview = re.sub(r"\s+", " ", str(text_full or '')).strip()
        except Exception:
            preview = ''
        if len(preview) > 220:
            preview = preview[:220] + '…'

        lines = []
        if kind == 'business_card':
            lines.append(f"Prima facie: this looks like a visiting card{conf_s}.")
            if emails:
                lines.append("Email: " + ", ".join(emails[:3]))
            if phones:
                lines.append("Phone: " + ", ".join(phones[:3]))
            if urls:
                lines.append("Website: " + ", ".join(urls[:3]))
            if address_like:
                lines.append("Address (best guess): " + address_like[0])
        elif kind == 'engineering_drawing':
            lines.append(f"Prima facie: this looks like an engineering drawing{conf_s}.")
            try:
                dom = (result.get('text') or {}).get('engineering_drawing_dom') if isinstance(result.get('text'), dict) else None
            except Exception:
                dom = None

            try:
                bc = float((((dom or {}).get('sheet') or {}).get('border') or {}).get('confidence_score') or 0.0)
                if bc:
                    lines.append(f"Border confidence (gate): {bc:.2f}")
            except Exception:
                pass

            try:
                vws = (dom or {}).get('views')
                views_count = len(vws) if isinstance(vws, list) else 0
                if views_count:
                    lines.append(f"Detected view regions (clusters): {views_count}")
            except Exception:
                pass

            try:
                anns = (dom or {}).get('annotations')
                dim_cands = (anns or {}).get('dimensions') if isinstance(anns, dict) else None
                dim_cand_count = len(dim_cands) if isinstance(dim_cands, list) else 0
                if dim_cand_count:
                    lines.append(f"Dimension text candidates (viewport): {dim_cand_count}")
            except Exception:
                pass
            if dims_count:
                lines.append(f"Detected {dims_count} dimension-like values.")

            # Intentionally suppress generic vision labels (e.g., CLIP/YOLO 'clock') for engineering drawings.
            # They are usually misleading and reduce trust; title block + structured extraction is the source of truth.
            try:
                ed = result.get('engineering_drawing') if isinstance(result.get('engineering_drawing'), dict) else None
            except Exception:
                ed = None

            md = (ed or {}).get('metadata') if isinstance(ed, dict) else None
            md_ev = (md or {}).get('evidence') if isinstance(md, dict) else None

            # Prefer canonical metadata (already gated against OCR garbage).
            try:
                pn = str((md or {}).get('part_name') or '').strip()
                pn_c = float(((md_ev or {}).get('part_name') or {}).get('confidence') or 0.0) if isinstance(md_ev, dict) else 0.0
                if pn and pn_c >= 0.6:
                    lines.append(f"Part: {pn}")
            except Exception:
                pass
            try:
                mat = str((md or {}).get('material') or '').strip()
                mat_c = float(((md_ev or {}).get('material') or {}).get('confidence') or 0.0) if isinstance(md_ev, dict) else 0.0
                if mat and mat_c >= 0.6:
                    lines.append(f"Material: {mat}")
            except Exception:
                pass

            # Only show a short text preview when we don't have any usable metadata.
            if (not isinstance(md, dict) or not (str(md.get('part_name') or '').strip() or str(md.get('material') or '').strip())) and preview:
                lines.append("Text preview: " + preview)
        elif kind in ('text_document', 'scanned_document', 'mixed_pdf', 'document', 'document_with_illustration', 'pdf'):
            doc_kind = 'PDF document' if is_pdf else 'document'
            lines.append(f"Prima facie: this looks like a {doc_kind}{conf_s}.")
            if top_scene and not obj_names:
                lines.append("Visual guess: " + str(top_scene))
            if obj_names:
                lines.append("Detected objects: " + ", ".join(obj_names[:3]))
            if preview:
                lines.append("Text preview: " + preview)
        else:
            if obj_names:
                lines.append(f"Prima facie: this looks like a photo/image of {', '.join(obj_names[:3])}{conf_s}.")
            elif top_scene:
                lines.append(f"Prima facie: this looks like {str(top_scene)}{conf_s}.")
            elif preview:
                lines.append(f"Prima facie: I found readable text{conf_s}. Text preview: {preview}")
            else:
                lines.append(f"Prima facie: I couldn't confidently classify the upload{conf_s}.")

        j["summary"] = "\n".join([ln for ln in lines if ln])
    except Exception as e:
        result["logs"].append({"stage": "judgement-summary", "error": str(e)})

    # 5) Save JSON
    out_path = os.path.join(RESULT_DIR, f"{job_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path
