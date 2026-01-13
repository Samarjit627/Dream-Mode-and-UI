"""
CV-Based Geometry Extraction using Skeletonization + Contour Tracing

This module traces the ACTUAL drawn strokes in automotive sketches by:
1. Converting to binary
2. Skeletonizing to get stroke center lines
3. Tracing contours along the skeleton
4. Fitting splines to get smooth curves
5. Outputting pixel-accurate geometry

NO LLM IS USED FOR GEOMETRY EXTRACTION.
"""
import cv2
import numpy as np
import base64
from typing import Dict, List, Tuple, Optional
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from scipy.interpolate import splprep, splev


def to_native(val):
    """Convert numpy types to native Python types for JSON serialization."""
    if hasattr(val, 'item'):
        return val.item()
    elif isinstance(val, (list, tuple)):
        return [to_native(v) for v in val]
    elif isinstance(val, dict):
        return {k: to_native(v) for k, v in val.items()}
    return val


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode base64 image to OpenCV format."""
    if "base64," in image_base64:
        image_data = image_base64.split("base64,")[1]
    else:
        image_data = image_base64
    
    img_bytes = base64.b64decode(image_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def preprocess_sketch(img: np.ndarray) -> np.ndarray:
    """
    Preprocess sketch for stroke extraction.
    Returns a binary image where strokes are white (255) on black (0).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert if sketch is dark on light background
    if np.mean(gray) > 127:
        gray = 255 - gray
    
    # Adaptive thresholding to handle varying line weights
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, -2
    )
    
    # Clean up noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary


def extract_skeleton(binary: np.ndarray) -> np.ndarray:
    """
    Skeletonize the binary image to get stroke center lines.
    This ensures we trace the CENTER of each stroke.
    """
    # Convert to boolean for skimage
    binary_bool = binary > 127
    
    # Skeletonize
    skeleton = skeletonize(binary_bool)
    
    # Convert back to uint8
    skeleton_uint8 = img_as_ubyte(skeleton)
    
    return skeleton_uint8


def trace_contours(skeleton: np.ndarray) -> List[np.ndarray]:
    """
    Trace contours from the skeletonized image.
    Returns list of contours, each as an array of (x, y) points.
    """
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Filter out tiny contours (noise)
    min_length = 50
    contours = [c for c in contours if cv2.arcLength(c, False) > min_length]
    
    # Sort by arc length (longest first)
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
    
    return contours


def order_contour_points(contour: np.ndarray) -> List[Tuple[int, int]]:
    """
    Order contour points from left to right (for horizontal lines like roofline).
    """
    points = contour.reshape(-1, 2)
    
    # Sort by x coordinate
    sorted_indices = np.argsort(points[:, 0])
    ordered = points[sorted_indices]
    
    return [(int(p[0]), int(p[1])) for p in ordered]


def fit_spline(points: List[Tuple[int, int]], n_samples: int = 4) -> List[Tuple[int, int]]:
    """
    Fit a smooth spline through the points and sample n_samples points.
    This gives us clean Bézier-compatible control points.
    """
    if len(points) < 4:
        # Not enough points for spline, return evenly spaced points
        indices = np.linspace(0, len(points) - 1, n_samples, dtype=int)
        return [points[i] for i in indices]
    
    # Extract x and y
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    # Remove duplicates
    _, unique_idx = np.unique(x, return_index=True)
    x = x[unique_idx]
    y = y[unique_idx]
    
    if len(x) < 4:
        indices = np.linspace(0, len(x) - 1, min(n_samples, len(x)), dtype=int)
        return [(int(x[i]), int(y[i])) for i in indices]
    
    try:
        # Fit spline
        tck, u = splprep([x, y], s=len(x) * 10, k=min(3, len(x) - 1))
        
        # Sample n_samples points
        u_new = np.linspace(0, 1, n_samples)
        x_new, y_new = splev(u_new, tck)
        
        return [(int(xi), int(yi)) for xi, yi in zip(x_new, y_new)]
    except Exception as e:
        print(f"Spline fitting failed: {e}")
        indices = np.linspace(0, len(x) - 1, min(n_samples, len(x)), dtype=int)
        return [(int(x[i]), int(y[i])) for i in indices]


def identify_roofline(contours: List[np.ndarray], width: int, height: int) -> Optional[List[Tuple[int, int]]]:
    """
    Identify the roofline from contours.
    The roofline is typically:
    - In the upper half of the image
    - Roughly horizontal (spans significant x range)
    - The HIGHEST significant contour
    """
    upper_half_contours = []
    
    for contour in contours:
        points = contour.reshape(-1, 2)
        
        # Check if mostly in upper half
        mean_y = np.mean(points[:, 1])
        if mean_y > height * 0.5:
            continue  # Skip lower half contours
        
        # Check if spans significant x range (at least 30% of width)
        x_span = np.max(points[:, 0]) - np.min(points[:, 0])
        if x_span < width * 0.3:
            continue
        
        # Score by how high (low y) and how long
        score = -mean_y + x_span / 10  # Higher = better
        upper_half_contours.append((contour, score))
    
    if not upper_half_contours:
        return None
    
    # Sort by score (highest first)
    upper_half_contours.sort(key=lambda x: x[1], reverse=True)
    best_contour = upper_half_contours[0][0]
    
    # Order points left to right
    ordered = order_contour_points(best_contour)
    
    # Fit spline and sample 4 points (for Bézier)
    sampled = fit_spline(ordered, n_samples=4)
    
    return sampled


def identify_beltline(contours: List[np.ndarray], width: int, height: int) -> Optional[List[Tuple[int, int]]]:
    """
    Identify the beltline from contours.
    The beltline is typically:
    - In the middle third of the image
    - Roughly horizontal
    - Below the roofline
    """
    middle_contours = []
    
    for contour in contours:
        points = contour.reshape(-1, 2)
        
        # Check if in middle region
        mean_y = np.mean(points[:, 1])
        if mean_y < height * 0.3 or mean_y > height * 0.65:
            continue
        
        # Check if spans significant x range
        x_span = np.max(points[:, 0]) - np.min(points[:, 0])
        if x_span < width * 0.3:
            continue
        
        # Score by length
        score = x_span
        middle_contours.append((contour, score))
    
    if not middle_contours:
        return None
    
    # Get the longest
    middle_contours.sort(key=lambda x: x[1], reverse=True)
    best_contour = middle_contours[0][0]
    
    # Order and sample 3 points
    ordered = order_contour_points(best_contour)
    sampled = fit_spline(ordered, n_samples=3)
    
    return sampled


def detect_wheels_circle(img: np.ndarray, width: int, height: int) -> List[Dict]:
    """
    Detect wheels using Hough Circle Transform.
    Returns list of {cx, cy, r} in pixel coordinates.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=width // 4,
        param1=100,
        param2=30,
        minRadius=int(width * 0.04),
        maxRadius=int(width * 0.12)
    )
    
    wheels = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :2]:
            wheels.append({
                "cx_px": int(c[0]),
                "cy_px": int(c[1]),
                "r_px": int(c[2]),
                "cx": float(c[0]) / width,
                "cy": float(c[1]) / height,
                "r": float(c[2]) / width
            })
    
    # Sort by x (front wheel first)
    wheels.sort(key=lambda w: w["cx_px"])
    
    return wheels


def extract_primitives_cv(image_base64: str, image_width: int, image_height: int) -> Dict:
    """
    MAIN ENTRY POINT: Extract pixel-accurate geometry using CV.
    
    Uses skeletonization + contour tracing to trace ACTUAL drawn strokes.
    
    NO LLM IS USED HERE.
    """
    # Decode image
    img = decode_base64_image(image_base64)
    
    if img is None:
        return {"error": "Failed to decode image", "primitives": {}}
    
    height, width = img.shape[:2]
    
    print("=" * 60)
    print("CV GEOMETRY EXTRACTION (SKELETONIZATION + CONTOUR TRACING)")
    print(f"  Image size: {width}x{height}")
    
    # Step 1: Preprocess
    binary = preprocess_sketch(img)
    print("  Step 1: Preprocessed sketch to binary")
    
    # Step 2: Skeletonize
    skeleton = extract_skeleton(binary)
    print("  Step 2: Skeletonized to get stroke centers")
    
    # Step 3: Trace contours
    contours = trace_contours(skeleton)
    print(f"  Step 3: Found {len(contours)} significant contours")
    
    primitives = {}
    
    # Step 4: Identify roofline
    roofline = identify_roofline(contours, width, height)
    if roofline:
        primitives["roofline"] = {
            "pixel_points": roofline,
            "points": [[p[0] / width, p[1] / height] for p in roofline],
            "parametric_order": "front_to_rear"
        }
        print(f"  Roofline: {len(roofline)} points - {roofline}")
    
    # Step 5: Identify beltline
    beltline = identify_beltline(contours, width, height)
    if beltline:
        primitives["beltline"] = {
            "pixel_points": beltline,
            "points": [[p[0] / width, p[1] / height] for p in beltline],
            "parametric_order": "front_to_rear"
        }
        print(f"  Beltline: {len(beltline)} points - {beltline}")
    
    # Step 6: Detect wheels
    wheels = detect_wheels_circle(img, width, height)
    if len(wheels) >= 1:
        primitives["front_wheel"] = wheels[0]
        print(f"  Front wheel: ({wheels[0]['cx']:.3f}, {wheels[0]['cy']:.3f}), r={wheels[0]['r']:.3f}")
    if len(wheels) >= 2:
        primitives["rear_wheel"] = wheels[1]
        print(f"  Rear wheel: ({wheels[1]['cx']:.3f}, {wheels[1]['cy']:.3f}), r={wheels[1]['r']:.3f}")
    
    print(f"  Detected primitives: {list(primitives.keys())}")
    print("=" * 60)
    
    if not primitives:
        return {"error": "No primitives detected", "primitives": {}}
    
    return to_native({
        "vehicle_type": "detected",
        "view": "side",
        "image_width": width,
        "image_height": height,
        "primitives": primitives
    })
