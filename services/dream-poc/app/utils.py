from PIL import Image
import io
import numpy as np
from typing import Optional, Union

try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
except Exception:
    # HEIC/HEIF support is optional
    pass


def read_image_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _hex_from_rgb(rgb: np.ndarray) -> str:
    r, g, b = [int(x) for x in rgb.tolist()]
    return f"#{r:02x}{g:02x}{b:02x}"


def dominant_color_hex(image: Image.Image, mask: Optional[Union[Image.Image, np.ndarray]]) -> Optional[str]:
    """
    Compute an approximate dominant color inside mask using median in RGB space.
    Avoids heavy deps (no sklearn). Returns hex string like #rrggbb or None.
    """
    try:
        im_np = np.array(image)  # HxWx3
        if mask is None:
            pixels = im_np.reshape(-1, 3)
        else:
            if isinstance(mask, Image.Image):
                mask_np = np.array(mask.convert("L")) > 0
            else:
                mask_np = mask.astype(bool)
            if im_np.shape[:2] != mask_np.shape[:2]:
                # best-effort resize mask to image size
                mask_img = Image.fromarray((mask_np.astype("uint8") * 255))
                mask_img = mask_img.resize((im_np.shape[1], im_np.shape[0]))
                mask_np = np.array(mask_img) > 0
            pixels = im_np[mask_np]
            if pixels.size == 0:
                pixels = im_np.reshape(-1, 3)
        med = np.median(pixels, axis=0)
        return _hex_from_rgb(med)
    except Exception:
        return None
