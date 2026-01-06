import os
import logging

logger = logging.getLogger("models_loader")

class ModelRegistry:
    def __init__(self):
        self._yolo = None
        self._sam = None
        self._clip = None

    def get_yolo(self):
        if self._yolo is None:
            try:
                from ultralytics import YOLO
                # Tiny model auto-downloads weights on first run
                self._yolo = YOLO("yolov8n.pt")
                logger.info("Loaded YOLOv8n")
            except Exception as e:
                logger.exception("YOLO load failed: %s", e)
                self._yolo = None
        return self._yolo

    def get_sam(self):
        if self._sam is None:
            try:
                weights_path = os.environ.get("SAM_WEIGHT_PATH")
                if not weights_path or not os.path.exists(weights_path):
                    logger.warning("SAM weights missing; set SAM_WEIGHT_PATH to enable segmentation")
                    self._sam = None
                else:
                    from segment_anything import sam_model_registry, SamPredictor
                    model_type = os.environ.get("SAM_MODEL_TYPE", "vit_b")
                    sam = sam_model_registry[model_type](checkpoint=weights_path)
                    self._sam = SamPredictor(sam)
                    logger.info("Loaded SAM predictor (%s)", model_type)
            except Exception as e:
                logger.exception("SAM load failed: %s", e)
                self._sam = None
        return self._sam

    def get_clip(self):
        if self._clip is None:
            try:
                import clip  # type: ignore
                import torch
                model_name = os.environ.get("CLIP_MODEL", "ViT-B/32")
                model, preprocess = clip.load(model_name, device="cpu")
                self._clip = (model, preprocess)
                logger.info("Loaded CLIP %s", model_name)
            except Exception as e:
                logger.exception("CLIP load failed: %s", e)
                self._clip = None
        return self._clip
