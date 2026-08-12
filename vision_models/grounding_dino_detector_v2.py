"""
Grounding DINO detector using native groundingdino package + local weights.
Supports single-prompt detection to avoid multi-class confidence dilution.

Usage:
    detector = GroundingDinoDetectorV2(confidence_threshold=0.3)
    
    # Target object detection
    target_dets = detector.detect(image_rgb, prompt="chair")
    
    # Stair detection (independent call, no confidence dilution)
    stair_dets = detector.detect(image_rgb, prompt="stairs")
"""
import numpy as np
import torch
import torchvision.transforms.functional as F
from pathlib import Path
from typing import List, Optional

from groundingdino.util.inference import load_model, predict


class GroundingDinoDetectorV2:
    """
    Single-model detector that supports per-call prompt switching.
    Call detect(image, prompt) with different prompts for different tasks.
    
    Uses native groundingdino package with local SwinT weights.
    Output format matches YOLOWorldDetector: {"boxes": [...], "classes": [...], "scores": [...]}
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        weights_path: str = "weights/groundingdino_swint_ogc.pth",
    ):
        self.confidence_threshold = confidence_threshold
        config_path = self._find_config()
        self.model = load_model(
            model_config_path=config_path,
            model_checkpoint_path=weights_path,
        ).to("cuda")
        self.model.eval()
        print(f"[GDinoV2] Loaded: config={config_path}, weights={weights_path}, "
              f"thresh={confidence_threshold}")

        # Compatibility: store classes for set_classes interface
        self.classes: Optional[List[str]] = None

    def _find_config(self) -> str:
        """Auto-detect GroundingDINO config from installed package."""
        import groundingdino
        cfg = Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"
        if cfg.is_file():
            return str(cfg)
        raise FileNotFoundError(
            f"Cannot find GroundingDINO config at {cfg}. "
            "Ensure groundingdino package is properly installed."
        )

    def set_classes(self, classes: List[str]) -> None:
        """Compatibility interface. Stores classes for default detect() calls."""
        self.classes = classes

    def detect(self, image: np.ndarray, prompt: Optional[str] = None) -> dict:
        """
        Detect objects with a single text prompt.

        Args:
            image: (H, W, 3) RGB uint8 numpy array
            prompt: text prompt (e.g. "chair" or "stairs").
                    If None, uses self.classes[0].

        Returns:
            dict: {"boxes": [[x1,y1,x2,y2], ...], "classes": [str, ...], "scores": [float, ...]}
        """
        if prompt is None:
            if self.classes is None or len(self.classes) == 0:
                return {"boxes": [], "classes": [], "scores": []}
            prompt = self.classes[0]

        H, W = image.shape[:2]
        image_tensor = F.to_tensor(image)
        image_transformed = F.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        with torch.inference_mode():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=prompt,
                box_threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
            )

        result = {"boxes": [], "classes": [], "scores": []}
        for box, logit, phrase in zip(boxes, logits, phrases):
            box_np = box.cpu().numpy()
            # predict() returns cxcywh normalized -> convert to xyxy pixel
            x1 = int((box_np[0] - box_np[2] / 2) * W)
            y1 = int((box_np[1] - box_np[3] / 2) * H)
            x2 = int((box_np[0] + box_np[2] / 2) * W)
            y2 = int((box_np[1] + box_np[3] / 2) * H)

            # Filter degenerate boxes
            if x2 > x1 and y2 > y1:
                result["boxes"].append([x1, y1, x2, y2])
                result["classes"].append(phrase)
                result["scores"].append(float(logit))

        return result