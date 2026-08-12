"""SegFormer-B2 ADE20K stair segmentation for Navigator3D."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class SegFormerStairDetector:
    """Expose ADE20K ``stairs``/``stairway`` masks through the detector API."""

    STAIR_LABELS = {"stairs", "stairway"}

    def __init__(
        self,
        model_path: str = "weights/segformer_b2_ade20k",
        pixel_threshold: float = 0.25,
        min_component_area: int = 80,
        max_components: int = 8,
        use_fp16: bool = True,
        device: str = "cuda",
    ) -> None:
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"SegFormer stair model not found: {path}. "
                "Expected a local Hugging Face save_pretrained directory."
            )
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pixel_threshold = float(pixel_threshold)
        self.min_component_area = max(1, int(min_component_area))
        self.max_components = max(1, int(max_components))
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")

        self.processor = SegformerImageProcessor.from_pretrained(
            str(path), local_files_only=True
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            str(path), local_files_only=True
        ).to(self.device).eval()
        if self.use_fp16:
            self.model.half()

        self.stair_class_ids = tuple(
            int(class_id)
            for class_id, label in self.model.config.id2label.items()
            if str(label).strip().lower() in self.STAIR_LABELS
        )
        if not self.stair_class_ids:
            raise RuntimeError(
                f"SegFormer checkpoint has no {sorted(self.STAIR_LABELS)} labels"
            )
        names = [self.model.config.id2label[index] for index in self.stair_class_ids]
        precision = "fp16" if self.use_fp16 else "fp32"
        print(
            f"[SegFormerStair] Loaded {path} on {self.device} ({precision}); "
            f"classes={list(zip(self.stair_class_ids, names))}, "
            f"pixel_threshold={self.pixel_threshold:.2f}, min_area={self.min_component_area}"
        )

    @staticmethod
    def _empty_result() -> dict:
        return {"boxes": [], "classes": [], "scores": [], "masks": []}

    def _semantic_maps(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
            raise ValueError(f"Expected RGB HxWx3 image, got {image_rgb.shape}")
        height, width = image_rgb.shape[:2]
        inputs = self.processor(
            images=np.ascontiguousarray(image_rgb[..., :3]),
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.use_fp16:
            pixel_values = pixel_values.half()
        with torch.inference_mode():
            logits = self.model(pixel_values=pixel_values).logits
            logits = F.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = torch.softmax(logits.float(), dim=1)
            confidence, predicted_class = probabilities.max(dim=1)
        return (
            predicted_class[0].cpu().numpy().astype(np.int32),
            confidence[0].cpu().numpy().astype(np.float32),
        )

    def detect(self, image_rgb: np.ndarray) -> dict:
        predicted_class, confidence = self._semantic_maps(image_rgb)
        stair_mask = np.isin(predicted_class, self.stair_class_ids)
        stair_mask &= confidence >= self.pixel_threshold
        stair_mask = stair_mask.astype(np.uint8)
        if not np.any(stair_mask):
            return self._empty_result()

        kernel = np.ones((3, 3), dtype=np.uint8)
        stair_mask = cv2.morphologyEx(stair_mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            stair_mask, connectivity=8
        )
        components = []
        for label_id in range(1, count):
            x, y, width, height, area = [int(value) for value in stats[label_id]]
            if area < self.min_component_area or width < 3 or height < 3:
                continue
            mask = labels == label_id
            score = float(np.quantile(confidence[mask], 0.90))
            components.append((score, area, [x, y, x + width, y + height], mask))

        components.sort(key=lambda item: (item[0], item[1]), reverse=True)
        result = self._empty_result()
        for score, _, box, mask in components[:self.max_components]:
            result["boxes"].append(box)
            result["classes"].append("stairs")
            result["scores"].append(score)
            result["masks"].append(mask)
        return result
