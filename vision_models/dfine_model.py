"""ASCENT D-FINE-X inference with LCN's existing target-detector interface."""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from vision_models.coco_classes import COCO_CLASSES
from vision_models.dfine.core import YAMLConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "vision_models/dfine/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml"
DEFAULT_CHECKPOINT = "weights/dfine_x_obj2coco.pth"


class DFineDetector:
    """Closed COCO-80 detector; return only the selected HM3D target.

    Preserve ASCENT preprocessing and FP32 deploy inference. The local adapter
    replaces only the HTTP/ObjectDetections interface, not the model behavior.
    Output is pixel XYXY boxes, COCO class names, and unmodified confidences.
    """

    backend_name = "dfine"
    class_map = {
        "chair": "chair", "bed": "bed", "toilet": "toilet",
        "plant": "potted plant", "potted plant": "potted plant",
        "sofa": "couch", "couch": "couch", "tv": "tv", "tv_monitor": "tv",
    }

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        config_path: Union[str, Path] = DEFAULT_CONFIG,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
        device: Optional[str] = None,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("D-FINE confidence threshold must be in [0, 1]")
        self.confidence_threshold = float(confidence_threshold)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes = None
        self.classes_oi = {COCO_CLASSES.index(name) for name in self.class_map.values()}
        self.config_path = self._local_path(config_path)
        self.checkpoint_path = self._local_path(checkpoint_path)
        for path in (self.config_path, self.checkpoint_path):
            if not path.is_file():
                raise FileNotFoundError(f"D-FINE local asset not found: {path}")

        cfg = YAMLConfig(str(self.config_path))
        # The complete local detector checkpoint already contains its backbone.
        # Do not download or load an ImageNet backbone during evaluation.
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("ema", {}).get("module", checkpoint.get("model"))
        if state is None:
            raise ValueError("D-FINE checkpoint has neither ema.module nor model weights")
        model = cfg.model
        model.load_state_dict(state)
        self.model = model.deploy().float().to(self.device).eval()
        self.postprocessor = cfg.postprocessor.deploy().to(self.device).eval()
        self.transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        if self.device.type != "cpu":
            with torch.no_grad():
                self.model(torch.rand(1, 3, 640, 640, device=self.device))
        print(
            f"[D-FINE-X] target detector loaded: confidence>={self.confidence_threshold:.2f}, "
            f"device={self.device}, precision=FP32"
        )

    @staticmethod
    def _local_path(path: Union[str, Path]) -> Path:
        path = Path(path).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    def set_classes(self, classes: Sequence[str]) -> None:
        normalized = [self.class_map.get(name, name) for name in classes]
        unsupported = [name for name in normalized if name not in self.class_map.values()]
        if unsupported:
            raise ValueError(f"Unsupported closed-vocabulary D-FINE target: {unsupported}")
        self.classes = normalized

    def detect(self, image: np.ndarray) -> dict:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("D-FINE expects an H×W×3 uint8 RGB image")
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("D-FINE cannot detect an empty image")
        im_pil = Image.fromarray(image).convert("RGB")
        tensor = self.transform(im_pil).unsqueeze(0).to(self.device)
        orig_size = torch.tensor([[width, height]], device=self.device)
        with torch.inference_mode():
            outputs = self.model(tensor)
            labels, boxes, scores = self.postprocessor(outputs, orig_size)

        # Same two-stage confidence filtering as ASCENT: service >0.4, then
        # navigation >=0.8. Do not return other target categories to Navigator.
        labels, boxes, scores = labels[0], boxes[0], scores[0]
        keep = (scores > 0.4) & (scores >= self.confidence_threshold)
        labels = labels[keep].cpu().numpy()
        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        target_ids = (
            {COCO_CLASSES.index(name) for name in self.classes}
            if self.classes is not None else self.classes_oi
        )
        results = {"boxes": [], "scores": [], "classes": []}
        for label, box, score in zip(labels, boxes, scores):
            label = int(label)
            if label not in target_ids or not np.isfinite(box).all() or not np.isfinite(score):
                continue
            box = np.asarray(box).copy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            results["boxes"].append(box.tolist())
            results["scores"].append(float(score))
            results["classes"].append(COCO_CLASSES[label])
        return results
