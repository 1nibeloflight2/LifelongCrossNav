# typing
from typing import List, Dict

import numpy as np
# inference
from inference.models import YOLOWorld

# cv2
import cv2

# supervision
import supervision as sv


class YOLOWorldDetector:
    def __init__(self,
                 confidence_threshold: float
                 ):
        self.model = YOLOWorld(model_id="yolo_world/l")
        self.confidence_threshold = confidence_threshold
        self.classes = None

    def set_classes(self,
                    classes: List[str]
                    ):
        self.classes = classes
        self.model.set_classes(classes)

    def detect(self, image: np.ndarray) -> dict:
        if self.classes is None:
            raise ValueError("Classes must be set before detecting")

        results = self.model.infer(image, confidence=self.confidence_threshold)

        # DEBUG: print raw inference results
        n_raw = len(results.predictions)
        if n_raw > 0:
            raw_info = [(d.class_name, f"{d.confidence:.3f}") for d in results.predictions[:5]]
            print(f"[YOLOWorld] raw infer: {n_raw} predictions (top5: {raw_info})")
        else:
            print(f"[YOLOWorld] raw infer: 0 predictions (classes={self.classes}, conf_thresh={self.confidence_threshold})")

        preds = {
            "boxes": [],
            "classes": [],
            "scores": []
        }

        for detection in results.predictions:
            class_name = detection.class_name

            if detection.confidence > self.confidence_threshold:
                x1 = detection.x - detection.width / 2
                y1 = detection.y - detection.height / 2
                x2 = detection.x + detection.width / 2
                y2 = detection.y + detection.height / 2

                # Check if box is not a point
                if x1 != x2 and y1 != y2:
                    preds["boxes"].append([x1, y1, x2, y2])
                    preds["classes"].append(class_name)
                    preds["scores"].append(detection.confidence)

        return preds

if __name__ == "__main__":
    # Test the YOLO World Detector
    detector = YOLOWorldDetector(confidence_threshold=0.5)
    detector.set_classes(["person", "car", "truck", "bus", "bicycle", "motorbike", "traffic light", "stop sign"])

    # Load an image
    image = cv2.imread("test_images/a.jpg")

    # Detect objects in the image
    detections = detector.detect(image)

    # Display the detections
    print(f"Detected {len(detections['boxes'])} objects: {detections['classes']}")
