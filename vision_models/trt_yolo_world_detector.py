# YOLO World Detector using TensorRT for inference on Jetson. Exportable from ultralytics, see
# https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/export.md
# typing
from typing import List

import numpy as np
# inference
from ultralytics import YOLOWorld, YOLO

# cv2
import cv2

# supervision
import supervision as sv

class TRTYOLOWorldDetector:
    def __init__(self,
                 confidence_threshold: float
                 ):
        self.model = YOLO("yolov8s-worldv2.engine")
        self.confidence_threshold = confidence_threshold
        self.classes =[]

    def set_classes(self,
                    classes: list
                    ):
        self.classes = classes

    def detect(self,
               image: np.ndarray
               ):
        results = self.model(image, verbose=False)
        # results[0].show()
        pred = results[0].boxes
        # print("results")
        # print(results)
        preds = {}
        preds["boxes"] = []
        preds["scores"] = []
        #### Modify for correct format
        for i in range(len(pred)):
            cls = pred.cls[i].item()
            class_name = results[0].names[cls]
            if class_name == self.classes[0]:
                if pred.conf[i] > self.confidence_threshold:
                    box = pred.xyxy[i]
                    if not (box[0].item() == box[2].item() or box[1].item() == box[3].item()):
                        preds["boxes"].append([box[0].item(), box[1].item(), box[2].item(), box[3].item()])
                        preds["scores"].append(pred.conf[i].item())
        return preds

if __name__ == "__main__":
    # Test the YOLO World Detector
    detector = TRTYOLOWorldDetector(confidence_threshold=0.5)
    detector.set_classes(["person"])

    # Load an image
    image = cv2.imread("/home/spot/Pictures/bottles.png")

    # Detect objects in the image
    detections = detector.detect(image)
    print(detections)

    # Display the image with the detections
    # image_with_detections = detections.draw_on_image(image)
    # cv2.imshow("Detections", image_with_detections)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
