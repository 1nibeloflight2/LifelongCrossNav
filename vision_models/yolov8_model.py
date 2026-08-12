import cv2
from ultralytics import YOLO
import numpy as np
from vision_models.coco_classes import COCO_CLASSES




class YoloV8Detector:
    def __init__(self,
                 confidence_threshold: float
                 ):
        self.model = YOLO("./weights/yolov8x.pt")
        self.confidence_threshold = confidence_threshold
        self.classes = None

    def set_classes(self,
                    classes: list
                    ):
        self.classes = classes
        # self.model.set_classes(classes)

    def detect(self,
               image: np.ndarray
               ):
        image = np.flip(image, axis=-1) # to bgr
        results = self.model(image, verbose=False)[0]
        preds = {}
        preds["boxes"] = []
        preds["scores"] = []
        boxes = results.boxes
        results.save("bed_test.jpg")
        for bbox, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            class_name = COCO_CLASSES[int(cls)]
            if class_name == self.classes[0] and conf > self.confidence_threshold:
                    # print(pred.class_name, pred.confidence)
                    preds["boxes"].append([bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()])
                    preds["scores"].append(conf.item())

        return preds


if __name__ == "__main__":
    from PIL import Image
    # Test the YOLO v8 Detector
    detector = YoloV8Detector(confidence_threshold=0.8)
    detector.set_classes(["bed"])
    # Load an image
    image = np.array(Image.open("/home/finn/active/MON/bed.jpeg"))

    # Detect objects in the image
    detections = detector.detect(image)


