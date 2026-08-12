import os
import sys

import cv2
import numpy as np
import torch
from vision_models.utils.yolov7_utils import letterbox, non_max_suppression, scale_coords, check_img_size
from vision_models.coco_classes import COCO_CLASSES
# a = sys.path.pop(0)
sys.path.insert(0, "yolov7/")
#try:
from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import (  # noqa: E402
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from utils.torch_utils import TracedModel  # noqa: E402
#except Exception:
#    print("Could not import yolov7. This is OK if you are only using the client.")
sys.path.pop(0)
# sys.path.insert(0, a)
class YOLOv7Detector:
    def __init__(self,
                 confidence_threshold: float
                 ):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.half_precision = self.device.type != "cpu"
        # PyTorch 2.6+ defaults torch.load() to weights_only=True, while this
        # trusted local legacy YOLOv7 checkpoint stores the model class. Scope
        # the compatibility override to this one load and restore the caller's
        # environment immediately afterwards.
        force_no_weights_only = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        previous_force_setting = os.environ.get(force_no_weights_only)
        os.environ[force_no_weights_only] = "1"
        try:
            self.model = attempt_load(
                'weights/yolov7-e6e.pt', map_location=self.device
            )  # load FP32 model
        finally:
            if previous_force_setting is None:
                os.environ.pop(force_no_weights_only, None)
            else:
                os.environ[force_no_weights_only] = previous_force_setting
        stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(640, s=stride)  # check img_size
        self.model = TracedModel(self.model, self.device, self.image_size)
        if self.half_precision:
            self.model.half()  # to
        self.confidence_threshold = confidence_threshold
        self.classes = None
        self.class_map = {
            "chair": "chair",
            "tv": "tv",
            "tv_monitor": "tv",
            "potted plant": "potted plant",
            "plant": "potted plant",
            "couch": "couch",
            "sofa": "couch",
            "bed": "bed",
            "toilet": "toilet",
        }
        classes_oi = ["chair", "tv", "potted plant", "bed", "toilet", "couch"]
        self.classes_oi = [COCO_CLASSES.index(c) for c in classes_oi]
        # self.classes_oi = None

    def predict(self, image):
        results = self.model(image)
        return results

    def set_classes(self,
                    classes: list
                    ):
        self.classes = [self.class_map.get(c, c) for c in classes]

    def detect(self,
               image: np.ndarray
               ):
        orig_shape = image.shape

        # Preserve the sensor image aspect ratio.  The previous fixed
        # 640x448 resize distorted square Habitat observations before padding,
        # which made object confidence unstable (notably for toilets/beds).
        # YOLOv7's letterbox performs the required scale + padding itself.
        img = image
        img = letterbox(img, new_shape=self.image_size)[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)

        img = img.half() if self.half_precision else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.inference_mode():  # Calculating gradients causes a GPU memory leak
            pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            0.25,
            0.45,
            classes=self.classes_oi,
            agnostic=False,
        )[0]
        preds = {"boxes": [], "scores": [], "classes": []}
        if pred is None or pred.shape[0] == 0:
            return preds
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], orig_shape).round()
        pred[:, 0] = torch.clip(pred[:, 0], 0, orig_shape[1] - 1)
        pred[:, 1] = torch.clip(pred[:, 1], 0, orig_shape[0] - 1)
        pred[:, 2] = torch.clip(pred[:, 2], 0, orig_shape[1] - 1)
        pred[:, 3] = torch.clip(pred[:, 3], 0, orig_shape[0] - 1)
        boxes = pred[:, :4]
        logits = pred[:, 4]
        target_class = self.classes[0] if self.classes else None
        for i in range(pred.shape[0]):
            class_name = COCO_CLASSES[int(pred[i, 5])]
            if target_class is not None and class_name != target_class:
                continue
            score = float(logits[i].detach().cpu().item())
            if score <= self.confidence_threshold:
                continue
            box = boxes[i]
            if box[0].item() == box[2].item() or box[1].item() == box[3].item():
                continue
            preds["boxes"].append([box[0].item(), box[1].item(), box[2].item(), box[3].item()])
            preds["scores"].append(score)
            preds["classes"].append(class_name)

        return preds

if __name__ == "__main__":
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # clip = ClipModel('weights/clip.pth')
    detector = YOLOv7Detector(0.5)
    detector.set_classes(["couch"])
    image = cv2.imread("/home/spot/Pictures/sofa.jpeg")
    image = cv2.resize(image, (640, 640))
    # print(image.shape)
    preds = detector.detect(image)
    start.record()
    preds = detector.detect(image)
    end.record()
    torch.cuda.synchronize()
    print("Complete forward: ", start.elapsed_time(end) / 1000)
    print(preds)
    print("Done!")
