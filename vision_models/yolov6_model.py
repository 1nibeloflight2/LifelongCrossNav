import argparse

import torch
import cv2
import numpy as np
from yolov6.layers.common import DetectBackend
from yolov6.utils.nms import non_max_suppression
from yolov6.data.data_augment import letterbox
import sys

class YOLOV6Detector:
    def __init__(self, confidence_threshold):
        weights = 'weights/yolov6l6.pt'
        device = 'cuda'
        yaml_file = 'YOLOv6/data/coco.yaml'
        img_size = 640
        half = False
        iou_thres = 0.45
        sys.path.insert(0, "YOLOv6/")
        self.img_size = (img_size, img_size)
        self.conf_thres = confidence_threshold
        self.iou_thres = iou_thres
        self.half = half and device != 'cpu'

        # Initialize device
        self.device = torch.device(device)

        # Load model
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.img_size = self.check_img_size(self.img_size, s=self.stride)

        # Switch model to deploy status
        self.model_switch(self.model.model, self.img_size)

        if self.half:
            self.model.model.half()
        else:
            self.model.model.float()

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))

        # Load class names
        import yaml
        self.classes = None

        with open(yaml_file, 'r') as f:
            self.class_names = yaml.safe_load(f)['names']
        sys.path.pop(0)

    def set_classes(self,
                    classes: list
                    ):
        self.classes = classes

    def detect(self, img):
        # Preprocess image
        img = img.copy()
        img_processed, img_origin = self.preprocess_image(img)
        img_processed = img_processed.to(self.device)
        if len(img_processed.shape) == 3:
            img_processed = img_processed[None]

        # Inference
        pred_results = self.model(img_processed)

        # Apply NMS
        det = non_max_suppression(pred_results, self.conf_thres, self.iou_thres)[0]

        # Rescale boxes to original image
        det[:, :4] = self.rescale(img_processed.shape[2:], det[:, :4], img_origin.shape).round()

        # Clip boxes to image boundaries
        det[:, 0].clamp_(0, img_origin.shape[1] - 1)
        det[:, 1].clamp_(0, img_origin.shape[0] - 1)
        det[:, 2].clamp_(0, img_origin.shape[1] - 1)
        det[:, 3].clamp_(0, img_origin.shape[0] - 1)

        # Process results
        preds = {"boxes": [], "scores": []}
        for *xyxy, conf, cls in det:
            class_name = self.class_names[int(cls)]
            # if class_name == self.classes[0]:
            if conf > self.conf_thres:
                if not (xyxy[0] == xyxy[2] or xyxy[1] == xyxy[3]):  # Ensure box has area
                    preds["boxes"].append([coord.item() for coord in xyxy])
                    preds["scores"].append(conf.item())
                    print(f"Confidence: {conf.item()}")

        return preds

    def preprocess_image(self, img):
        img_origin = img.copy()
        img = letterbox(img, self.img_size, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img, img_origin

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    @staticmethod
    def check_img_size(img_size, s=32, floor=0):
        if isinstance(img_size, int):
            return max(int(img_size // s * s), floor)
        else:
            return [max(int(x // s * s), floor) for x in img_size]

    @staticmethod
    def model_switch(model, img_size):
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()


def main():
    parser = argparse.ArgumentParser(description="Object Detection on a single image")
    parser.add_argument('--weights', type=str, default='weights/yolov6l6.pt', help='Path to the model weights')
    parser.add_argument('--yaml', type=str, default='YOLOv6/data/coco.yaml', help='Path to the YAML file with class names')
    parser.add_argument('--image', type=str, default="/home/finn/active/MON/bed.jpeg", help='Path to the input image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (e.g. cuda:0 or cpu)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    args = parser.parse_args()

    # Initialize the detector
    detector = YOLOV6Detector(
        args.conf_thres
    )

    # Read the image
    image = cv2.imread(args.image)
    image = image[:, :, ::-1]  # Convert BGR to RGB
    image = np.ascontiguousarray(image)
    if image is None:
        print(f"Error: Unable to read image at {args.image}")
        return

    # Perform detection
    results = detector.detect(image)

    # Draw results on the image
    for box, score in zip(results['boxes'], results['scores']):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        label = f"Score: {score:.2f}"
        cv2.putText(image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display results
    cv2.imshow('Detections', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print results
    print(f"Detected {len(results['boxes'])} objects:")
    for box, score in zip(results['boxes'], results['scores']):
        print(f"  Box: {box}, Score: {score:.2f}")

if __name__ == "__main__":
    main()