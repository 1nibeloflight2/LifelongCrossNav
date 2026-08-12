from typing import List

import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class GroundingDinoDetector:
    def __init__(self,
                 confidence_threshold: float
                 ):
        model_id = "IDEA-Research/grounding-dino-base"
        self.confidence_threshold = confidence_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")

    def set_classes(self,
                    classes: List[str]
                    ):
        self.classes = classes

    def detect(self,
               image: Image
               ):
        inputs = self.processor(images=image, text=self.classes[0], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.confidence_threshold,
            text_threshold=self.confidence_threshold,
            target_sizes=[image.shape[::-1]]
        )[0] # (top_left_x, top_left_y, bottom_right_x, bottom_right_y),
        res = {}
        res["boxes"] = []
        res["scores"] = []
        for box, score in zip(results['boxes'], results['scores']):
            res["boxes"].append([box[0].item(), box[1].item(), box[2].item(), box[3].item()])
            res["scores"].append(score)
        return res