from typing import List

import numpy as np
import torch

from .base_model import BaseModel


class OVSegModel(BaseModel):
    def get_image_features(self, image: np.ndarray) -> torch.Tensor:
        pass

    def get_text_features(self, texts: List[str]) -> torch.Tensor:
        pass

    def compute_similarity(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        pass