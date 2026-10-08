# numpy
import numpy as np

# typing
from typing import List, Optional

# abstract class
from abc import ABC, abstractmethod

# torch
import torch


class BaseModel(ABC):
    feature_dim: int

    @abstractmethod
    def get_image_features(self,
                           image: np.ndarray
                           ) -> torch.Tensor:
        """

        :param image: Input image as numpy array in RGB, [B C H W]
        :return: image_features as torch Tensor of shape [B F Hf Wf], where F is the features and Hf, Wf the model's
                output resolution. For pixel-level vision_models, Hf = H, Wf = W, for single feature vision_models Hf = Wf = 1.
        """
        pass

    @abstractmethod
    def get_text_features(self,
                          texts: List[str]
                          ) -> torch.Tensor:
        """
        Get text features from the model
        :param texts: List of strings
        :return: torch.Tensor of shape [B F], where F is the features
        """
        pass

    @abstractmethod
    def compute_similarity(self,
                           image_feats: torch.Tensor,
                           text_feats: torch.Tensor,
                           ) -> torch.Tensor:
        """
        Computes the similarity between image and text features
        :param image_feats: torch.Tensor of shape [B C Hf Wf], where coordinate system is in the top left corner of the image
        :param text_feats: torch.Tensor of shape [Bt C], where Bt is the text batch-dim (usually one)
        :return: torch.Tensor of shape [B Bt Hf Wf], similarity over image space, coordinate system starts in top left corner
        """
        pass
