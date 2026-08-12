from abc import ABC
from typing import List

import numpy as np

from .base_model import BaseModel

# lavis
from lavis.models import load_model_and_preprocess

# torch
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class BLIP2Model(BaseModel):
    def __init__(self) -> None:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
            name="blip2_image_text_matching",
            model_type="pretrain",
            is_eval=True,
            device=device,
        )
        self.device = device
        self.model = self.model.eval()
        self.image_atts = torch.ones((1, 257), dtype=torch.long).to(
            self.device
        )
        # self.set_batch_size(21)

    def set_batch_size(self, batch_size: int) -> None:
        self.image_atts = torch.ones((batch_size, 257), dtype=torch.long).to(
            self.device
        )

    @torch.inference_mode()
    def preprocess_image(self, image: np.ndarray | torch.Tensor
                         ) -> torch.Tensor:
        # img = image.squeeze()
        if image.shape[-1] == 3 or image.shape[-2] == 3:
            # throw error
            raise ValueError(
                "Image needs to be [B, C, H, W]. Got Channel dimension (3) in one of the last two dims.")

        transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                self.vis_processors["eval"].normalize,
            ]
        )
        image = torch.Tensor(image).to(self.device) / 255.0

        image = transform(image).to(self.device)
        return image

    @torch.inference_mode()
    def process_image(self, image: torch.Tensor) -> torch.Tensor:
        with self.model.maybe_autocast():
            image_embeds = self.model.ln_vision(self.model.visual_encoder(image))

        image_embeds = image_embeds.float()

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=self.image_atts,
            return_dict=True,
        )
        image_feats = F.normalize(
            self.model.vision_proj(query_output.last_hidden_state), dim=-1
        )
        return image_feats

    @torch.inference_mode()
    def get_image_features(self, image: np.ndarray) -> torch.Tensor:
        image = self.preprocess_image(image)
        image_feats = self.process_image(image)
        # TODO Need to adjust dimensions, this is probably NOT B F H W, but B Q F
        image_feats = image_feats.mean(dim=1)
        image_feats = image_feats.unsqueeze(-1).unsqueeze(-1)
        return image_feats

    @torch.inference_mode()
    def get_text_features(self, texts: List[str]) -> torch.Tensor:
        text = [self.text_processors["eval"](text) for text in texts]
        text = self.model.tokenizer(
            text,
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        text_output = self.model.Qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        return text_feat

    def compute_similarity(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        pass
