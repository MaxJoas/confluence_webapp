import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# Type aliases
ImageType = np.ndarray
Device = torch.device
ModelOutput = Tuple[ImageType, float]


class SAMModel(nn.Module):
    """Segment Anything Model wrapper for fine-tuning."""

    def __init__(
        self,
        freeze_encoder: bool = True,
        freeze_decoder: bool = False,
        freeze_prompt_encoder: bool = True,
        image_size: int = 1024,
    ):
        """
        Args:
            freeze_encoder (bool): Whether to freeze encoder weights
            freeze_decoder (bool): Whether to freeze decoder weights
            freeze_prompt_encoder (bool): Whether to freeze prompt encoder weights
            image_size (int): Input image size
        """
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_prompt_encoder = freeze_prompt_encoder
        self.transform = ResizeLongestSide(image_size)
        self.model = None

    def setup(self) -> None:
        """Initialize and configure the model."""
        if not os.path.exists(os.path.join("models", "sam_vit_h_4b8939.pth")):
            raise FileNotFoundError(
                "SAM model weights not found. Please download the weights from the link in the README."
            )
        self.model = sam_model_registry["vit_h"](os.path.join("models", "sam_vit_h_4b8939.pth"))

        if self.freeze_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.freeze_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): Input images

        Returns:
            masks (torch.Tensor): Predicted masks
            iou_predictions (torch.Tensor): IoU predictions
        """
        _, _, H, W = images.shape
        image_embeddings = self.model.image_encoder(images)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = F.interpolate(
            low_res_masks, (H, W), mode="bilinear", align_corners=False
        )
        return masks, iou_predictions
