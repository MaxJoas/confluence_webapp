import logging
import os
import random
from typing import Any, Tuple

import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from utils.detectron_utils import (
    get_confluence,
    resize_image,
    visualize_result,
)
from utils.unet_utils import BasicDataset

from models.sam import SAMModel
from models.unet_model import UNet

# Type aliases
ImageType = np.ndarray
Device = torch.device
ModelOutput = Tuple[ImageType, float]
CellposeModel = Any


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class PredictionManager:
    """Manages model predictions and confluence calculations."""

    def __init__(self, device: Device):
        """
        Args:
            device (Device): Torch device for computation
        """
        self.device = device
        self.logger = logging.getLogger(__name__)

    def calc_confluence(self, mask: np.ndarray, mask_class: int = 0) -> float:
        """
        Calculate confluence percentage from mask.

        Args:
            mask (np.ndarray): Binary mask

        Returns:
            float: Confluence percentage
        """
        total_pixels = mask.shape[0] * mask.shape[1]
        segmented_pixels = (mask == mask_class).sum()
        return segmented_pixels / total_pixels

    def predict_unet(
        self,
        net: UNet,
        image: Image.Image,
        scale_factor: float = 0.35,
        threshold: float = 0.1,
    ) -> np.ndarray:
        """
        Make prediction using UNet model.

        Args:
            net (UNet): UNet model
            image (Image.Image): Input image
            scale_factor (float): Image scaling factor
            threshold (float): Prediction threshold

        Returns:
            np.ndarray: Predicted mask
        """
        net.eval()
        img = torch.from_numpy(
            BasicDataset.preprocess(image, scale_factor, is_mask=False)
        )
        img = img.unsqueeze(0).to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = net(img)
            probs = (
                torch.sigmoid(output)[0]
                if net.n_classes == 1
                else F.softmax(output, dim=1)[0]
            )

            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((image.size[1], image.size[0])),
                    transforms.ToTensor(),
                ]
            )

            full_mask = tf(probs.cpu()).squeeze()

        return (full_mask > threshold).numpy(), full_mask

    def predict_detectron(
        self,
        cfg: Any,
        image_path: str,
        weights_path: str,
    ) -> ModelOutput:
        """
        Make prediction using Detectron2 model with enhanced visualization.

        Args:
            cfg (Any): Detectron2 configuration
            image_path (str): Path to input image
            weights_path (str): Path to model weights
            confidence_threshold (float, optional): Minimum prediction confidence. Defaults to 0.5.
            iou_threshold (float, optional): IoU threshold for mask suppression. Defaults to 0.3.

        Returns:
            img_out (ImageType): Output image
            confluence (float): Confluence percentage
        """
        cfg.MODEL.WEIGHTS = weights_path
        predictor = DefaultPredictor(cfg)
        # Check inference input size
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST

        img = np.array(Image.open(image_path))
        seed_everything(100)

        try:
            pred = predictor(img)
            # with open("detectron_prediction.pkl", "wb") as f:
            # pickle.dump(pred, f)
        except Exception as e:
            print(f"Error in Detectron2 prediction, trying to resize image: {e}")
            try:
                print(img.shape)
                img = resize_image(image=img, min_size=min_size, max_size=max_size)
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=2)
                print("after resize")
                print(img.shape)
                pred = predictor(img)

            except Exception as e:

                raise ValueError(f"Error in Detectron2 prediction: {e}")
            pass

        d = {
            "file_name": image_path,
            "height": img.shape[0],
            "width": img.shape[1],
            "file": image_path,
        }
        confluence, anns = get_confluence(d, pred)
        print(f"shape of sample mask: {pred['instances'].pred_masks[0].shape}")
        print(f"shape of image: {img.shape}")
        img_out = visualize_result(out=pred, img=img)
        return img_out, confluence

    def predict_sam(
        self, image: torch.Tensor, sam_path: str, original_size: Tuple
    ) -> float:
        """
        Make prediction using SAM model.

        Args:
            image (torch.Tensor): Input image
            sam_path (str): Path to SAM model weights
            original_size (tuple): Original image size

        Returns:
            confluence (float): Confluence percentage
            mask_resize (np.ndarray): Resized mask
        """
        # Initialize SAM model
        model = SAMModel()
        model.setup()
        model.to(self.device)
        model.load_state_dict(torch.load(sam_path, map_location=self.device))
        # get image dimensions that the model has been trained on

        model.eval()
        with torch.no_grad():
            pred, _ = model(image.to(self.device))
        # torch.save(pred, "sam_prediction.pt")
        pred = pred.squeeze().detach().cpu().numpy()
        pred_normalized = pred - pred.min()  # Shift to start at 0
        pred_normalized /= pred_normalized.max()  # Scale to [0, 1]
        mask_binary = pred_normalized > 0.5

        pred_normalized = (pred_normalized * 255).astype(np.uint8)

        # Calculate confluence
        confluence = self.calc_confluence(mask_binary)

        return mask_binary, confluence, pred_normalized

    def predict_cellpose(
        self,
        image: ImageType,
        model: CellposeModel,
        flow_threshold: int,
        cellprob_threshold: int,
    ) -> ModelOutput:
        """
        Make prediction using Cellpose model.

        Args:
            image (ImageType): Input image

        Returns:
            img_out (ImageType): Output image
            confluence (float): Confluence percentage
        """
        # Placeholder for Cellpose model prediction
        masks, flows, *_ = model.eval(
            [image],
            channels=[0, 0],
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        probs = flows[0][2]
        # make display image
        probs_sig = torch.sigmoid(torch.from_numpy(probs))

        prediction_mask = masks[0]
        prediction_mask = prediction_mask.astype(np.uint8)
        # make binary mask
        prediction_mask = (prediction_mask > 0).astype(np.uint8)

        return prediction_mask, probs_sig
