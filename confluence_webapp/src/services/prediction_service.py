import logging
from typing import Any, Tuple

import numpy as np
import torch
import cv2
from detectron2.engine import DefaultPredictor
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from models.unet_model import UNet
from models.sam import SAMModel
from utils.unet_utils import BasicDataset
from utils.utils import get_confluence
from utils.detectron_utils import visualize_detectron_results

# Type aliases
ImageType = np.ndarray
Device = torch.device
ModelOutput = Tuple[ImageType, float]


class PredictionManager:
    """Manages model predictions and confluence calculations."""

    def __init__(self, device: Device):
        """
        Args:
            device (Device): Torch device for computation
        """
        self.device = device
        self.logger = logging.getLogger(__name__)

    def calc_confluence(self, mask: np.ndarray) -> float:
        """
        Calculate confluence percentage from mask.

        Args:
            mask (np.ndarray): Binary mask

        Returns:
            float: Confluence percentage
        """
        total_pixels = mask.shape[0] * mask.shape[1]
        segmented_pixels = (mask == 0).sum()
        return segmented_pixels / total_pixels

    def predict_unet(self, 
                    net: UNet, 
                    image: Image.Image, 
                    scale_factor: float = 0.35, 
                    threshold: float = 0.1) -> np.ndarray:
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
        img = torch.from_numpy(BasicDataset.preprocess(image, scale_factor, is_mask=False))
        img = img.unsqueeze(0).to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = net(img)
            probs = torch.sigmoid(output)[0] if net.n_classes == 1 else F.softmax(output, dim=1)[0]

            tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image.size[1], image.size[0])),
                transforms.ToTensor()
            ])

            full_mask = tf(probs.cpu()).squeeze()

        return (full_mask > threshold).numpy()

    def predict_detectron(
        self, cfg: Any, image_path: str, weights_path: str
    ) -> ModelOutput:
        """
        Make prediction using Detectron2 model.

        Args:
            cfg (Any): Detectron2 configuration
            image_path (str): Path to input image
            weights_path (str): Path to model weights

        Returns:
            img_out (ImageType): Output image
            confluence (float): Confluence percentage
        """
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
        predictor = DefaultPredictor(cfg)

        img = np.array(Image.open(image_path))
        try:
            pred = predictor(img)
        except Exception:
            return img, -9999

        d = {
            "file_name": image_path,
            "height": img.shape[1],
            "width": img.shape[0],
            "file": image_path,
        }
        confluence, anns = get_confluence(d, pred)
        img_out = visualize_detectron_results(data_dict=d, anns=anns)
        return img_out, confluence

    def predict_sam(self, image: torch.Tensor, sam_path: str, original_size: Tuple) -> float:
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
        model.eval()
        model.load_state_dict(torch.load(sam_path, map_location=self.device))

 
        model.eval()
        pred, _ = model(image)
        # output prediction to console in Streamlit
        self.logger.info(f"Prediction: {pred}")
        

        # Process results
        mask_numpy = pred.cpu().detach().numpy().squeeze()
        mask_resized = cv2.resize(
            mask_numpy, (original_size[0], original_size[1])
        )
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        self.logger.info(f"Mask binary: {mask_binary}")

        # Calculate confluence
        confluence = self.calc_confluence(mask_binary)
 
        # Placeholder for SAM model prediction
        return mask_binary, confluence
