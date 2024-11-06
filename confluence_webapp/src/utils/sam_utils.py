from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class WebappDataset(Dataset):
    """
    A PyTorch Dataset to load data from a json file in COCO format.

    Attributes
    ----------
    root_dir : str
        the root directory containing the images and annotations
    annotation_file : str
        name of the json file containing the annotations (in root_dir)
    transform : callable
        a function/transform to apply to each image
    """

    def __init__(
        self,
        file_list: List[str],
        transform: Optional[Callable[[NDArray[np.uint8]], torch.Tensor]] = None,
    ) -> None:
        self.file_list = file_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            image: Image.Image = Image.open(self.file_list[idx])
            image_array: NDArray[np.uint8] = np.array(image)
        except Exception as e:
            print("ERROR IN READING IMG")
            print(f" path: {self.file_list[idx]}")
            print(e)
            raise ValueError(f"image {self.file_list[idx]} cannot be read")

        try:
            image_rgb: NDArray[np.uint8] = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("ERROR IN COLOR")
            print(f"image: {self.file_list[idx]}")
            print(e)
            raise ValueError(f"image: BGR to RGB does not work {self.file_list[idx]}")

        if self.transform:
            image_transformed: torch.Tensor = self.transform(image_rgb)
            return image_transformed

        return torch.from_numpy(image_rgb)


class ResizeAndPad:
    """
    Resize and pad images and masks to a target size.

    Attributes
    ----------
    target_size : int
        the target size of the image
    transform : ResizeLongestSide
        a transform to resize the image and masks
    """

    def __init__(self, target_size: int) -> None:
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image: NDArray[np.uint8]) -> torch.Tensor:
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image_resized: NDArray[np.uint8] = self.transform.apply_image(image)
        image_tensor: torch.Tensor = self.to_tensor(image_resized)

        # Pad image and masks to form a square
        _, h, w = image_tensor.shape
        max_dim: int = max(w, h)
        pad_w: int = (max_dim - w) // 2
        pad_h: int = (max_dim - h) // 2

        padding: Tuple[int, int, int, int] = (
            pad_w,
            pad_h,
            max_dim - w - pad_w,
            max_dim - h - pad_h,
        )
        image_padded: torch.Tensor = transforms.Pad(padding)(image_tensor)

        return image_padded


def load_datasets(file_list: List[str]) -> DataLoader:
    """Load the training and validation datasets in PyTorch DataLoader objects.

    Args:
        file_list: List of file paths to images

    Returns:
        DataLoader for inference
    """
    transform = ResizeAndPad(1024)
    inference = WebappDataset(file_list=file_list, transform=transform)
    inference_dataloader: DataLoader = DataLoader(
        inference, batch_size=1, shuffle=True, num_workers=1
    )
    return inference_dataloader


def get_totalmask(masks: torch.Tensor) -> torch.Tensor:
    """Get all masks into one image.

    Args:
        masks: shape: (N, H, W) where N is the number of masks
              masks H,W is usually 1024,1024

    Returns:
        All masks combined into one image
    """
    total_gt: torch.Tensor = torch.zeros_like(masks[0, :, :])
    for k in range(len(masks)):
        total_gt += masks[k, :, :]
    return total_gt
