import os
import pickle
from typing import Dict, List, Union

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from skimage import measure


def resize_image(image: np.ndarray, min_size: int, max_size: int) -> np.ndarray:
    """
    Resizes an input image while maintaining its aspect ratio,
    ensuring the shorter edge is at least `min_size` and the
    longer edge does not exceed `max_size`.

    Args:
        image (np.ndarray): The input image to resize, in the form of a NumPy array.
        min_size (int): The minimum size for the shorter edge of the image.
        max_size (int): The maximum allowed size for the longer edge of the image.

    Returns:
        np.ndarray: The resized image as a NumPy array.

    Example:
        resized_image = resize_image(input_image, 800, 1333)
    """
    h, w = image.shape[:2]

    # Scale to the shorter edge
    scale = min_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Ensure the longer edge does not exceed max_size
    if max(new_h, new_w) > max_size:
        scale = max_size / max(new_h, new_w)
        new_h, new_w = int(new_h * scale), int(new_w * scale)

    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image


def setup_cfg(iter=10):
    """
    Set up the Detectron2 configuration
    Args:
        iter (int): Number of iterations
    Returns:
        cfg (CfgNode): Detectron2 configuration
    """

    cfg = get_cfg()
    config_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
    cfg.SOLVER.IMS_PER_BATCH = 1
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.WARMUP_ITERS = 5
    cfg.SOLVER.MAX_ITER = iter
    cfg.SOLVER.STEPS = []
    # Small value=Frequent save  need a lot of storage.
    cfg.SOLVER.CHECKPOINT_PERIOD = 100000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    return cfg


def visualize_detectron_results(data_dict, anns, org_image):
    """
    Visualize the results of the Detectron2 model.

    Args:
        data_dict (dict): Dictionary containing image data
        anns (list): List of annotations
        org_image (PIL.Image): Original image
    Returns:
        img_out (Image): Output image
    """

    img_out = Image.new(
        "L",
        [
            data_dict["width"],
            data_dict["height"],
        ],
    )

    org_image = org_image.convert("RGBA")

    img_out1 = ImageDraw.Draw(img_out)
    overlay_out = ImageDraw.Draw(org_image, "RGBA")
    for ann in anns:
        if len(ann[0]) < 4:
            continue
        try:
            segs = ann[0]
            img_out1.polygon(segs, fill="white", outline="white")
            overlay_out.polygon(segs, fill=(21, 239, 116, 27), outline="#B7FF00")
        except Exception as e:
            print(f"Error: {e}")
            raise e
    return org_image



def visualize_result(out, img):
    """gets an instance and displays image"""
    # Load metadata
    with open(os.path.join("data","datasetdicts.pkl"), "rb") as f:
        dataset_dicts, metadata_dicts = pickle.load(f)

    vis = Visualizer(
        img[:, :, ::-1],
        scale=0.5,
        metadata=metadata_dicts,
        instance_mode=ColorMode.IMAGE,
    )  # This ensures uniform coloring

    for mask in out["instances"].pred_masks.to("cpu"):
        color = (1.0, 0.0, 0.0)  # Red, but in 0-1 range
        vis.draw_binary_mask(mask.numpy(), color=color, alpha=0.1)

    v = vis.get_output()
    img = v.get_image()[:, :, ::-1]
    return img


def close_contour(contour: NDArray[np.float64]) -> NDArray[np.float64]:
    """Close a contour by adding the first point to the end if necessary.

    Args:
        contour: Array of shape (N, 2) containing contour coordinates

    Returns:
        Closed contour array of shape (N+1, 2) or (N, 2)
    """
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(
    binary_mask: NDArray[np.bool_], tolerance: float = 0
) -> List[List[float]]:
    """Converts a binary mask to COCO polygon representation.

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    Returns:
        List of polygons, where each polygon is a list of flattened x,y coordinates
    """
    polygons: List[List[float]] = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask: NDArray[np.bool_] = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours: List[NDArray[np.float64]] = measure.find_contours(padded_binary_mask, 0.5)

    for contour in contours:
        contour_closed: NDArray[np.float64] = close_contour(contour)
        contour_approx: NDArray[np.float64] = measure.approximate_polygon(
            contour_closed, tolerance
        )
        if len(contour_approx) < 3:
            continue
        contour_flipped: NDArray[np.float64] = np.flip(contour_approx, axis=1)
        segmentation: List[float] = contour_flipped.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def get_confluence(
    data_dict: Dict[str, Union[str, int]], pred: Dict[str, torch.Tensor]
) -> tuple[float, List[List[List[float]]]]:
    """Calculates confluence from detectron predictions, handling overlapping cells.

    Args:
        data_dict: Metadata about image and annotations with keys:
            - file_name: str or Path
            - height: int
            - width: int
        pred: Data of detectron2 predictions containing:
            - instances: object with pred_masks tensor

    Returns:
        Tuple containing:
            - confluence: percentage of cells on image (float between 0 and 1)
            - annotations: List of polygon segmentations
    """
    masks: torch.Tensor = pred["instances"].pred_masks

    # Create a unique pixel tracking matrix
    unique_pixel_mask = np.zeros(
        (data_dict["height"], data_dict["width"]), dtype=np.bool_
    )
    annotations: List[List[List[float]]] = []

    for i in range(len(pred["instances"])):
        mask_array: NDArray[np.bool_] = masks[i, :, :].detach().cpu().clone().numpy()

        try:
            unique_pixels = np.logical_and(mask_array, ~unique_pixel_mask)
            unique_pixel_mask = np.logical_or(unique_pixel_mask, mask_array)

            if unique_pixels.any():
                segmentation: List[List[float]] = binary_mask_to_polygon(unique_pixels)
                annotations.append(segmentation)

        except Exception as e:
            print(f"Error: {e}")

    confluence: float = unique_pixel_mask.sum() / (
        data_dict["height"] * data_dict["width"]
    )

    confluence = min(confluence, 1.0)

    return confluence, annotations
