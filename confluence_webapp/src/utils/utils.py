import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from numpy.typing import NDArray
from skimage import measure

logger = logging.getLogger(__name__)

import cv2


def visualize_contours(img: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Visualize contours on a given grayscale image using a prediction mask.

    Args:
        img (np.ndarray): Grayscale image (e.g., from cell imaging).
        pred_mask (np.ndarray): Binary mask with predictions.

    Returns:
        np.ndarray: RGB image with contours drawn.
    """
    # Find contours in the prediction mask
    contours_pred, _ = cv2.findContours(
        pred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Convert grayscale image to RGB for visualization
    cells_pred = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    cells_pred = cv2.drawContours(
        cells_pred, contours_pred, -1, (0, 255, 255), 2
    )
    
    return cells_pred

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
    binary_mask: NDArray[np.bool_], 
    tolerance: float = 0
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
        binary_mask, 
        pad_width=1, 
        mode="constant", 
        constant_values=0
    )
    contours: List[NDArray[np.float64]] = measure.find_contours(padded_binary_mask, 0.5)
    
    for contour in contours:
        contour_closed: NDArray[np.float64] = close_contour(contour)
        contour_approx: NDArray[np.float64] = measure.approximate_polygon(
            contour_closed, 
            tolerance
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
    data_dict: Dict[str, Union[str, int, Path]], 
    pred: Dict[str, torch.Tensor]
) -> tuple[float, List[List[List[float]]]]:
    """Calculates impedance from detectron predictions.

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
    file_name: Union[str, Path] = data_dict["file_name"]
    if not isinstance(file_name, str):
        file_name = file_name.name  # when file comes from webapp
        
    annotations: List[List[List[float]]] = []
    pixel_sum: int = 0
    
    for i in range(len(pred["instances"])):
        mask_array: NDArray[np.bool_] = masks[i, :, :].detach().cpu().clone().numpy()
        pixel_sum += int(mask_array.sum())
        
        try:
            segmentation: List[List[float]] = binary_mask_to_polygon(mask_array)
            annotations.append(segmentation)
        except Exception as e:
            logger.warning(e)
            logger.warning(
                f"Could not create polygon from prediction {i} from image {file_name}"
            )
            logger.warning(f"Shape: {np.unique(mask_array)}")

    confluence: float = pixel_sum / (data_dict["height"] * data_dict["width"])
    return confluence, annotations