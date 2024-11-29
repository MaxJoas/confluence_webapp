import logging
from typing import List

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage import measure

logger = logging.getLogger(__name__)



def overlay(image, mask, alpha, color=(255,0,0), resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    print(colored_mask.shape)
    print(image.shape)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def visualize_contours(img: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Visualize contours on a given image using a prediction mask.

    Args:
        img (np.ndarray): Image (grayscale or RGB).
        pred_mask (np.ndarray): Binary mask with predictions.

    Returns:
        np.ndarray: RGB image with contours drawn.
    """
    # Find contours in the prediction mask
    contours_pred, _ = cv2.findContours(
        pred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Check if the image is grayscale (2D) or RGB (3D with 3 channels)
    if len(img.shape) == 2 or img.shape[2] == 1:
        # Convert grayscale to RGB
        cells_pred = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # Use the image directly if it's already RGB
        cells_pred = img

    # Draw contours on the RGB image
    cells_pred = cv2.drawContours(
        cells_pred, contours_pred, -1, (0, 0, 255), 2
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

