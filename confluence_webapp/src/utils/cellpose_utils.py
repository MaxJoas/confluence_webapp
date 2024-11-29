import numpy as np
from skimage import measure


def calculate_mean_diameters(masks):
    horizontal_diameters = []
    vertical_diameters = []

    # Iterate through each mask
    for mask in masks:
        # Find contours of the mask
        contours = measure.find_contours(mask, 0.5)

        # Calculate the diameter using the maximum distance between points in the contour
        if len(contours) > 0:
            # contour = contours[0]
            for contour in contours:
                y, x = contour.T
                dx = x[:, np.newaxis] - x
                dy = y[:, np.newaxis] - y
                horizontal_distances = np.abs(dx)
                vertical_distances = np.abs(dy)
                horizontal_diameter = np.max(horizontal_distances)
                vertical_diameter = np.max(vertical_distances)
                horizontal_diameters.append(horizontal_diameter)
                vertical_diameters.append(vertical_diameter)

    # Calculate the mean diameters
    mean_horizontal_diameter = np.mean(horizontal_diameters)
    mean_vertical_diameter = np.mean(vertical_diameters)
