import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


######################################################################
# Lable Generation for Threshold Maps
######################################################################
def calculate_distance(grid_x, grid_y, segment_start, segment_end):
    """
    Calculate the perpendicular distance from each point in a grid to a line segment.

    Args:
        - grid_x (np.ndarray): The x-coordinates of the grid.
        - grid_y (np.ndarray): The y-coordinates of the grid.
        - segment_start (tuple): The starting point of the line segment.
        - segment_end (tuple): The ending point of the line segment.

    Returns:
        - np.ndarray: An array of distances.
    """
    # Calculating the square of distances from the grid points to the line segment points
    distance_to_start_sq = np.square(grid_x - segment_start[0]) + np.square(
        grid_y - segment_start[1]
    )
    distance_to_end_sq = np.square(grid_x - segment_end[0]) + np.square(
        grid_y - segment_end[1]
    )

    # Calculating the square of the distance between the line segment points
    segment_length_sq = np.square(
        segment_start[0] - segment_end[0]
    ) + np.square(segment_start[1] - segment_end[1])

    # Calculating cosine of the angle at the grid points
    cosine_angle = (
        segment_length_sq - distance_to_start_sq - distance_to_end_sq
    ) / (2 * np.sqrt(distance_to_start_sq * distance_to_end_sq) + 1e-6)

    # Calculating the square of sine of the angle
    sine_square = 1 - np.square(cosine_angle)
    sine_square = np.nan_to_num(sine_square)

    # Calculating the perpendicular distance
    perpendicular_distance = np.sqrt(
        distance_to_start_sq
        * distance_to_end_sq
        * sine_square
        / segment_length_sq
    )
    perpendicular_distance[cosine_angle < 0] = np.sqrt(
        np.fmin(distance_to_start_sq, distance_to_end_sq)
    )[cosine_angle < 0]

    return perpendicular_distance


def draw_border_map_on_canvas(polygon_points, canvas, mask, shrink_ratio):
    """
    Draw a border map on a canvas based on a polygon and a shrink ratio.

    Args:
        - polygon_points (list): Points of the polygon.
        - canvas (np.ndarray): The canvas to draw on.
        - mask (np.ndarray): The mask to use for drawing.
        - shrink_ratio (float): The ratio to shrink the polygon by.
    """
    polygon = np.array(polygon_points)
    polygon_shape = Polygon(polygon)

    # Exit if the polygon area is not positive
    if polygon_shape.area <= 0:
        return

    # Calculating the distance to shrink the polygon by
    shrink_distance = (
        polygon_shape.area
        * (1 - np.power(shrink_ratio, 2))
        / polygon_shape.length
    )

    # Creating the shrunk polygon
    subject_polygon = [tuple(l) for l in polygon]
    polygon_padder = pyclipper.PyclipperOffset()
    polygon_padder.AddPath(
        subject_polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON
    )
    shrunk_polygon = np.array(polygon_padder.Execute(shrink_distance)[0])

    # Filling the polygon in the mask
    cv2.fillPoly(mask, [shrunk_polygon.astype(np.int32)], 1.0)

    # Calculating bounding box for the shrunk polygon
    xmin, xmax = shrunk_polygon[:, 0].min(), shrunk_polygon[:, 0].max()
    ymin, ymax = shrunk_polygon[:, 1].min(), shrunk_polygon[:, 1].max()
    width, height = xmax - xmin + 1, ymax - ymin + 1

    # Adjusting polygon points relative to the bounding box
    adjusted_polygon = polygon.copy()
    adjusted_polygon[:, 0] -= xmin
    adjusted_polygon[:, 1] -= ymin

    # Creating grids for distance calculation
    grid_x = np.broadcast_to(
        np.linspace(0, width - 1, num=width).reshape(1, width), (height, width)
    )
    grid_y = np.broadcast_to(
        np.linspace(0, height - 1, num=height).reshape(height, 1),
        (height, width),
    )

    # Initializing the distance map
    distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)

    # Calculating the distance to each edge of the polygon
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = calculate_distance(
            grid_x, grid_y, adjusted_polygon[i], adjusted_polygon[j]
        )
        distance_map[i] = np.clip(absolute_distance / shrink_distance, 0, 1)

    # Taking the minimum distance to any edge
    distance_map = distance_map.min(axis=0)

    # Validating and adjusting the bounding box coordinates
    xmin_valid, xmax_valid = min(max(0, xmin), canvas.shape[1] - 1), min(
        max(0, xmax), canvas.shape[1] - 1
    )
    ymin_valid, ymax_valid = min(max(0, ymin), canvas.shape[0] - 1), min(
        max(0, ymax), canvas.shape[0] - 1
    )

    # Applying the distance map to the canvas
    canvas_slice = canvas[
        ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1
    ]
    distance_map_slice = distance_map[
        ymin_valid - ymin : ymax_valid - ymax + height,
        xmin_valid - xmin : xmax_valid - xmax + width,
    ]
    canvas[ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1] = np.fmax(
        1 - distance_map_slice, canvas_slice
    )


def generate_threshold_label(
    data, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7
):
    """
    Generate a threshold label for an image based on given polygons and ignore tags.

    Args:
        - data (dict): A dictionary containing 'image', 'polygons', and 'ignore_tags'.
        - shrink_ratio (float): Ratio to shrink polygons by.
        - thresh_min (float): Minimum threshold value.
        - thresh_max (float): Maximum threshold value.
    """
    image = data["image"]
    polygons = data["polygons"]
    ignore_flags = data["ignore_flags"]

    # Initializing canvas and mask
    canvas = np.zeros(image.shape[:2], dtype=np.float32)
    mask = np.zeros(image.shape[:2], dtype=np.float32)

    # Drawing each polygon on the canvas
    for polygon, ignore_flag in zip(polygons, ignore_flags):
        if not ignore_flag:
            draw_border_map_on_canvas(
                polygon, canvas, mask=mask, shrink_ratio=shrink_ratio
            )

    # Adjusting the canvas based on thresholds
    canvas = canvas * (thresh_max - thresh_min) + thresh_min

    # Updating the data dictionary
    data["threshold_map"] = canvas
    data["threshold_mask"] = mask


######################################################################
# Label Generation for Probability and Binary Maps
######################################################################
def adjust_polygons_within_image_bounds(
    polygons, ignore_flags, image_height, image_width
):
    """
    Adjusts polygons to ensure they fit within the image bounds and calculates
    their area to update ignore flags for invalid polygons.

    Parameters:
        - polygons (np.ndarray): Array of polygons coordinates.
        - ignore_flags (list): List of boolean flags to ignore certain polygons.
        - image_height (int): Height of the image.
        - image_width (int): Width of the image.

    Returns:
        - Tuple: (Adjusted polygons, Updated ignore flags)
    """
    for i, polygon in enumerate(polygons):
        # Clip polygon coordinates to be within image dimensions
        polygons[i] = np.clip(
            polygon, [0, 0], [image_width - 1, image_height - 1]
        )

        # If polygon area is too small, mark it as ignored
        if Polygon(polygon).area < 1:
            ignore_flags[i] = True
            polygons[i] = polygons[i][::-1]  # Reverse polygon coordinates

    return polygons, ignore_flags


def shrink_polygon(polygon, shrink_ratio):
    """
    Shrinks a given polygon according to the specified shrink ratio.

    Parameters:
        - polygon (Polygon): A Shapely Polygon object.
        - shrink_ratio (float): Ratio to shrink the polygon.

    Returns:
        - Polygon: The shrunk polygon.
    """
    if polygon.area == 0:
        return Polygon()

    # Calculate shrinking distance based on area and perimeter
    shrink_distance = polygon.area * (1 - shrink_ratio**2) / polygon.length
    return polygon.buffer(-shrink_distance, join_style=2)


def generate_text_probability_map(data, min_text_size=8, shrink_ratio=0.4):
    """
    Generates a probability map for text detection in an image by processing
    the provided polygons.

    Parameters:
        - data (dict): Contains the image, polygons, and ignore tags.
        - min_text_size (int): Minimum size of text to be detected.
        - shrink_ratio (float): Ratio for shrinking the polygons.

    Modifies:
        - data (dict): Adds the shrink_map and shrink_mask to the data.
    """
    image = data["image"]
    polygons = data["polygons"]
    ignore_flags = data["ignore_flags"]
    image_height, image_width = image.shape[:2]

    polygons, ignore_flags = adjust_polygons_within_image_bounds(
        polygons, ignore_flags, image_height, image_width
    )
    text_region_map = np.zeros((image_height, image_width), dtype=np.float32)
    mask_map = np.ones((image_height, image_width), dtype=np.float32)

    for i, polygon in enumerate(polygons):
        # Check for ignored polygons or those too small to consider
        if (
            ignore_flags[i]
            or min(polygon[:, 1].ptp(), polygon[:, 0].ptp()) < min_text_size
        ):
            cv2.fillPoly(mask_map, [polygon.astype(np.int32)], 0)
            continue

        shrunk_polygon = shrink_polygon(Polygon(polygon), shrink_ratio)
        if shrunk_polygon.is_empty:
            cv2.fillPoly(mask_map, [polygon.astype(np.int32)], 0)
        else:
            cv2.fillPoly(
                text_region_map,
                [np.array(shrunk_polygon.exterior.coords).astype(np.int32)],
                1,
            )

    data["shrink_map"] = text_region_map
    data["shrink_mask"] = mask_map
