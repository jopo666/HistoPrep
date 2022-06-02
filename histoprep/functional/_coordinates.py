import itertools
import logging
from typing import List, Tuple, Union

import numpy

__all__ = ["tile_coordinates", "filter_coordinates"]


def tile_coordinates(
    dimensions: Tuple[int, int],
    width: int,
    height: int = None,
    overlap: float = 0.0,
) -> List[Tuple[int, int, int, int]]:
    """Extract a list of tile coordinates based on image dimensions.

    Args:
        dimensions: Image dimensions (height, width).
        width: Width of a tile.
        height: Height of a tile. If None, will be set to width. Defaults to None.
        overlap: Overlap between neighbouring tiles. Defaults to 0.0.

    Returns:
        Tile coordinates in XYWH format.

    Example:
        ```python
        import histoprep.functional as F
        from histoprep.helpers import read_image

        # Read image and extract tile coordinates.
        image = read_image("path/to/image.jpeg")
        coordinates = F.tile_coordinates(
            dimensions=image.size,
            width=512,
            overlap=0.25,
        )
        ```
    """
    if not isinstance(dimensions, tuple) or not isinstance(dimensions[0], int):
        raise TypeError("Dimensions should be a tuple of integers.")
    elif len(dimensions) != 2:
        raise ValueError(
            "Dimensions should contain 2 values for height and width, "
            "not {}.".format(len(dimensions))
        )
    if not 0 <= overlap < 1.0:
        raise ValueError("Overlap should be in range [0, 1).")
    if height is None:
        height = width
    if not (isinstance(width, int) and isinstance(height, int)):
        raise TypeError("Height and width should be integers.")
    if height <= 0 or width <= 0:
        raise ValueError("Height and width should be over 0.")
    if height > dimensions[0] or width > dimensions[1]:
        raise ValueError("Tile height or width is larger than image dimensions.")
    # Collect y coords.
    y = [0]
    overlap_y = int(height * overlap)
    while y[-1] < dimensions[0]:
        y.append(y[-1] + height - overlap_y)
    y = y[:-1]
    # Collect x coords.
    x = [0]
    overlap_x = int(width * overlap)
    while x[-1] < dimensions[1]:
        x.append(x[-1] + width - overlap_x)
    x = x[:-1]
    # Take product.
    coordinates = list(itertools.product(x, y))
    # Add width and height.
    coordinates = [xy + (width, height) for xy in coordinates]
    return coordinates


def filter_coordinates(
    coordinates: List[Tuple[int, int, int]],
    tissue_mask: numpy.ndarray,
    max_background: float = 0.95,
    downsample: Union[float, Tuple[float, float]] = 1.0,
) -> List[Tuple[int, int, int, int, float]]:
    """Filter a list of coordinates based on the amount of background.

    Args:
        coordinates: List of coordinates in XYWH format.
        mask: Tissue mask.
        max_background: Maximum amount of background in tile.  Defaults to 0.95.
        downsample: Downsample of the tissue mask. Defaults to 1.

    Returns:
        Filtered list of coordinates.

    Example:
        ```python
        import histoprep.functional as F
        from histoprep.helpers import read_image

        # Read image and extract tile coordinates.
        image = read_image("path/to/image.jpeg")
        coordinates = F.tile_coordinates(
            dimensions=image.size,
            width=512,
            overlap=0.25,
        )
        # Detect tissue.
        tissue_mask = F.detect_tissue(image)
        # Filter coordinates based on the amount of background.
        filtered_coordinates = F.filter_coordinates(
            coordinates=coordinates,
            tissue_mask=tissue_mask,
            max_background=0.9,
        )
        ```
    """
    if not 0 <= max_background <= 1:
        raise ValueError("Maximum baground should be in range [0,1].")
    if not isinstance(coordinates, list):
        raise TypeError("Coordinates should be a list of tuples (X, Y, W, H).")
    elif len(coordinates) == 0:
        logging.debug("Passed empty list of coordinates to filter_coordinates.")
        return coordinates
    if (
        not isinstance(coordinates[0], tuple)
        or not isinstance(coordinates[0][0], int)
        or len(coordinates[0]) != 4
    ):
        raise ValueError("Coordinates should be a list of tuples (X, Y, W, H).")
    filtered = []
    if not isinstance(downsample, (tuple, list)):
        downsample = (downsample, downsample)
    for x, y, w, h in coordinates:
        x_d = round(x / downsample[1])
        w_d = round(w / downsample[1])
        y_d = round(y / downsample[0])
        h_d = round(w / downsample[0])
        tile_mask = tissue_mask[y_d : y_d + h_d, x_d : x_d + w_d]
        if tile_mask.size > 0:
            background_percentage = (1 - tile_mask).sum() / tile_mask.size
            if background_percentage <= max_background:
                filtered.append((x, y, w, h))
    return filtered
