from __future__ import annotations

__all__ = [
    "get_non_overlapping_regions",
    "get_overlap_area",
    "get_overlap_index",
    "get_tile_coordinates",
]

import itertools

import numpy as np

ERROR_TYPE = "Tile width and height should be integers, got {} and {}."
ERROR_NONZERO = (
    "Tile width and height should non-zero positive integers, got {} and {}."
)
ERROR_DIMENSION = (
    "Tile width ({}) and height ({}) should be smaller than image dimensions ({})."
)
ERROR_OVERLAP = "Overlap should be in range [0, 1), got {}."

OVERLAP_LIMIT = 1.0
XYWH = tuple[int, int, int, int]


def get_tile_coordinates(
    dimensions: tuple[int, int],
    width: int,
    *,
    height: int | None = None,
    overlap: float = 0.0,
    out_of_bounds: bool = False,
) -> list[XYWH]:
    """Create tile coordinates (xywh).

    Args:
        dimensions: Image dimensions (height, width).
        width: Tile width.
        height: Tile height. If None, will be set to `width`. Defaults to None.
        overlap: Overlap between neighbouring tiles. Defaults to 0.0.
        out_of_bounds: Allow tiles to go out of image bounds. Defaults to False.

    Raises:
        TypeError: Height and/or width are not integers.
        ValueError: Height and/or width are zero or less.
        ValueError: Height and/or width are larger than dimensions.
        ValueError: Overlap is not in range [0, 1).

    Returns:
        List of xywh-coordinates.

    Examples:
        >>> get_tile_coordinates((16, 8), width=8, overlap=0.5)
        [(0, 0, 8, 8), (0, 4, 8, 8), (0, 8, 8, 8)]
    """
    # Check arguments.
    if height is None:
        height = width
    if not isinstance(height, int) or not isinstance(width, int):
        raise TypeError(ERROR_TYPE.format(width, height))
    if height <= 0 or width <= 0:
        raise ValueError(ERROR_NONZERO.format(width, height))
    if height > dimensions[0] or width > dimensions[1]:
        raise ValueError(ERROR_DIMENSION.format(width, height, dimensions))
    if not 0 <= overlap < OVERLAP_LIMIT:
        raise ValueError(ERROR_OVERLAP.format(overlap))
    # Collect xy-coordinates.
    level_height, level_width = dimensions
    width_step = max(width - round(width * overlap), 1)
    height_step = max(height - round(height * overlap), 1)
    x_coords = range(0, level_width, width_step)
    y_coords = range(0, level_height, height_step)
    # Filter out of bounds coordinates.
    if not out_of_bounds and max(x_coords) + width > level_width:
        x_coords = x_coords[:-1]
    if not out_of_bounds and max(y_coords) + height > level_height:
        y_coords = y_coords[:-1]
    # Take product and add width and height.
    return [(x, y, width, height) for y, x in itertools.product(y_coords, x_coords)]


def get_non_overlapping_regions(
    tile_coordinates: list[XYWH], overlap: float
) -> np.ndarray:
    """Extract non-overlapping xywh-coordinate regions from tile coordinates.

    Args:
        tile_coordinates: Tile coordinates.
        overlap: Overlap between neighbouring tile coordinates.

    Returns:
        List of non-overlapping tile coordinates.

    Examples:
        >>> coords = [[0, 0, 4, 4], [0, 2, 4, 4]]
        >>> get_non_overlapping_coordinates(coords, overlap=0.5).tolist()
        [[0, 0, 2, 2], [0, 2, 2, 2]]
    """
    if overlap <= 0:
        return tile_coordinates
    return np.array(
        [
            (x, y, int(w - w * overlap), int(h - h * overlap))
            for x, y, w, h in tile_coordinates
        ]
    )


def get_overlap_index(xywh: XYWH, coordinates: list[XYWH]) -> np.ndarray:
    """Indices of tiles in `coordinates` which overlap with `xywh`.

    Args:
        xywh: Coordinates.
        tile_coordinates: List of tile coordinates.

    Returns:
        Indices of tiles which overlap with xywh.

    Examples:
        >>> xywh = [5, 5, 5, 5]
        >>> coordinates = [[0, 0, 5, 5], [0, 0, 5, 6], [0, 0, 6, 6], [10, 10, 1, 1]]
        >>> get_overlap_index(xywh, coordinates)
        array([2])
    """
    x, y, w, h = xywh
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates, dtype=int)
    return np.argwhere(
        (coordinates[:, 0] < x + w)
        & (coordinates[:, 0] + coordinates[:, 2] > x)
        & (coordinates[:, 1] < y + h)
        & (coordinates[:, 1] + coordinates[:, 3] > y)
    ).flatten()


def get_overlap_area(
    xywh: tuple[int, int, int, int],
    coordinates: list[XYWH],
) -> np.ndarray:
    """Calculate how much each coordinate overlaps with `xywh`.

    Args:
        xywh: Coordinates.
        coordinates: List of coordinates.

    Returns:
        Overlapping area for each tile in `coordinates`

    Examples:
        >>> xywh = [5, 5, 5, 5]
        >>> coordinates = [[0, 0, 100, 100], [0, 0, 4, 4], [4, 4, 2, 2], [11, 11, 2, 2]]
        >>> get_overlap_area(xywh, coordinates)
        array([25,  0,  1,  0])
    """
    x, y, w, h = xywh
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    areas = np.zeros(len(coordinates))
    overlap_index = get_overlap_index(xywh, coordinates)
    x_overlap = coordinates[overlap_index, 0] + coordinates[overlap_index, 2] - w
    y_overlap = coordinates[overlap_index, 1] + coordinates[overlap_index, 3] - h
    areas[overlap_index] = np.minimum(x_overlap, w) * np.minimum(y_overlap, h)
    return areas.astype(int)
