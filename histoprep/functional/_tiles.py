__all__ = ["get_tile_coordinates", "multiply_xywh", "allowed_xywh", "pad_tile"]

import itertools
from typing import Optional, Union

import numpy as np

ERROR_TYPE = "Tile width and height should be integers, got {} and {}."
ERROR_NONZERO = (
    "Tile width and height should non-zero positive integers, got {} and {}."
)
ERROR_DIMENSION = (
    "Tile width ({}) and height ({}) should be smaller than image dimensions ({})."
)
ERROR_OVERLAP = "Overlap should be in range [0, 1), got {}."
ERROR_FILL = "Fill value should be between [0, 255], got {}."

OVERLAP_LIMIT = 1.0
MAX_FILL_VALUE = 255


def get_tile_coordinates(
    dimensions: tuple[int, int],
    width: int,
    *,
    height: Optional[int] = None,
    overlap: float = 0.0,
    out_of_bounds: bool = False,
) -> list[tuple[int, int, int, int]]:
    """Create tile coordinates (xywh).

    Args:
        dimensions: Image dimensions (height, width).
        width: Tile width.
        height: Tile height. If None, will be set to `width`. Defaults to None.
        overlap: Overlap between neighbouring tiles. Defaults to 0.0.
        out_of_bounds: Allow tiles to go out of image bounds. Defaults to False.

    Raises:
        TypeError: Height and/or width are not integers.
        ValueError: Height and/or width are not less or equal to zero.
        ValueError: Height and/or width are larger than dimensions.
        ValueError: Overlap is not in range [0, 1).

    Returns:
        List of xywh-coordinates.
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


def multiply_xywh(
    xywh: tuple[int, int, int, int], multiplier: Union[float, tuple[float, float]]
) -> tuple[int, int, int, int]:
    """Multiply xywh-coordinates with multiplier(s)."""
    if not isinstance(multiplier, (tuple, list)):
        multiplier = (multiplier, multiplier)
    w_m, h_m = multiplier
    x, y, w, h = xywh
    return round(x / w_m), round(y / h_m), round(w / w_m), round(h / h_m)


def allowed_xywh(
    xywh: tuple[int, int, int, int], dimensions: tuple[int, int]
) -> Optional[tuple[int, int, int, int]]:
    """Get allowed xywh coordinates which are inside dimensions.

    Args:
        xywh: xywh-coordinates to check.
        dimensions: Allowed dimensions (height, width).

    Returns:
        xywh-coordinates inside the dimensions.
    """
    x, y, w, h = xywh
    height, width = dimensions
    if y > height or x > width:
        # xywh is outside of dimensions.
        return None
    if y + h > height or x + w > width:
        allowed_h = max(0, min(height - y, h))
        allowed_w = max(0, min(width - x, w))
        return x, y, allowed_h, allowed_w
    return x, y, w, h


def pad_tile(
    tile: np.ndarray, xywh: tuple[int, int, int, int], fill: int = 255
) -> np.ndarray:
    """Pad tile image with fill value to match w and h in xywh.

    Args:
        tile: Tile image.
        xywh: xywh-coordinates of the tile.
        fill: Fill value. Defaults to 255.

    Returns:
        Tile image, padded with fill (right and bottom) if tile size does not match
        coordinates.
    """
    if not 0 <= fill <= MAX_FILL_VALUE:
        raise ValueError(ERROR_FILL.format(fill))
    __, __, w, h = xywh
    tile_h, tile_w = tile.shape[:2]
    if tile_h < h or tile_w < w:
        output = np.zeros((h, w), dtype=np.uint8) + fill
        output[:tile_h, :tile_h] = tile
        return output
    return tile
