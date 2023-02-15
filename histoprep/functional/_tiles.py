__all__ = ["get_tile_coordinates", "multiply_xywh"]

import itertools
from typing import Optional, Union

OVERLAP_LIMIT = 1.0
ERROR_NON_INTEGER_SHAPE = "Tile {} should be and integer, not {}."
ERROR_NON_POSITIVE_SHAPE = "Tile {} ({}) should positive non-zero integer."
ERROR_SHAPE_LARGER_THAN_DIM = "Tile {} ({}) is larger than image {} ({})."
ERROR_OVERLAP = "Overlap ({}) should be in range [0, 1)."


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
        height: Height of a tile. If None, will be set to `width`. Defaults to None.
        overlap: Overlap between neighbouring tiles. Defaults to 0.0.
        out_of_bounds: Allow tiles to go out of image bounds. Defaults to False.

    Returns:
        List of xywh-coordinates.
    """
    # Check arguments.
    if height is None:
        height = width
    for idx, (name, val) in enumerate([("height", height), ("width", width)]):
        if not isinstance(val, int):
            raise TypeError(ERROR_NON_INTEGER_SHAPE.format(name, type(val)))
        if not val > 0:
            raise ValueError(ERROR_NON_POSITIVE_SHAPE.format(name, val))
        if val > dimensions[idx]:
            raise ValueError(
                ERROR_SHAPE_LARGER_THAN_DIM.format(name, val, name, dimensions[idx])
            )
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
