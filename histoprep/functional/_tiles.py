from __future__ import annotations

__all__ = ["get_tile_coordinates"]

import itertools

ERROR_TYPE = "Tile width and height should be integers, got {} and {}."
ERROR_NONZERO = (
    "Tile width and height should non-zero positive integers, got {} and {}."
)
ERROR_DIMENSION = (
    "Tile width ({}) and height ({}) should be smaller than image dimensions ({})."
)
ERROR_OVERLAP = "Overlap should be in range [0, 1), got {}."

OVERLAP_LIMIT = 1.0


def get_tile_coordinates(
    dimensions: tuple[int, int],
    width: int,
    *,
    height: int | None = None,
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
