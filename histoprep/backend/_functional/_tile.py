__all__ = ["pad_tile"]

import numpy as np


def pad_tile(
    tile: np.ndarray, *, shape: tuple[int, int], fill: int = 255
) -> np.ndarray:
    """Pad tile image into shape."""
    tile_h, tile_w = tile.shape[:2]
    out_h, out_w = shape
    if tile_h == out_h and tile_w == out_w:
        return tile
    if tile.ndim > 2:
        shape = (out_h, out_w, tile.shape[-1])
    output = np.zeros(shape, dtype=np.uint8) + fill
    output[:tile_h, :tile_w] = tile
    return output
