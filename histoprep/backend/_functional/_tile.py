from __future__ import annotations

__all__ = ["read_tile", "pad_tile"]

from collections.abc import Callable
from typing import Any

import numpy as np


def read_tile(
    worker_state: dict,
    xywh: tuple[int, int, int, int],
    *,
    level: int,
    transform: Callable[[np.ndarray], Any] | None,
    raise_exception: bool,
) -> np.ndarray | Exception | Any:
    """Parallisable tile reading function."""
    reader = worker_state["reader"]
    try:
        tile = reader.read_region(xywh=xywh, level=level)
    except KeyboardInterrupt:
        raise KeyboardInterrupt from None
    except Exception as catched_exception:  # noqa
        if raise_exception:
            raise catched_exception  # noqa
        return catched_exception
    if transform is not None:
        return transform(tile)
    return tile


def pad_tile(
    tile: np.ndarray, *, shape: tuple[int, int], fill: int = 255
) -> np.ndarray:
    """Pad tile image into shape."""
    tile_h, tile_w = tile.shape[:2]
    out_h, out_w = shape
    if tile_h == out_h and tile_w == out_w:
        return tile
    if tile.ndim > 2:  # noqa
        shape = (out_h, out_w, tile.shape[-1])
    output = np.zeros(shape, dtype=np.uint8) + fill
    output[:tile_h, :tile_w] = tile
    return output
