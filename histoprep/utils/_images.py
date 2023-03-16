from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from tqdm.contrib.concurrent import process_map


def _create_collage(
    images: list[np.ndarray], n_cols: int, shape: tuple[int, int]
) -> Image.Image:
    """Collect images into a collage."""
    if len(images) == 0:
        return None
    output, row = [], []
    for img in images:
        resized = cv2.resize(img, dsize=shape[::-1])
        row.append(resized)
        if len(row) == n_cols:
            output.append(np.hstack(row))
            row = []
    if len(row) > 0:
        row.extend([np.zeros_like(resized)] * (n_cols - len(row)))
        output.append(np.hstack(row))
    return Image.fromarray(np.vstack(output))


def _read_images(paths: list[str | None], chunksize: int = 8) -> list[np.ndarray]:
    """Read images from list of paths."""
    return process_map(_read_image, paths, disable=True, chunksize=chunksize)


def _read_image(path: str) -> np.ndarray:
    """Parallisable."""
    if path is None:
        return None
    return np.array(Image.open(path))
