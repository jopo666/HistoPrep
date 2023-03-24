from __future__ import annotations

__all__ = [
    "get_random_image_collage",
    "_read_images_from_paths",
    "_create_image_collage",
]

import random
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from mpire import WorkerPool
from PIL import Image


def get_random_image_collage(
    paths: Iterable[str | Path],
    num_rows: int = 4,
    num_cols: int = 16,
    shape: tuple[int, int] = (64, 64),
    num_workers: int = 1,
) -> Image.Image:
    """Create image collage from randomly sampled images from paths.

    Args:
        paths: Image paths.
        num_rows: Number of rows in the collage. Ignored if there isn't enough images.
            Defaults to 4.
        num_cols: Number of columns in the collage. Defaults to 16.
        shape: Shape of each image in the collage. Defaults to (64, 64).
        num_workers: Number of image loading workers. Defaults to 1.

    Returns:
        Image collage.
    """
    if len(paths) > num_cols * num_rows:
        paths = random.choices(paths, k=num_cols * num_rows)  # noqa
    images = _read_images_from_paths(paths=paths, num_workers=num_workers)
    return _create_image_collage(images=images, num_cols=num_cols, shape=shape)


def _read_images_from_paths(
    paths: Iterable[str | Path | None], num_workers: int
) -> list[np.ndarray]:
    """Read images from paths.

    Args:
        paths: Image paths.
        num_workers: Number of image loading workers.

    Returns:
        List of numpy array images.
    """
    if num_workers <= 1:
        return [_read_image(x) for x in paths]
    with WorkerPool(n_jobs=num_workers) as pool:
        output = list(pool.imap(_read_image, paths))
    return output  # noqa


def _create_image_collage(
    images: list[np.ndarray], num_cols: int, shape: tuple[int, int]
) -> Image.Image:
    """Collect images into a collage.

    Args:
        images: List of array images.
        num_cols: Number of columns. Number of rows is determined by
            `ceil(images/num_cols)`.
        shape: Shape for each image.

    Returns:
        Image collage.
    """
    if len(images) == 0:
        return None
    output, row = [], []
    for img in images:
        resized = cv2.resize(img, dsize=shape[::-1])
        row.append(resized)
        if len(row) == num_cols:
            output.append(np.hstack(row))
            row = []
    if len(row) > 0:
        row.extend([np.zeros_like(resized)] * (num_cols - len(row)))
        output.append(np.hstack(row))
    return Image.fromarray(np.vstack(output))


def _read_image(path: str | None) -> np.ndarray:
    """Parallisable."""
    if path is None:
        return None
    return np.array(Image.open(path))
