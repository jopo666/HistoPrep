from collections.abc import Iterable
from pathlib import Path
from typing import Union

import mpire
import numpy as np
from PIL import Image

from ._check import check_image

MEAN = tuple[float, ...]
STD = tuple[float, ...]


def get_mean_and_std_from_images(images: Iterable[np.ndarray]) -> tuple[MEAN, STD]:
    """Get channel mean and std values for an iterable of images.

    Args:
        images: Iterable of array images.

    Returns:
        Tuple of channel mean and std values.
    """
    mean, std = [], []
    for img in images:
        m, s = _get_mean_and_std(img)
        mean.append(m)
        std.append(s)
    return tuple(np.array(mean).mean(0)), tuple(np.array(std).mean(0))


def get_mean_and_std_from_paths(
    filepaths: Iterable[Union[str, Path]], num_workers: int = 1
) -> tuple[MEAN, STD]:
    """Get channel mean and std values for an iterable of image paths.

    Args:
        filepaths: Iterable of image paths.
        num_workers: Number of image reading processes.

    Returns:
        Tuple of channel mean and std values.
    """
    if num_workers <= 1:
        return get_mean_and_std_from_images(_read_image(x) for x in filepaths)
    with mpire.WorkerPool(n_jobs=num_workers) as pool:
        images = pool.imap(
            _read_image, ((x,) for x in filepaths), iterable_len=len(filepaths)
        )
        return get_mean_and_std_from_images(images)


def _get_mean_and_std(image: np.ndarray) -> tuple[MEAN, STD]:
    """Calculate mean and standard deviation for each image channel (between [0, 1])."""
    check_image(image)
    if image.ndim == 2:  # noqa
        image = np.expand_dims(tile, -1)  # noqa
    return (
        tuple([image[..., i].mean() / 255 for i in range(image.shape[-1])]),
        tuple([image[..., i].std() / 255 for i in range(image.shape[-1])]),
    )


def _read_image(path: Union[str, Path]) -> np.ndarray:
    return np.array(Image.open(path))
