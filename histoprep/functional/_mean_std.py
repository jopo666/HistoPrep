__all__ = ["_get_mean_and_std"]

from collections.abc import Iterable

import numpy as np

from ._check import check_image


def _get_mean_and_std(
    images: Iterable[np.ndarray],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Calculate mean and standard deviation for each image channel (between [0, 1]).

    Args:
        images: Iterable of images.

    Returns:
        Mean and standard deviation for each channel.
    """
    mean, std = [], []
    for image, __ in images:
        check_image(image)
        if image.ndim == 2:  # noqa
            image = np.expand_dims(tile, -1)  # noqa
        mean.append([image[..., i].mean() for i in range(image.shape[-1])])
        std.append([image[..., i].std() for i in range(image.shape[-1])])
    return tuple(np.array(mean).mean(0) / 255), tuple(np.array(std).mean(0) / 255)
