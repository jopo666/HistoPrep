from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._images import read_images


def plot_histogram(
    values: np.ndarray, n_bins: int, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    """Plot histogram."""
    if ax is None:
        plt.hist(values, bins=n_bins, **kwargs)
        ax = plt.gca()
    else:
        ax.hist(values, bins=n_bins, **kwargs)
    ax.set_xlim(values.min(), values.max())
    return ax


def get_bin_collages(
    paths: np.ndarray,
    values: np.ndarray,
    n_bins: int,
    n_images_per_bin: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Collect image paths based on binned values."""
    __, cutoffs = np.histogram(values, bins=n_bins)
    bin_indices = np.digitize(values, cutoffs[:-1]) - 1
    bin_paths = _collect_bin_paths(
        paths, bin_indices=bin_indices, n_images_per_bin=n_images_per_bin, rng=rng
    )
    return _read_bin_paths(bin_paths, n_bins=n_bins, n_images_per_bin=n_images_per_bin)


def _collect_bin_paths(
    paths: np.ndarray,
    bin_indices: np.ndarray,
    n_images_per_bin: int,
    rng: np.random.Generator,
) -> list[str]:
    output = []
    for bin_idx in range(bin_indices.max() + 1):
        bin_paths = paths[bin_indices == bin_idx]
        if len(bin_paths) == 0:
            output += [None] * n_images_per_bin
        elif len(bin_paths) < n_images_per_bin:
            output += bin_paths.tolist()
            output += [None] * (n_images_per_bin - len(bin_paths))
        else:
            output += rng.choice(bin_paths, n_images_per_bin).tolist()
    return output


def _read_bin_paths(
    bin_paths: list[str | None], n_bins: int, n_images_per_bin: int
) -> list[np.ndarray]:
    """Read paths for each bin and generate vertically stacked array."""
    # Read all images to get resize shape for empty images.
    all_images = []
    resize_shape = None
    for image in read_images(bin_paths):
        if resize_shape is None and image is not None:
            resize_shape = image.shape
        all_images.append(image)
    if resize_shape is None:
        # No images.
        empty = np.zeros((2 * n_images_per_bin, 2, 3), dtype=np.uint8) + 255
        return [empty for __ in range(n_bins)]
    # Generate collages.
    output, column = [], []
    for img in all_images:
        if img is None:
            img = np.zeros(resize_shape, dtype=np.uint8) + 255  # noqa
        column.append(img)
        if len(column) == n_images_per_bin:
            output.append(np.vstack(column))
            column = []
    return output
