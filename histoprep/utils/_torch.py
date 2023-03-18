from __future__ import annotations

__all__ = ["SlideReaderDataset", "TileImageDataset"]

import ctypes
import math
import multiprocessing
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from histoprep._reader import SlideReader

try:
    from torch.utils.data import Dataset, IterableDataset

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    Dataset = object
    IterableDataset = object

ERROR_PYTORCH = "Could not import torch, make sure it has been installed!"
ERROR_LENGTH_MISMATCH = "Path length ({}) does not match label length ({})."
ERROR_TILE_SHAPE = "Tile shape must be defined to create a cache array."


class SlideReaderDataset(Dataset):
    def __init__(
        self,
        reader: SlideReader,
        coordinates: Iterator[tuple[int, int, int, int]],
        level: int = 0,
        transform: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        """Torch dataset yielding tile images from reader.

        Args:
            reader: `SlideReader` instance.
            coordinates: Iterator of xywh-coordinates.
            level: Slide level for reading tile image. Defaults to 0.
            transform: Transform function for tile images. Defaults to None.
        """
        if not HAS_PYTORCH:
            raise ImportError(ERROR_PYTORCH)
        super().__init__()
        self.reader = reader
        self.coordinates = coordinates
        self.level = level
        self.transform = transform

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, index: int) -> tuple[np.ndarray | Any, np.ndarray]:
        xywh = self.coordinates[index]
        tile = self.reader.read_region(xywh, level=self.level)
        if self.transform is not None:
            tile = self.transform(tile)
        return tile, np.array(xywh)


class TileImageDataset(Dataset):
    def __init__(
        self,
        paths: list[str | Path],
        *,
        labels: list[str] | None = None,
        transform: Callable[[np.ndarray], Any] | None = None,
        use_cache: bool = False,
        tile_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Torch dataset yielding tile images from paths.

        Args:
            paths: Paths to tile images.
            labels: Indexable list of labels for each path. Defaults to None.
            transform: Transform function for tile images.. Defaults to None.
            use_cache: Cache each image to shared array, requires that each tile has the
                same shape. Defaults to False.
            tile_shape: Tile shape for creating a shared cache array. Defaults to None.
        """
        super().__init__()
        if not HAS_PYTORCH:
            raise ImportError(ERROR_PYTORCH)
        if labels is not None and len(paths) != len(labels):
            raise ValueError(ERROR_LENGTH_MISMATCH.format(len(paths), len(labels)))
        if use_cache and tile_shape is None:
            raise ValueError(ERROR_TILE_SHAPE)
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self._use_cache = use_cache
        self._cached_indices = set()
        self._cache_array = None
        if self._use_cache:
            self._cache_array = create_shared_array(
                num_samples=len(self.paths), shape=tile_shape
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[np.ndarray, str] | tuple[Any, str]:
        path = self.paths[index]
        if self._use_cache:
            if index not in self._cached_indices:
                self._cache_array[index] = np.array(Image.open(path))
                self._cached_indices.add(index)
            image = self._cache_array[index]
        else:
            image = np.array(Image.open(path))
        if self.transform is not None:
            image = self.transform(image)
        labels = () if self.labels is None else self.labels[index]
        if not isinstance(labels, (tuple, list)):
            labels = (labels,)
        return image, path, *labels


def create_shared_array(num_samples: int, shape: tuple[int, ...]) -> np.ndarray:
    shared_array_base = multiprocessing.Array(
        ctypes.c_uint8, num_samples * math.prod(shape)
    )
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    return shared_array.reshape(num_samples, *shape)
