from __future__ import annotations

__all__ = ["SlideReaderDataset"]

from collections.abc import Callable
from typing import Any

import numpy as np

from histoprep._reader import SlideReader
from histoprep.backend import TileCoordinates

try:
    from torch.utils.data import Dataset

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    Dataset = object

ERROR_PYTORCH = "Could not import torch, make sure it has been installed!"


class SlideReaderDataset(Dataset):
    def __init__(
        self,
        reader: SlideReader,
        coordinates: TileCoordinates | list[tuple[int, int, int, int]],
        level: int = 0,
        transform: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        """Torch dataset yielding tile images.

        Args:
            reader: `SlideReader` instance.
            coordinates: `TileCoordinates` instance or a list of xywh-coordinates.
            level: Slide level for reading tile image. Defaults to 0.
            transform: Transform function for tile images. Defaults to None.
        """
        if not HAS_PYTORCH:
            raise ImportError(ERROR_PYTORCH)
        super().__init__()
        self.reader = reader
        self.coordinates = (
            coordinates.coordinates
            if isinstance(coordinates, TileCoordinates)
            else coordinates
        )
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
