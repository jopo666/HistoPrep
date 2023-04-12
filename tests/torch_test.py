import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from histoprep import SlideReader
from histoprep.utils import (
    SlideReaderDataset,
    TileImageDataset,
)

from ._utils import (
    SLIDE_PATH_CZI,
    SLIDE_PATH_JPEG,
    TMP_DIRECTORY,
    clean_temporary_directory,
)


def test_reader_dataset_loader() -> None:
    reader = SlideReader(SLIDE_PATH_CZI)
    __, tissue_mask = reader.get_tissue_mask()
    coords = reader.get_tile_coordinates(tissue_mask, 512, max_background=0.01)
    # CZI fails if multiple workers read data from same instance, which should not
    # happen here as there is some voodoo shit going on with `Dataset` & `DataLoader`...
    dataset = SlideReaderDataset(reader, coords, level=1, transform=lambda z: z)
    assert isinstance(dataset, Dataset)
    loader = DataLoader(dataset, batch_size=4, num_workers=2)
    batch_images, batch_coords = next(iter(loader))
    assert batch_images.shape == (4, 256, 256, 3)
    assert isinstance(batch_images, torch.Tensor)
    assert batch_coords.shape == (4, 4)
    assert isinstance(batch_coords, torch.Tensor)


def test_tile_dataset_loader() -> None:
    clean_temporary_directory()
    reader = SlideReader(SLIDE_PATH_JPEG)
    metadata = reader.save_regions(TMP_DIRECTORY, reader.get_tile_coordinates(None, 96))
    dataset = TileImageDataset(
        metadata["path"].to_numpy(),
        labels=metadata[list("xywh")].to_numpy(),
        transform=lambda x: x[..., 0],
    )
    batch_images, batch_paths, batch_coords = next(
        iter(DataLoader(dataset, batch_size=32))
    )
    clean_temporary_directory()
    assert batch_images.shape == (32, 96, 96)
    assert len(batch_paths) == 32
    assert batch_coords.shape == (32, 4)


def test_tile_dataset_cache() -> None:
    clean_temporary_directory()
    reader = SlideReader(SLIDE_PATH_JPEG)
    metadata = reader.save_regions(TMP_DIRECTORY, reader.get_tile_coordinates(None, 96))
    dataset = TileImageDataset(
        metadata["path"].to_numpy(),
        labels=metadata[list("xywh")].to_numpy(),
        transform=lambda x: x[..., 0],
        use_cache=True,
        tile_shape=(96, 96, 3),
    )
    batch_images, batch_paths, batch_coords = next(
        iter(DataLoader(dataset, batch_size=32))
    )
    clean_temporary_directory()
    assert batch_images.shape == (32, 96, 96)
    assert len(batch_paths) == 32
    assert batch_coords.shape == (32, 4)
    assert dataset._cached_indices == set(range(32))
    assert np.equal(dataset._cache_array[0][..., 0], batch_images[0].numpy()).all()
