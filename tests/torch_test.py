import torch
from torch.utils.data import DataLoader, Dataset

from histoprep import SlideReader
from histoprep.utils import SlideReaderDataset

from .utils import SLIDE_PATH_CZI


def test_torch_dataset() -> None:
    reader = SlideReader(SLIDE_PATH_CZI)
    tissue_mask = reader.get_tissue_mask()
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
