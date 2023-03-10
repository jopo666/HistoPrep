import numpy as np
import pytest

from histoprep import SlideReader
from histoprep.process import TileMetadata

from .utils import SLIDE_PATH_JPEG, TMP_DIRECTORY, clean_temporary_directory


def test_tile_metadata() -> None:
    # Save tiles.
    clean_temporary_directory()
    reader = SlideReader(SLIDE_PATH_JPEG)
    tissue = reader.get_tissue_mask()
    tiles = reader.get_tile_coordinates(tissue, 64, max_background=0.75)
    dataframe = reader.save_tiles(TMP_DIRECTORY, tiles, overwrite=True)
    # Initialize TileMetadata.
    metadata = TileMetadata(dataframe)
    # Test properites.
    assert np.equal(
        np.array(metadata.mean_and_std).round(3),
        np.array([[0.807, 0.63, 0.729], [0.135, 0.156, 0.122]]),
    ).all()
    assert len(metadata.metric_columns) == 64
    assert metadata.metrics.shape == (434, 64)
    assert len(metadata.outliers) == 0
    # test plotting and methods.
    metadata.plot_histogram("background")
    clusters = metadata.cluster_kmeans(10)
    metadata.plot_pca(clusters)
    with pytest.raises(ValueError):  # noqa
        metadata.plot_histogram("black_pixels")
    with pytest.raises(ValueError):  # noqa
        metadata.plot_histogram("black_pixels", ax="fake_axis", n_images=100)
    assert metadata.random_collage(
        selection=metadata["hue_std"] > 50, shape=(64, 64)
    ).size == (16 * 64, 4 * 64)
    clean_temporary_directory()
