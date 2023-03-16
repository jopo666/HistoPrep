import numpy as np
import polars as pl
import pytest

from histoprep import SlideReader
from histoprep.utils import TileMetadata

from ._utils import SLIDE_PATH_JPEG, TMP_DIRECTORY, clean_temporary_directory


def generate_metadata(*, clean_tmp: bool = True, **kwargs) -> pl.DataFrame:
    clean_temporary_directory()
    reader = SlideReader(SLIDE_PATH_JPEG)
    metadata = reader.save_regions(
        TMP_DIRECTORY,
        reader.get_tile_coordinates(None, 256, overlap=0.0),
        save_metrics=True,
        threshold=200,
        **kwargs,
    )
    if clean_tmp:
        clean_temporary_directory()
    return metadata


def test_metadata_properties() -> None:
    metadata = TileMetadata(generate_metadata())
    assert isinstance(metadata.dataframe, pl.DataFrame)
    assert metadata.outliers.sum() == 0
    assert len(metadata.outlier_selections) == 0
    assert len(metadata.metric_columns) == 64
    assert metadata.metrics.shape == (100, 64)
    assert metadata.mean_and_std == (
        (0.8448732156862745, 0.7013530588235295, 0.7794474117647058),
        (0.13158384313725494, 0.1708792549019608, 0.13072776470588235),
    )
    assert str(metadata) == "TileMetadata(num_images=100, num_outliers=0)"


def test_metadata_index_columns() -> None:
    metadata = TileMetadata(generate_metadata())
    assert isinstance(metadata["path"], np.ndarray)
    assert isinstance(metadata["red_mean"], np.ndarray)


def test_metadata_from_parquet() -> None:
    generate_metadata(clean_tmp=False)
    metadata = TileMetadata.from_parquet(TMP_DIRECTORY / "slide" / "*.parquet")
    assert len(metadata.metric_columns) == 64
    clean_temporary_directory()


def test_metadata_from_csv() -> None:
    generate_metadata(clean_tmp=False, use_csv=True)
    metadata = TileMetadata.from_csv(TMP_DIRECTORY / "slide" / "*.csv")
    assert len(metadata.metric_columns) == 64
    clean_temporary_directory()


def test_metadata_plot_histogram() -> None:
    metadata = TileMetadata(generate_metadata(clean_tmp=False))
    metadata.plot_histogram("red_mean", num_images=0)
    metadata.plot_histogram("red_mean", num_images=12)
    clean_temporary_directory()


def test_metadata_plot_histogram_fail() -> None:
    metadata = TileMetadata(generate_metadata())
    with pytest.raises(ValueError, match="Difference between min=0.0 and max=0.0"):
        metadata.plot_histogram("black_pixels", num_images=0)


def test_metadata_plot_pca() -> None:
    metadata = TileMetadata(generate_metadata())
    metadata.plot_pca()


def test_metadata_plot_collage() -> None:
    metadata = TileMetadata(generate_metadata(clean_tmp=False))
    assert metadata.random_collage(~metadata.outliers, n_rows=4).size == (1024, 256)
    assert metadata.random_collage(~metadata.outliers, n_rows=2).size == (1024, 128)
    with pytest.raises(ValueError, match="Empty selection"):
        metadata.random_collage(metadata.outliers)
    clean_temporary_directory()


def test_metadata_add_outliers() -> None:
    metadata = TileMetadata(generate_metadata())
    metadata.add_outliers(metadata["background"] > 0.5, desc="too high background")
    assert metadata.outliers.sum() == 27
    assert len(metadata.outlier_selections) == 1
    assert metadata.outlier_selections[0]["desc"] == "too high background"
    assert metadata.outlier_selections[0]["selection"].sum() == 27


def test_metadata_cluster() -> None:
    metadata = TileMetadata(generate_metadata())
    clusters = metadata.cluster_kmeans(10)
    assert len(clusters) == len(metadata)


# def test_tile_metadata() -> None:
#     # Save tiles.
#     clean_temporary_directory()
#     reader = SlideReader(SLIDE_PATH_JPEG)
#     threshold, tissue_mask = reader.get_tissue_mask()
#     tiles = reader.get_tile_coordinates(tissue_mask, 64, max_background=0.75)
#     dataframe = reader.save_regions(
#         TMP_DIRECTORY, tiles, overwrite=True, save_metrics=True, threshold=threshold
#     )
#     # Initialize TileMetadata.
#     metadata = TileMetadata(dataframe)
#     # Test properites.
#     assert np.equal(
#         np.array(metadata.mean_and_std).round(3),
#         np.array([[0.807, 0.63, 0.729], [0.135, 0.156, 0.122]]),
#     ).all()
#     assert len(metadata.metric_columns) == 64
#     assert metadata.metrics.shape == (434, 64)
#     assert len(metadata.outliers) == 434
#     assert metadata.outliers.sum() == 0
#     # test plotting and methods.
#     metadata.plot_histogram("background")
#     clusters = metadata.cluster_kmeans(10)
#     metadata.plot_pca(clusters)
#     with pytest.raises(ValueError):  # noqa
#         metadata.plot_histogram("black_pixels")
#     with pytest.raises(ValueError):  # noqa
#         metadata.plot_histogram("black_pixels", ax="fake_axis", n_images=100)
#     assert metadata.random_collage(
#         selection=metadata["hue_std"] > 50, shape=(64, 64)
#     ).size == (16 * 64, 4 * 64)
#     # Add outliers.
#     metadata.add_outliers(metadata["background"] > 0.7, desc="high background")
#     assert metadata.outliers.sum() == 11
#     assert metadata.__repr__() == "TileMetadata(n_images=434, n_outliers=11)"
#     clean_temporary_directory()
