import warnings

import numpy as np
import polars as pl
import pytest
from openslide import OpenSlideUnsupportedFormatError
from PIL import Image, UnidentifiedImageError

import histoprep.functional as F
from histoprep import SlideReader
from histoprep.backend import CziBackend, OpenSlideBackend, PillowBackend
from histoprep.data import SpotCoordinates, TileCoordinates

from ._utils import (
    DATA_DIRECTORY,
    SLIDE_PATH_CZI,
    SLIDE_PATH_JPEG,
    SLIDE_PATH_SVS,
    SLIDE_PATH_TMA,
    TMP_DIRECTORY,
    clean_temporary_directory,
)
from .backend_test import (
    read_invalid_level,
    read_region_from_all_levels,
    read_zero_sized_region,
)


def test_reader_init_no_match() -> None:
    with pytest.raises(ValueError, match="Could not automatically assing reader"):
        __ = SlideReader("i/have/a/bad/extension.penis")


def test_reader_init_no_file() -> None:
    with pytest.raises(FileNotFoundError):
        __ = SlideReader("i/dont/exist.czi")


def test_reader_init_pillow() -> None:
    __ = SlideReader(SLIDE_PATH_JPEG)
    __ = SlideReader(SLIDE_PATH_JPEG, backend=PillowBackend)
    __ = SlideReader(SLIDE_PATH_JPEG, backend="PIL")
    __ = SlideReader(SLIDE_PATH_JPEG, backend="PILlow")
    with pytest.raises(UnidentifiedImageError):
        __ = SlideReader(SLIDE_PATH_CZI, backend="PILlow")


def test_reader_init_czi() -> None:
    __ = SlideReader(SLIDE_PATH_CZI)
    __ = SlideReader(SLIDE_PATH_CZI, backend=CziBackend)
    __ = SlideReader(SLIDE_PATH_CZI, backend="CZI")
    __ = SlideReader(SLIDE_PATH_CZI, backend="cZi")
    with pytest.raises(RuntimeError):
        __ = SlideReader(SLIDE_PATH_TMA, backend="czi")


def test_reader_init_openslide() -> None:
    __ = SlideReader(SLIDE_PATH_SVS)
    __ = SlideReader(SLIDE_PATH_SVS, backend=OpenSlideBackend)
    __ = SlideReader(SLIDE_PATH_SVS, backend="open")
    __ = SlideReader(SLIDE_PATH_SVS, backend="openSLIDe")
    with pytest.raises(OpenSlideUnsupportedFormatError):
        __ = SlideReader(SLIDE_PATH_JPEG, backend="openslide")


def test_reader_properties_backend() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    assert reader.path == reader.backend.path
    assert reader.name == reader.backend.name
    assert reader.data_bounds == reader.backend.data_bounds
    assert reader.dimensions == reader.backend.dimensions
    assert reader.level_count == reader.backend.level_count
    assert reader.level_dimensions == reader.backend.level_dimensions
    assert reader.level_downsamples == reader.backend.level_downsamples
    assert str(reader) == f"SlideReader(path={reader.path}, backend=PILLOW)"


def test_reader_methods_backend() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    read_zero_sized_region(reader)
    read_region_from_all_levels(reader)
    read_invalid_level(reader)


def test_get_level_methods() -> None:
    reader = SlideReader(SLIDE_PATH_CZI)
    #  0: (134009, 148428)
    #  1: (67004, 74214)
    #  2: (33502, 37107)
    #  3: (16751, 18554)
    #  4: (8376, 9277)
    #  5: (4188, 4638)
    #  6: (2094, 2319)
    #  7: (1047, 1160)
    assert reader.level_from_max_dimension(1) == reader.level_count - 1
    assert reader.level_from_dimensions((1, 1)) == reader.level_count - 1
    assert reader.level_from_max_dimension(4000) == 6
    assert reader.level_from_dimensions((5000, 5000)) == 5


def test_tissue_mask() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    threshold, tissue_mask = reader.get_tissue_mask(level=1, sigma=0.5, threshold=200)
    assert tissue_mask.shape == reader.level_dimensions[1]
    assert threshold == 200
    downsample = F.get_downsample(tissue_mask, reader.dimensions)
    assert downsample == reader.level_downsamples[1]
    with pytest.raises(ValueError, match="Threshold should be in range"):
        reader.get_tissue_mask(level=1, sigma=0.5, threshold=300)


def test_tile_coordinates_properties() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(None, width=1024, max_background=0.2)
    assert isinstance(tile_coords, TileCoordinates)
    assert tile_coords.width == 1024
    assert tile_coords.height == 1024
    assert tile_coords.max_background is None
    assert tile_coords.overlap == 0.0
    assert str(tile_coords) == "TileCoordinates(num_tiles=9, shape=(1024, 1024))"


def test_tile_coordinates_mask() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    __, tissue_mask = reader.get_tissue_mask(level=1, threshold=240)
    tile_coords = reader.get_tile_coordinates(
        tissue_mask, width=1024, max_background=0.2
    )
    assert isinstance(tile_coords, TileCoordinates)
    assert tile_coords.coordinates == [(1024, 0, 1024, 1024), (1024, 1024, 1024, 1024)]


def test_tile_coordinates_out_of_bounds() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(None, width=2400, out_of_bounds=True)
    assert tile_coords.coordinates == [
        (0, 0, 2400, 2400),
        (2400, 0, 2400, 2400),
        (0, 2400, 2400, 2400),
        (2400, 2400, 2400, 2400),
    ]


def test_tile_coordinates_no_mask() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(tissue_mask=None, width=1000)
    assert tile_coords.coordinates == [
        (0, 0, 1000, 1000),
        (1000, 0, 1000, 1000),
        (2000, 0, 1000, 1000),
        (0, 1000, 1000, 1000),
        (1000, 1000, 1000, 1000),
        (2000, 1000, 1000, 1000),
        (0, 2000, 1000, 1000),
        (1000, 2000, 1000, 1000),
        (2000, 2000, 1000, 1000),
    ]


def test_spot_coordinates_properties() -> None:
    reader = SlideReader(SLIDE_PATH_TMA)
    __, tissue_mask = reader.get_tissue_mask(level=-1, sigma=2.0, threshold=220)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spot_coords = reader.get_spot_coordinates(tissue_mask)
    assert isinstance(spot_coords, SpotCoordinates)
    assert len(spot_coords) == 94
    assert len(spot_coords.spot_names) == 94
    assert len(spot_coords.coordinates) == 94
    assert str(spot_coords) == "SpotCoordinates(num_spots=94)"


def test_spot_coordinates_good_sigma() -> None:
    reader = SlideReader(SLIDE_PATH_TMA)
    __, tissue_mask = reader.get_tissue_mask(level=-1, sigma=2.0, threshold=220)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        __ = reader.get_spot_coordinates(tissue_mask)


def test_spot_coordinates_bad_sigma() -> None:
    reader = SlideReader(SLIDE_PATH_TMA)
    __, tissue_mask = reader.get_tissue_mask()
    with pytest.warns():
        __ = reader.get_spot_coordinates(tissue_mask)


def test_annotated_thumbnail_tiles() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tiles = reader.get_tile_coordinates(None, width=512)
    thumbnail = reader.get_annotated_thumbnail(reader.read_level(-1), tiles)
    excpected = Image.open(DATA_DIRECTORY / "thumbnail_tiles.png")
    assert np.equal(np.array(thumbnail), np.array(excpected)).all()


def test_annotated_thumbnail_regions() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tiles = reader.get_tile_coordinates(None, width=512)
    thumbnail = reader.get_annotated_thumbnail(reader.read_level(-1), tiles.coordinates)
    excpected = Image.open(DATA_DIRECTORY / "thumbnail_regions.png")
    assert np.equal(np.array(thumbnail), np.array(excpected)).all()


def test_annotated_thumbnail_spots() -> None:
    reader = SlideReader(SLIDE_PATH_TMA)
    __, tissue_mask = reader.get_tissue_mask(level=-1, sigma=2.0, threshold=220)
    spots = reader.get_spot_coordinates(tissue_mask)
    thumbnail = reader.get_annotated_thumbnail(reader.read_level(-2), spots)
    excpected = Image.open(DATA_DIRECTORY / "thumbnail_spots.png")
    assert np.equal(np.array(thumbnail), np.array(excpected)).all()


def test_yield_regions() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(tissue_mask=None, width=512, height=256)
    yielded_coords = []
    for tile, xywh in reader.yield_regions(tile_coords):
        assert tile.shape == (256, 512, 3)
        yielded_coords.append(xywh)
    assert tile_coords.coordinates == yielded_coords


def test_yield_regions_concurrent() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(tissue_mask=None, width=512, height=256)
    yielded_coords = []
    for tile, xywh in reader.yield_regions(tile_coords, num_workers=4):
        assert tile.shape == (256, 512, 3)
        yielded_coords.append(xywh)
    assert tile_coords.coordinates == yielded_coords


def test_yield_regions_nonzero_level() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(tissue_mask=None, width=512, height=256)
    yielded_coords = []
    for tile, xywh in reader.yield_regions(tile_coords, level=1):
        assert tile.shape == (128, 256, 3)
        yielded_coords.append(xywh)
    assert tile_coords.coordinates == yielded_coords


def test_yield_regions_transform() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(tissue_mask=None, width=512, height=256)
    for tile, __ in reader.yield_regions(tile_coords, transform=lambda x: x[..., 0]):
        assert tile.shape == (256, 512)


def test_save_regions() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    metadata = reader.save_regions(TMP_DIRECTORY, regions)
    assert isinstance(metadata, pl.DataFrame)
    assert metadata.columns == ["x", "y", "w", "h", "path"]
    assert [f.name for f in (TMP_DIRECTORY / reader.name).iterdir()] == [
        "thumbnail.jpeg",
        "thumbnail_tiles.jpeg",
        "tiles",
        "metadata.parquet",
    ]
    expected = ["x{}_y{}_w{}_h{}.jpeg".format(*xywh) for xywh in regions]
    assert [
        f.name for f in (TMP_DIRECTORY / reader.name / "tiles").iterdir()
    ] == expected
    clean_temporary_directory()


def test_save_regions_concurrent() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    reader.save_regions(TMP_DIRECTORY, regions, num_workers=4)
    clean_temporary_directory()


def test_save_regions_tiles() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(None, width=512)
    clean_temporary_directory()
    reader.save_regions(TMP_DIRECTORY, tile_coords)
    assert [f.name for f in (TMP_DIRECTORY / reader.name).iterdir()] == [
        "properties.json",
        "thumbnail.jpeg",
        "thumbnail_tiles.jpeg",
        "tiles",
        "metadata.parquet",
    ]
    expected = ["x{}_y{}_w{}_h{}.jpeg".format(*xywh) for xywh in tile_coords]
    assert [
        f.name for f in (TMP_DIRECTORY / reader.name / "tiles").iterdir()
    ] == expected
    clean_temporary_directory()


def test_save_regions_spots() -> None:
    reader = SlideReader(SLIDE_PATH_TMA)
    __, tissue_mask = reader.get_tissue_mask(sigma=2)
    spot_coords = reader.get_spot_coordinates(tissue_mask)
    clean_temporary_directory()
    reader.save_regions(TMP_DIRECTORY, spot_coords)
    assert [f.name for f in (TMP_DIRECTORY / reader.name).iterdir()] == [
        "thumbnail.jpeg",
        "thumbnail_spots.jpeg",
        "spots",
        "metadata.parquet",
    ]
    expected = [
        "{}_x{}_y{}_w{}_h{}.jpeg".format(name, *xywh)
        for name, xywh in zip(spot_coords.spot_names, spot_coords)
    ]
    assert [
        f.name for f in (TMP_DIRECTORY / reader.name / "spots").iterdir()
    ] == expected
    clean_temporary_directory()


def test_save_regions_overwrite() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    clean_temporary_directory()
    (TMP_DIRECTORY / reader.name).mkdir(parents=True)
    # Should pass with an empty directory...
    reader.save_regions(TMP_DIRECTORY, regions, overwrite=False)
    # ... but not with full.
    with pytest.raises(ValueError, match="Output directory exists"):
        reader.save_regions(TMP_DIRECTORY, regions, overwrite=False)
    clean_temporary_directory()


def test_save_regions_no_thumbnails() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    metadata = reader.save_regions(TMP_DIRECTORY, regions, save_thumbnails=False)
    assert isinstance(metadata, pl.DataFrame)
    assert metadata.columns == ["x", "y", "w", "h", "path"]
    assert [f.name for f in (TMP_DIRECTORY / reader.name).iterdir()] == [
        "tiles",
        "metadata.parquet",
    ]


def test_save_regions_with_csv() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    metadata = reader.save_regions(TMP_DIRECTORY, regions, use_csv=True)
    assert isinstance(metadata, pl.DataFrame)
    assert metadata.columns == ["x", "y", "w", "h", "path"]
    assert [f.name for f in (TMP_DIRECTORY / reader.name).iterdir()] == [
        "thumbnail.jpeg",
        "thumbnail_tiles.jpeg",
        "tiles",
        "metadata.csv",
    ]


def test_save_regions_no_threshold() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    clean_temporary_directory()
    with pytest.raises(ValueError, match="Threshold argument is required"):
        reader.save_regions(TMP_DIRECTORY, regions, overwrite=False, save_masks=True)
    with pytest.raises(ValueError, match="Threshold argument is required"):
        reader.save_regions(TMP_DIRECTORY, regions, overwrite=False, save_metrics=True)
    clean_temporary_directory()


def test_save_regions_with_masks() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    clean_temporary_directory()
    metadata = reader.save_regions(
        TMP_DIRECTORY, regions, save_masks=True, threshold=200
    )
    assert "mask_path" in metadata.columns
    expected = ["x{}_y{}_w{}_h{}.png".format(*xywh) for xywh in regions]
    assert [
        f.name for f in (TMP_DIRECTORY / reader.name / "masks").iterdir()
    ] == expected
    clean_temporary_directory()


def test_save_regions_with_metrics() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    clean_temporary_directory()
    metadata = reader.save_regions(
        TMP_DIRECTORY, regions, save_metrics=True, threshold=200
    )
    assert metadata.columns == [
        "x",
        "y",
        "w",
        "h",
        "path",
        "background",
        "black_pixels",
        "white_pixels",
        "laplacian_std",
        "gray_mean",
        "gray_std",
        "red_mean",
        "red_std",
        "green_mean",
        "green_std",
        "blue_mean",
        "blue_std",
        "hue_mean",
        "hue_std",
        "saturation_mean",
        "saturation_std",
        "brightness_mean",
        "brightness_std",
        "gray_q5",
        "gray_q10",
        "gray_q25",
        "gray_q50",
        "gray_q75",
        "gray_q90",
        "gray_q95",
        "red_q5",
        "red_q10",
        "red_q25",
        "red_q50",
        "red_q75",
        "red_q90",
        "red_q95",
        "green_q5",
        "green_q10",
        "green_q25",
        "green_q50",
        "green_q75",
        "green_q90",
        "green_q95",
        "blue_q5",
        "blue_q10",
        "blue_q25",
        "blue_q50",
        "blue_q75",
        "blue_q90",
        "blue_q95",
        "hue_q5",
        "hue_q10",
        "hue_q25",
        "hue_q50",
        "hue_q75",
        "hue_q90",
        "hue_q95",
        "saturation_q5",
        "saturation_q10",
        "saturation_q25",
        "saturation_q50",
        "saturation_q75",
        "saturation_q90",
        "saturation_q95",
        "brightness_q5",
        "brightness_q10",
        "brightness_q25",
        "brightness_q50",
        "brightness_q75",
        "brightness_q90",
        "brightness_q95",
    ]
    clean_temporary_directory()


def test_estimate_mean_and_std() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    mean, std = reader.get_mean_and_std(reader.get_tile_coordinates(None, 512))
    assert mean == (0.8447404647527956, 0.7014915950999541, 0.779204397164139)
    assert std == (0.1367226593250863, 0.18658047647561957, 0.1402206641594302)
