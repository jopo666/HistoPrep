import json
import warnings

import polars as pl
import pytest
from PIL import Image

from histoprep import SlideReader
from histoprep.backend import TileCoordinates, TissueMask, TMASpotCoordinates

from .utils import (
    SLIDE_PATH_JPEG,
    SLIDE_PATH_TMA,
    TMP_DIRECTORY,
    clean_temporary_directory,
)


def test_tissue_mask() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tissue_mask = reader.get_tissue_mask(level=1, sigma=0.5, threshold=200)
    assert isinstance(tissue_mask, TissueMask)
    assert tissue_mask.mask.shape == reader.level_dimensions[1]
    assert tissue_mask.level == 1
    assert tissue_mask.sigma == 0.5
    assert tissue_mask.threshold == 200
    assert tissue_mask.to_pil().size == reader.level_dimensions[1][::-1]
    assert tissue_mask.read_region(xywh=(0, 0, 100, 80)).shape == (40, 50)
    assert tissue_mask.read_region(xywh=(0, 0, 100, 80), shape=(80, 100)).shape == (
        80,
        100,
    )


def test_tile_coords() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tissue_mask = reader.get_tissue_mask(level=1, threshold=240)
    tile_coords = reader.get_tile_coordinates(
        tissue_mask, width=512, max_background=0.2
    )
    assert isinstance(tile_coords, TileCoordinates)
    assert tile_coords.num_tiles == 4
    assert tile_coords.width == 512
    assert tile_coords.height == 512
    assert tile_coords.coordinates == [
        (1536, 0, 512, 512),
        (512, 512, 512, 512),
        (1024, 512, 512, 512),
        (1536, 512, 512, 512),
    ]
    assert tile_coords.max_background == 0.2
    assert tile_coords.overlap == 0.0
    assert isinstance(tile_coords.tissue_mask, TissueMask)
    assert isinstance(tile_coords.thumbnail, Image.Image)
    assert isinstance(tile_coords.thumbnail_tiles, Image.Image)
    assert isinstance(tile_coords.thumbnail_tissue, Image.Image)
    clean_temporary_directory()
    tile_coords.save_thumbnails(TMP_DIRECTORY)
    tile_coords.save_properties(
        TMP_DIRECTORY, level=0, level_downsample=reader.level_downsamples[0]
    )
    assert (TMP_DIRECTORY / "thumbnail.jpeg").exists()
    assert (TMP_DIRECTORY / "thumbnail_tiles.jpeg").exists()
    assert (TMP_DIRECTORY / "thumbnail_tissue.jpeg").exists()
    with (TMP_DIRECTORY / "properties.json").open() as f:
        assert json.load(f) == {
            "num_tiles": 4,
            "level": 0,
            "level_downsample": [1.0, 1.0],
            "width": 512,
            "height": 512,
            "overlap": 0.0,
            "max_background": 0.2,
            "threshold": 240,
            "sigma": 0.0,
        }
    clean_temporary_directory()


def test_spot_coordinates() -> None:
    reader = SlideReader(SLIDE_PATH_TMA)
    tissue_mask = reader.get_tissue_mask(level=-1, sigma=2.0, threshold=220)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spot_coords = reader.get_spot_coordinates(tissue_mask)
    assert isinstance(spot_coords, TMASpotCoordinates)
    assert spot_coords.num_spots == 38
    assert len(spot_coords.names) == 38
    assert len(spot_coords.coordinates) == 38
    assert isinstance(spot_coords.tissue_mask, TissueMask)
    assert isinstance(spot_coords.thumbnail, Image.Image)
    assert isinstance(spot_coords.thumbnail_spots, Image.Image)
    assert isinstance(spot_coords.thumbnail_tissue, Image.Image)
    clean_temporary_directory()
    spot_coords.save_thumbnails(TMP_DIRECTORY)
    spot_coords.save_properties(
        TMP_DIRECTORY, level=0, level_downsample=reader.level_downsamples[0]
    )
    assert (TMP_DIRECTORY / "thumbnail.jpeg").exists()
    assert (TMP_DIRECTORY / "thumbnail_spots.jpeg").exists()
    assert (TMP_DIRECTORY / "thumbnail_tissue.jpeg").exists()
    with (TMP_DIRECTORY / "properties.json").open() as f:
        assert json.load(f) == {
            "num_spots": 38,
            "level": 0,
            "level_downsample": [1.0, 1.0],
            "threshold": 220,
            "sigma": 2.0,
        }
    clean_temporary_directory()


def test_save_tiles() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tissue_mask = reader.get_tissue_mask(level=1, threshold=240)
    tile_coords = reader.get_tile_coordinates(
        tissue_mask, width=512, max_background=0.2
    )
    clean_temporary_directory()
    metadata = reader.save_tiles(TMP_DIRECTORY, tile_coords, overwrite=False)
    with pytest.raises(ValueError):
        __ = reader.save_tiles(TMP_DIRECTORY, tile_coords, overwrite=False)
    assert isinstance(metadata, pl.DataFrame)
    assert metadata.columns == [
        "path",
        "x",
        "y",
        "w",
        "h",
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
    assert [f.name for f in (TMP_DIRECTORY / "slide").iterdir()] == [
        "thumbnail.jpeg",
        "thumbnail_tiles.jpeg",
        "thumbnail_tissue.jpeg",
        "properties.json",
        "tiles",
        "metadata.parquet",
    ]
    assert [f.name for f in (TMP_DIRECTORY / "slide" / "tiles").iterdir()] == [
        "x1536_y0_w512_h512.jpeg",
        "x512_y512_w512_h512.jpeg",
        "x1024_y512_w512_h512.jpeg",
        "x1536_y512_w512_h512.jpeg",
    ]
    metadata = reader.save_tiles(
        TMP_DIRECTORY,
        tile_coords,
        overwrite=True,
        save_metrics=False,
        save_masks=True,
        use_csv=True,
    )
    assert [f.name for f in (TMP_DIRECTORY / "slide").iterdir()] == [
        "thumbnail.jpeg",
        "thumbnail_tiles.jpeg",
        "thumbnail_tissue.jpeg",
        "properties.json",
        "tiles",
        "masks",
        "metadata.csv",
    ]
    assert [f.name for f in (TMP_DIRECTORY / "slide" / "masks").iterdir()] == [
        "x1536_y0_w512_h512.png",
        "x512_y512_w512_h512.png",
        "x1024_y512_w512_h512.png",
        "x1536_y512_w512_h512.png",
    ]
    assert metadata.columns == ["path", "mask_path", "x", "y", "w", "h"]
    clean_temporary_directory()


def test_save_spots() -> None:
    # Uses the same implementation so most of the above tests cover it!
    reader = SlideReader(SLIDE_PATH_TMA)
    tissue_mask = reader.get_tissue_mask(level=-1, threshold=220, sigma=2)
    spot_coords = reader.get_spot_coordinates(tissue_mask)
    clean_temporary_directory()
    metadata = reader.save_spots(TMP_DIRECTORY, coordinates=spot_coords, level=-1)
    assert isinstance(metadata, pl.DataFrame)
    assert metadata.columns == [
        "name",
        "path",
        "x",
        "y",
        "w",
        "h",
    ]
    assert [f.name for f in (TMP_DIRECTORY / "slide_tma").iterdir()] == [
        "thumbnail.jpeg",
        "thumbnail_spots.jpeg",
        "thumbnail_tissue.jpeg",
        "properties.json",
        "spots",
        "metadata.parquet",
    ]
    assert (
        len([f.name for f in (TMP_DIRECTORY / "slide_tma" / "spots").iterdir()])
        == spot_coords.num_spots
    )
    # Collect all filenames that should be present.
    filenames = set()
    for x, y, w, h in spot_coords.coordinates:
        filenames.add(f"x{x}_y{y}_w{w}_h{h}.jpeg")
    assert (
        len(
            [
                f.name
                for f in (TMP_DIRECTORY / "slide_tma" / "spots").iterdir()
                if f.name not in filenames
            ]
        )
        == 0
    )
    clean_temporary_directory()


def test_yield_tiles() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tissue_mask = reader.get_tissue_mask()
    tile_coords = reader.get_tile_coordinates(tissue_mask, 512, max_background=1.0)
    for i, (tile, xywh) in enumerate(reader.yield_tiles(tile_coords)):
        assert tile.shape == (512, 512, 3)
        assert tile_coords.coordinates[i] == xywh
    for tile, __ in reader.yield_tiles(tile_coords, transform=lambda x: x[..., 0]):
        assert tile.shape == (512, 512)
    for tile, __ in reader.yield_tiles(tile_coords, level=1):
        assert tile.shape == (256, 256, 3)
