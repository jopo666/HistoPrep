import functools
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import histoprep.functional as F
import numpy as np
import polars as pl
import tqdm
from histoprep.data import (
    Properties,
    TileCoordinate,
    TileCoordinates,
    TileImage,
    TissueMask,
)
from mpire import WorkerPool
from PIL import Image

from ._exceptions import TileReadingError

ERROR_BACKGROUND_PERCENTAGE = "Background percentage should be between [0, 1]."
ERROR_TILECOORDS_EMPTY = "Could not save tiles as tile coordinates is empty."
ERROR_OUTPUT_DIR_IS_FILE = "Output directory exists but it is a file."
ERROR_CANNOT_OVERWRITE = "Output directory exists, but `overwrite=False`."


class BaseReader(ABC):
    def __init__(self, path: Union[str, Path]) -> None:
        """Base class for all slide-reader backends.

        Args:
            path: Path to image file.

        Raises:
            FileNotFoundError: Path not found.
            IsADirectoryError: Path is a directory.
        """
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError
        if path.is_dir():
            raise IsADirectoryError
        self.path = path
        # Define slide name by removing extension from basename.
        self.slide_name = path.name.rstrip(path.suffix)

    @property
    @abstractmethod
    def backend(self) -> None:
        pass

    @property
    @abstractmethod
    def data_bounds(self) -> tuple[int, int, int, int]:
        """XYWH-coordinates at `level=0` defining the area containing data."""

    @property
    @abstractmethod
    def dimensions(self) -> tuple[int, int]:
        """Image dimensions (height, width) at `level=0`."""

    @property
    @abstractmethod
    def level_count(self) -> int:
        """Number of slide levels."""

    @property
    @abstractmethod
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        """Image dimensions (height, width) for each level."""

    @property
    @abstractmethod
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        """Image downsample factors (height, width) for each level."""

    @abstractmethod
    def read_level(self, level: int) -> np.ndarray:
        """Read image from a given level to memory.

        Args:
            level: Image level to read.

        Returns:
            Array containing image data for the level.
        """

    @abstractmethod
    def _read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        pass

    def _check_and_format_level(self, level: int) -> int:
        """Check if a `level` exists and format it correctly."""
        available_levels = list(self.level_dimensions.keys())
        if level < 0:
            if abs(level) > len(available_levels):
                error_msg = (
                    f"Could not find level {level} as there are "
                    f"only {len(available_levels)} levels."
                )
                raise ValueError(error_msg)
            level = available_levels[level]
        if level not in available_levels:
            error_msg = (
                f"Level {level} does not exist, choose from: {available_levels}."
            )
            raise ValueError(error_msg)
        return level

    def __level_from_dimension(self, maximum: int) -> int:
        """Get first level where both dimensions are leq to maximum."""
        for level, (level_h, level_w) in self.level_dimensions.items():
            if level_h <= maximum and level_w <= maximum:
                return level
        return -1

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        """Read region based on xywh-coordinates.

        Args:
            xywh: Coordinates for the region.
            level: Slide level to read from.

        Returns:
            Array containing image data from the region.
        """
        # Check that level exists.
        level = self._check_and_format_level(level)
        level_height, level_width = self.level_dimensions[level]
        # Check allowed width and height.
        x, y, w, h = xywh
        if y > level_height and x > level_width:
            # Both out of bounds, return empty image.
            return np.zeros((h, w), dtype=np.uint8) + 255
        if y + h > level_height or x + w > level_width:
            # Either w or h goes out of bounds, read allowed area.
            allowed_h = min(level_height - y, h)
            allowed_w = min(level_width - x, h)
            data = self._read_region(xywh=(x, y, allowed_w, allowed_h), level=level)
            # Pad image to requested size.
            shape = (h, w, *data.shape[2:])
            image = np.zeros(shape, dtype=np.uint8) + 255
            image[:allowed_h, :allowed_w] = data
            return image
        return self._read_region(xywh=(x, y, w, h), level=level)

    def detect_tissue(
        self,
        *,
        level: Optional[int] = None,
        max_dimension: int = 4096,
        threshold: Optional[int] = None,
        multiplier: float = 1.0,
        sigma: float = 1.0,
        ignore_white: bool = True,
        ignore_black: bool = True,
    ) -> TissueMask:
        """Detect tissue from image.

        Args:
            level: Slide level to use for tissue detection. If None, attempts to select
                level based on the `max_dimension` argument. Defaults to None.
            max_dimension: Selects the first slide level, where both dimensions are
                below `max_dimension`. If such level doesn not exist, selects the
                smallest level (-1). Ignored if `level` is not None. Defaults to 4096.
            threshold: Threshold for tissue detection. If set, will detect tissue by
                global thresholding, and otherwise Otsu's method is used to find
                a threshold. Defaults to None.
            multiplier: Otsu's method is used to find an optimal threshold by
                minimizing the weighted within-class variance. This threshold is
                then multiplied with `multiplier`. Ignored if `threshold` is not None.
                Defaults to 1.0.
            sigma: Sigma for gaussian blurring. Defaults to 1.0.
            ignore_white: Does not consider white pixels with Otsu's method. Useful
                for slide images where large areas are artificially set to white.
                Defaults to True.
            ignore_black: Does not consider black pixels with Otsu's method. Useful
                for slide images where large areas are artificially set to black.
                Defaults to True.

        Returns:
            `TissueMask` instance.
        """
        if level is None:
            level = self.__level_from_dimension(maximum=max_dimension)
        level = self._check_and_format_level(level)
        # Detect tissue.
        threshold, tissue_mask = F.detect_tissue(
            image=self.read_level(level),
            threshold=threshold,
            multiplier=multiplier,
            sigma=sigma,
            ignore_white=ignore_white,
            ignore_black=ignore_black,
        )
        return TissueMask(
            mask=tissue_mask,
            threshold=threshold,
            sigma=sigma,
            level=level,
            level_downsample=self.level_downsamples[level],
        )

    def get_tile_coordinates(
        self,
        tissue_mask: TissueMask,
        width: int,
        *,
        height: Optional[int] = None,
        overlap: float = 0.0,
        level: int = 0,
        out_of_bounds: bool = True,
        max_background: float = 0.95,
    ) -> TileCoordinates:
        """Generate tile coordinates.

        Args:
            tissue_mask: `TissueMask` for background percentage calculation.
            width: Width of a tile.
            height: Height of a tile. If None, will be set to `width`. Defaults to None.
            overlap: Overlap between neighbouring tiles. Defaults to 0.0.
            level: Pyramid level for the tiles. Defaults to 0.
            out_of_bounds: Keep tiles which contain regions outside of the image.
                Defaults to True.
            max_background: Maximum amount of background in tiles. Defaults to 0.95.

        Returns:
            `TileCoordinates` instance.
        """
        # Check that level exists and get level downsample.
        level = self._check_and_format_level(level)
        level_downsample = self.level_downsamples[level]
        # Check background percentage.
        if not 0 <= max_background <= 1:
            raise ValueError(ERROR_BACKGROUND_PERCENTAGE)
        # Get xywh-coordinates.
        coordinates = F.get_tile_coordinates(
            dimensions=self.level_dimensions[level],
            width=width,
            height=height,
            overlap=overlap,
            out_of_bounds=out_of_bounds,
        )
        # Collect tile coordinates.
        tile_coordinates = []
        for xywh in coordinates:
            tile_mask = tissue_mask.read_region(xywh=xywh)
            background_perc = (tile_mask == 0).sum() / tile_mask.size
            # Filter tiles with too much background.
            if background_perc >= max_background:
                continue
            tile_coordinates.append(
                TileCoordinate(
                    xywh=xywh,
                    level=level,
                    level_xywh=F.downsample_xywh(xywh, level_downsample),
                    level_downsample=self.level_downsamples[level],
                    tissue_threshold=tissue_mask.threshold,
                    tissue_sigma=tissue_mask.sigma,
                    background_percentage=round(background_perc, 3),
                )
            )
        # Create thumbnail image from tissue mask level.
        thumbnail_pil = Image.fromarray(self.read_level(level=tissue_mask.level))
        # Annotate tiles to the thumbnail.
        thumbnail_tiles_pil = F.draw_tiles(
            thumbnail_pil,
            coordinates=[x.xywh for x in tile_coordinates],
            downsample=self.level_downsamples[tissue_mask.level],
            rectangle_outline="red",
            rectangle_fill=None,
            rectangle_width=2,
            highlight_first=True,
            highlight_outline="blue",
        )
        return TileCoordinates(
            tile_coordinates=tile_coordinates,
            num_tiles=len(tile_coordinates),
            width=width,
            height=width if height is None else height,
            overlap=overlap,
            max_background=max_background,
            tissue_threshold=tissue_mask.threshold,
            thumbnail=thumbnail_pil,
            thumbnail_tiles=thumbnail_tiles_pil,
            thumbnail_tissue=tissue_mask.to_pil(),
        )

    def read_tile(
        self, tile: TileCoordinate, *, skip_metrics: bool = False
    ) -> TileImage:
        """Read tile region, detect tissue and calculate image metrics.

        Args:
            tile: Tile region defined by `TileCoordinate`.
            skip_metrics: Skip image metrics calculation. Defaults to False.

        Returns:
            `TileImage` instance.
        """
        # Read region.
        tile_image = self.read_region(xywh=tile.xywh, level=tile.level)
        # Detect tissue.
        __, tile_mask = F.detect_tissue(
            tile_image, threshold=tile.tissue_threshold, sigma=tile.tissue_sigma
        )
        # Calculate image metrics.
        tile_metrics = (
            {"background": (tile_mask == 1).sum() / tile_mask.size}
            if skip_metrics
            else F.calculate_metrics(image=tile_image, tissue_mask=tile_mask)
        )
        return TileImage(
            image=tile_image,
            xywh=tile.xywh,
            level=tile.level,
            level_downsample=tile.level_downsample,
            level_xywh=tile.level_xywh,
            tissue_mask=tile_mask,
            tissue_threshold=tile.tissue_threshold,
            tissue_sigma=tile.tissue_sigma,
            image_metrics=tile_metrics,
        )

    def save_tiles(
        self,
        parent_dir: Union[str, Path],
        tile_coordinates: TileCoordinates,
        *,
        save_paths: bool = True,
        save_metrics: bool = False,
        save_masks: bool = False,
        overwrite: bool = False,
        raise_exception: bool = True,
        num_workers: int = 1,
        format: str = "jpeg",  # noqa
        quality: int = 80,
        verbose: bool = True,
        **writer_kwargs,
    ) -> None:
        """Save `TileCoordinates` and summary images.

        Args:
            parent_dir: Parent directory for output. All output is saved to
                `parent_dir/{slide_name}/`.
            tile_coordinates: `TileCoordinates` instance.
            save_paths: Adds file paths to `metadata.parquet`. Defaults to True.
            save_metrics: Save image metrics to `metadata.parquet`. Defaults to True.
            save_masks: Save tissue masks as `png` images. Defaults to False.
            overwrite: Overwrite everything in `parent_dir/{slide_name}/` if it exists.
                Defaults to False.
            raise_exception: Whether to raise an `Exception`, or continue saving tiles
                if there are problems with reading tile regions. Defaults to True.
            num_workers: Number of data saving workers. Defaults to 1.
            verbose: Enables `tqdm` progress bar. Defaults to True.
            format: File format for `PIL` image writer. Defaults to "jpeg".
            quality: JPEG compression quality if `format="jpeg"`. Defaults to 80.
            **writer_kwargs: Extra parameters for `PIL` image writer.
        """
        # Prepare output paths.
        paths = prepare_output_paths(
            parent_dir=parent_dir,
            slide_name=self.slide_name,
            suffix=format,
            overwrite=overwrite,
            overwrite_unfinished=overwrite_unfinished,
            save_masks=save_masks,
        )
        # Save summary images and properties.
        tile_coordinates.thumbnail.save(paths["thumbnail"])
        tile_coordinates.thumbnail_tiles.save(paths["thumbnail_tiles"])
        tile_coordinates.thumbnail_tissue.save(paths["thumbnail_tissue"])
        Properties.from_tile_coordinates(tile_coordinates).to_json(paths["properties"])
        # Define worker initialization and tile saving function.
        init_fn = functools.partial(
            worker_init, reader_class=self.__class__, path=self.path
        )
        save_fn = functools.partial(
            worker_save_tile,
            paths=paths,
            save_paths=save_paths,
            save_masks=save_masks,
            save_metrics=save_metrics,
            format=format,
            quality=quality,
            raise_exception=raise_exception,
            **writer_kwargs,
        )
        # Define iterable.
        if num_workers > 1:
            pool = WorkerPool(n_jobs=num_workers, use_worker_state=True)
            iterable = pool.imap(save_fn, tile_coordinates, worker_init=init_fn)
        else:
            iterable = (
                save_fn(worker_state={"reader": self}, tile_coordinate=x)
                for x in tile_coordinates
            )
        # Define progress bar.
        pbar = tqdm.tqdm(
            iterable,
            desc=self.slide_name,
            disable=not verbose,
            total=len(tile_coordinates),
        )
        # Collect metadata.
        metadata_dicts = []
        num_failed = 0
        for result in pbar:
            if isinstance(result, Exception):
                num_failed += 1
                pbar.set_postfix({"failed": num_failed}, refresh=False)
            else:
                metadata_dicts.append(result)
        # Close pool.
        if num_workers > 1:
            pool.close()
        # Save metadata.
        pl.from_dicts(metadata_dicts).write_parquet(paths["metadata"])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"


def prepare_output_paths(
    *,
    parent_dir: Union[str, Path],
    slide_name: str,
    overwrite: bool,
    save_masks: bool,
) -> int:
    """Prepare output paths for `save_tiles` function."""
    if not isinstance(parent_dir, Path):
        parent_dir = Path(parent_dir)
    # Prepare output directory.
    output_dir = parent_dir / slide_name
    if output_dir.exists():
        if output_dir.is_file():
            raise NotADirectoryError(ERROR_OUTPUT_DIR_IS_FILE)
        if not overwrite:
            raise ValueError(ERROR_CANNOT_OVERWRITE)
        shutil.rmtree(output_dir)
    # Define paths.
    paths = {
        "tile_dir": output_dir / "tiles",
        "mask_dir": output_dir / "tissue_masks",
        "thumbnail": output_dir / "thumbnail.jpeg",
        "thumbnail_tiles": output_dir / "thumbnail_tiles.jpeg",
        "thumbnail_tissue": output_dir / "thumbnail_tissue.jpeg",
        "properties": output_dir / "properties.json",
        "metadata": output_dir / "metadata.parquet",
    }
    # Create output directories.
    paths["tiles"].mkdir(parents=True, exist_ok=True)
    if save_masks:
        paths["masks"].mkdir(parents=True, exist_ok=True)
    return paths


def worker_init(worker_state, reader_class, path: Path) -> None:  # noqa
    """Worker initialization function for `worker_save_tile`."""
    worker_state["reader"] = reader_class(path)


def worker_save_tile(
    worker_state: dict,
    tile_coordinate: TileCoordinate,
    *,
    paths: dict[str, Path],
    save_paths: bool,
    save_metrics: bool,
    save_masks: bool,
    raise_exception: bool,
    format: str,  # noqa.
    **writer_kwargs,
) -> None:
    """Worker function to save tile and/or tissue mask images.

    Args:
        worker_state: Worker state containing a separate reader instance.
        tile_coordinate: TileCoodinate instance.
        paths: All output paths.
        save_paths: Save filapaths to metadata.
        save_metrics: Save metrics to metadata.
        save_masks: Save tissue masks.
        format: Format for PIL writer and also file suffix.
        raise_exception: Whether to raise exeption on error.
        **writer_kwargs: Keyword arguments for `PIL` writer.
    """
    # Read tile.
    tile = read_tile_safe(
        reader=worker_state["reader"],
        tile_coordinate=tile_coordinate,
        save_metrics=save_metrics,
        raise_exception=raise_exception,
    )
    if tile is None:
        return None
    # Save tile.
    tile_path = tile.save_tile(
        output_dir=paths["tile_dir"], suffix=format, **writer_kwargs
    )
    if save_masks:
        mask_path = tile.save_mask(output_dir=paths["mask_dir"], **writer_kwargs)
    # Define metadata.
    tile_meta = {k: v for k, v in zip("xywh", tile.xywh)}
    if save_paths:
        tile_meta["tile_path"] = tile_path
        if save_masks:
            tile_meta["mask_path"] = mask_path
    tile_meta.update(tile.image_metrics)
    return tile_meta


def read_tile_safe(
    *,
    reader,  # noqa
    tile_coordinate: TileCoordinate,
    save_metrics: bool,
    raise_exception: bool,
) -> Optional[TileImage]:
    """Read tile region."""
    try:
        return reader.read_tile(tile_coordinate, skip_metrics=not save_metrics)
    except KeyboardInterrupt:
        raise KeyboardInterrupt from None
    except Exception as e:  # noqa
        if raise_exception:
            raise TileReadingError from e
        return None
        raise KeyboardInterrupt from None
    except Exception as e:  # noqa
        if raise_exception:
            raise TileReadingError from e
        return None
        return None
        return None
        return None
