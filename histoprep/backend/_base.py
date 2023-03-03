import functools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
import tqdm
from mpire import WorkerPool
from PIL import Image

import histoprep.functional as F
from histoprep._errors import LevelNotFoundError
from histoprep.data import TileCoordinates, TissueMask, TMASpotCoordinates

from ._save import prepare_output_dir, worker_init, worker_save_region

ERROR_WRONG_TYPE = "Expected '{}' to be of type {}, not {}."
ERROR_TILECOORDS_EMPTY = "Could not save tiles as tile coordinates is empty."
ERROR_OUTPUT_DIR_IS_FILE = "Output directory exists but it is a file."
ERROR_CANNOT_OVERWRITE = "Output directory exists, but `overwrite=False`."


class BaseReader(ABC):
    def __init__(self, path: Union[str, Path]) -> None:
        """Base class for all slide-reader backends.

        Args:
            path: Path to image file.
        """
        self.path = path if isinstance(path, Path) else Path(path)
        self.slide_name = self.path.name.rstrip(self.path.suffix)
        self._read_slide(path=str(path))

    # === Abstract methods ===
    @property
    @abstractmethod
    def backend(self) -> None:
        pass

    @property
    @abstractmethod
    def data_bounds(self) -> tuple[int, int, int, int]:
        """xywh-coordinates at `level=0` defining the area containing data."""

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
    def _read_slide(self, path: str) -> None:
        pass

    @abstractmethod
    def _read_level(self, level: int) -> np.ndarray:
        pass

    @abstractmethod
    def _read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        pass

    # === Public methods ===
    def read_level(self, level: int) -> np.ndarray:
        """Read image from a given level to memory.

        Args:
            level: Image level to read.

        Raises:
            LevelNotFoundError: Invalid level argument.

        Returns:
            Array containing image data for the level.
        """
        return self._read_level(format_level_index(level, list(self.level_dimensions)))

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        """Read region based on xywh-coordinates.

        Args:
            xywh: Coordinates for the region.
            level: Slide level to read from.

        Raises:
            LevelNotFoundError: Invalid level argument.

        Returns:
            Array containing image data from the region.
        """
        level = format_level_index(level, list(self.level_dimensions))
        allowed_xywh = F.allowed_xywh(xywh, self.dimensions)
        __, __, out_w, out_h = F.multiply_xywh(xywh, self.level_downsamples[level])
        if out_w == 0 or out_h == 0:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)
        return F.pad_tile(
            tile=self._read_region(xywh=allowed_xywh, level=level),
            shape=(out_h, out_w),
        )

    def get_tissue_mask(
        self,
        *,
        threshold: Optional[int] = None,
        multiplier: float = 1.05,
        ignore_white: bool = True,
        ignore_black: bool = True,
        sigma: float = 0.0,
        level: Optional[int] = None,
        max_dimension: int = 4096,
    ) -> TissueMask:
        """Detect tissue from slide level image.

        Args:
            threshold: Threshold for tissue detection. If set, will detect tissue by
                global thresholding, and otherwise Otsu's method is used to find
                a threshold. Defaults to None.
            multiplier: Otsu's method is used to find an optimal threshold by
                minimizing the weighted within-class variance. This threshold is
                then multiplied with `multiplier`. Ignored if `threshold` is not None.
                Defaults to 1.0.
            ignore_white: Does not consider white pixels with Otsu's method. Useful
                for slide images where large areas are artificially set to white.
                Defaults to True.
            ignore_black: Does not consider black pixels with Otsu's method. Useful
                for slide images where large areas are artificially set to black.
                Defaults to True.
            sigma: Sigma for gaussian blurring. Defaults to 0.0.
            level: Slide level to use for tissue detection. If None, attempts to select
                level based on the `max_dimension` argument. Defaults to None.
            max_dimension: Selects the first slide level, where both dimensions are
                below `max_dimension`. If such level doesn not exist, selects the
                smallest level (-1). Ignored if `level` is not None. Defaults to 4096.

        Raises:
            LevelNotFoundError: Invalid level argument.
            ValueError: Threshold not between 0 and 255.
            ValueError: Multiplier is negative.
            ValueError: Sigma is negative.

        Returns:
            `TissueMask` dataclass.
        """
        # Check level.
        if level is None:
            level = level_from_max_dimension(max_dimension, self.level_dimensions)
        level = format_level_index(level, list(self.level_dimensions))
        # Detect tissue.
        threshold, tissue_mask = F.get_tissue_mask(
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
        max_background: float = 0.95,
        out_of_bounds: bool = True,
    ) -> TileCoordinates:
        """Generate tile coordinates.

        Args:
            tissue_mask: `TissueMask` for filtering tiles with too much background.
            width: Width of a tile.
            height: Height of a tile. If None, will be set to `width`. Defaults to None.
            overlap: Overlap between neighbouring tiles. Defaults to 0.0.
            max_background: Maximum proportion of background in tiles. Defaults to 0.95.
            out_of_bounds: Keep tiles which contain regions outside of the image.
                Defaults to True.

        Raises:
            TypeError: Tissue mask is not a `TissueMask` dataclass instance.
            ValueError: Height and/or width are not non-zero positive integers.
            ValueError: Height and/or width is larger than dimensions.
            ValueError: Overlap is not in range [0, 1).

        Returns:
            `TileCoordinates` dataclass.
        """
        # Check arguments.
        if not isinstance(tissue_mask, TissueMask):
            raise TypeError(
                ERROR_WRONG_TYPE.format("tissue_mask", TissueMask, type(tissue_mask))
            )
        # Filter tiles based on background.
        tile_coords = []
        for xywh in F.get_tile_coordinates(
            dimensions=self.dimensions,
            width=width,
            height=height,
            overlap=overlap,
            out_of_bounds=out_of_bounds,
        ):
            tile_mask = tissue_mask.read_region(xywh=xywh)
            if (tile_mask == 0).sum() / tile_mask.size <= max_background:
                tile_coords.append(xywh)
        # Create thumbnails.
        thumbnail = Image.fromarray(self.read_level(tissue_mask.level))
        thumbnail_tiles = F.draw_tiles(
            image=thumbnail,
            coordinates=tile_coords,
            downsample=self.level_downsamples[tissue_mask.level],
            highlight_first=True,
        )
        return TileCoordinates(
            num_tiles=len(tile_coords),
            coordinates=tile_coords,
            width=width,
            height=width if height is None else height,
            overlap=overlap,
            max_background=max_background,
            tissue_mask=tissue_mask,
            thumbnail=thumbnail,
            thumbnail_tiles=thumbnail_tiles,
            thumbnail_tissue=tissue_mask.to_pil(),
        )

    def get_spot_coordinates(
        self,
        tissue_mask: TissueMask,
        *,
        min_area_pixel: int = 10,
        max_area_pixel: Optional[int] = None,
        min_area_relative: float = 0.2,
        max_area_relative: Optional[float] = 2.0,
    ) -> TMASpotCoordinates:
        """Detect tissue microarray spots.

        Args:
            tissue_mask: Tissue mask of the slide. It's recommended to increase `sigma`
                value when detecting tissue to remove non-TMA spots from the mask. Rest
                of the areas can be handled with the following arguments.
            min_area_pixel: Minimum pixel area for contours. Defaults to 10.
            max_area_pixel: Maximum pixel area for contours. Defaults to None.
            min_area_relative: Relative minimum contour area, calculated from the median
                contour area after filtering contours with `[min,max]_pixel` arguments
                (`min_area_relative * median(contour_areas)`). Defaults to 0.2.
            max_area_relative: Relative maximum contour area, calculated from the median
                contour area after filtering contours with `[min,max]_pixel` arguments
                (`max_area_relative * median(contour_areas)`). Defaults to 2.0.

        Returns:
            `TMASpotCoordinates` instance.
        """
        spot_mask = F.clean_tissue_mask(
            tissue_mask=tissue_mask.mask,
            min_area_pixel=min_area_pixel,
            max_area_pixel=max_area_pixel,
            min_area_relative=min_area_relative,
            max_area_relative=max_area_relative,
        )
        spot_info = F.dearray_tma(spot_mask)
        spot_names = list(spot_info.keys())
        thumbnail = Image.fromarray(self.read_level(level=tissue_mask.level))
        thumbnail_spots = F.draw_tiles(
            image=thumbnail,
            coordinates=list(spot_info.values()),
            text_items=spot_names,
            downsample=tissue_mask.level_downsample,
        )
        thumbnail_tissue = tissue_mask.to_pil()
        # Upsample coords to get lvl zero xywh.
        level_upsample = [1 / x for x in tissue_mask.level_downsample]
        level_zero_coords = [
            F.multiply_xywh(x, level_upsample) for x in spot_info.values()
        ]
        return TMASpotCoordinates(
            num_spots=len(spot_info),
            coordinates=level_zero_coords,
            names=spot_names,
            tissue_mask=tissue_mask,
            thumbnail=thumbnail,
            thumbnail_spots=thumbnail_spots,
            thumbnail_tissue=thumbnail_tissue,
        )

    def save_tiles(
        self,
        parent_dir: Union[str, Path],
        coordinates: TileCoordinates,
        *,
        level: int = 0,
        overwrite: bool = False,
        save_paths: bool = True,
        save_metrics: bool = True,
        save_masks: bool = False,
        image_format: str = "jpeg",
        quality: int = 80,
        use_csv: bool = False,
        num_workers: int = 1,
        raise_exception: bool = True,
        verbose: bool = True,
    ) -> pl.DataFrame:
        """Save `TileCoordinates` and summary images.

        Args:
            parent_dir: Parent directory for output. All output is saved to
                `parent_dir/{slide_name}/`.
            coordinates: `TileCoordinates` instance.
            level: Slide level for extracting xywh-regions. Defaults to 0.
            overwrite: Overwrite everything in `parent_dir/{slide_name}/` if it exists.
                Defaults to False.
            save_paths: Adds file paths to metadata. Defaults to True.
            save_metrics: Save image metrics to metadata. Defaults to True.
            save_masks: Save tissue masks as `png` images. Defaults to False.
            image_format: File format for `PIL` image writer. Defaults to "jpeg".
            quality: JPEG compression quality if `format="jpeg"`. Defaults to 80.
            use_csv: Save metadata to csv-files instead of parquet-files. Defaults to
                False.
            num_workers: Number of data saving workers. Defaults to 1.
            raise_exception: Whether to raise an `Exception`, or continue saving tiles
                if there are problems with reading tile regions. Defaults to True.
            verbose: Enables `tqdm` progress bar. Defaults to True.

        Raises:
            TypeError: coordinates is not a `TileCoordinates` dataclass instance.
            ValueError: Invalid level argument.

        Returns:
            Polars dataframe with metadata.
        """
        # Check arguments.
        if not isinstance(coordinates, TileCoordinates):
            raise TypeError(
                ERROR_WRONG_TYPE.format(
                    "coordinates", TileCoordinates, type(coordinates)
                )
            )
        return self.__save_data(
            parent_dir=parent_dir,
            coordinates=coordinates,
            level=level,
            overwrite=overwrite,
            save_paths=save_paths,
            save_metrics=save_metrics,
            save_masks=save_masks,
            image_format=image_format,
            quality=quality,
            use_csv=use_csv,
            num_workers=num_workers,
            image_dir="tiles",
            image_names=None,
            raise_exception=raise_exception,
            verbose=verbose,
        )

    def save_spots(
        self,
        parent_dir: Union[str, Path],
        coordinates: TMASpotCoordinates,
        *,
        level: int = 0,
        overwrite: bool = False,
        save_paths: bool = True,
        save_metrics: bool = True,
        save_masks: bool = False,
        image_format: str = "jpeg",
        quality: int = 80,
        use_csv: bool = False,
        num_workers: int = 1,
        raise_exception: bool = True,
        verbose: bool = True,
    ) -> pl.DataFrame:
        """Save `TMASpotCoordinates` and summary images.

        Args:
            parent_dir: Parent directory for output. All output is saved to
                `parent_dir/{slide_name}/`.
            coordinates: `TMASpotCoordinates` instance.
            level: Slide level for extracting xywh-regions. Defaults to 0.
            overwrite: Overwrite everything in `parent_dir/{slide_name}/` if it exists.
                Defaults to False.
            save_paths: Adds file paths to metadata. Defaults to True.
            save_metrics: Save image metrics to metadata. Defaults to True.
            save_masks: Save tissue masks as `png` images. Defaults to False.
            image_format: File format for `PIL` image writer. Defaults to "jpeg".
            quality: JPEG compression quality if `format="jpeg"`. Defaults to 80.
            use_csv: Save metadata to csv-files instead of parquet-files. Defaults to
                False.
            num_workers: Number of data saving workers. Defaults to 1.
            raise_exception: Whether to raise an `Exception`, or continue saving tiles
                if there are problems with reading tile regions. Defaults to True.
            verbose: Enables `tqdm` progress bar. Defaults to True.

        Raises:
            TypeError: coordinates is not a `TMASpotCoordinates` dataclass instance.
            ValueError: Invalid level argument.

        Returns:
            Polars dataframe with metadata.
        """
        # Check arguments.
        if not isinstance(coordinates, TMASpotCoordinates):
            raise TypeError(
                ERROR_WRONG_TYPE.format(
                    "coordinates", TMASpotCoordinates, type(coordinates)
                )
            )
        return self.__save_data(
            parent_dir=parent_dir,
            coordinates=coordinates,
            level=level,
            overwrite=overwrite,
            save_paths=save_paths,
            save_metrics=save_metrics,
            save_masks=save_masks,
            image_format=image_format,
            quality=quality,
            use_csv=use_csv,
            num_workers=num_workers,
            image_dir="spots",
            image_names=coordinates.names,
            raise_exception=raise_exception,
            verbose=verbose,
        )

    def __save_data(
        self,
        parent_dir: Union[str, Path],
        coordinates: Union[TileCoordinates, TMASpotCoordinates],
        *,
        level: int,
        overwrite: bool,
        save_paths: bool,
        save_metrics: bool,
        save_masks: bool,
        image_format: str,
        quality: int,
        use_csv: bool,
        num_workers: int,
        image_dir: str,
        image_names: Optional[list[str]],
        raise_exception: bool,
        verbose: bool,
    ) -> pl.DataFrame:
        """Internal helper function to save regions from slide."""
        # Prepare output dir and save thumbnails & property.
        level = format_level_index(level, list(self.level_dimensions.keys()))
        output_dir = prepare_output_dir(
            parent_dir=parent_dir, slide_name=self.slide_name, overwrite=overwrite
        )
        coordinates.save_thumbnails(output_dir)
        coordinates.save_properties(output_dir, level, self.level_downsamples[level])
        # Prepare init and worker functions.
        init_fn = functools.partial(
            worker_init, reader_class=self.__class__, path=self.path
        )
        save_fn = functools.partial(
            worker_save_region,
            output_dir=output_dir,
            level=level,
            threshold=coordinates.tissue_mask.threshold,
            sigma=coordinates.tissue_mask.sigma,
            save_paths=save_paths,
            save_metrics=save_metrics,
            save_masks=save_masks,
            image_format=image_format,
            quality=quality,
            raise_exception=raise_exception,
            image_dir=image_dir,
        )
        # Define tile saving iterable.
        if num_workers > 1:
            pool = WorkerPool(n_jobs=num_workers, use_worker_state=True)
            iterable = pool.imap(
                func=save_fn,
                iterable_of_args=((x,) for x in coordinates.coordinates),
                iterable_len=len(coordinates),
                worker_init=init_fn,
            )
        else:
            pool = None
            iterable = (save_fn({"reader": self}, x) for x in coordinates.coordinates)
        # Save tiles.
        progbar = tqdm.tqdm(
            iterable, desc=self.slide_name, disable=not verbose, total=len(coordinates)
        )
        num_failed = 0
        metadata_dicts = []
        for idx, result in enumerate(progbar):
            if isinstance(result, Exception):
                num_failed += 1
                progbar.set_postfix({"failed": num_failed}, refresh=False)
            else:
                # Add spot name.
                if image_names is not None:
                    result = {"name": image_names[idx], **result}  # noqa
                metadata_dicts.append(result)
        # Close pool.
        if pool is not None:
            pool.terminate()
        # Save metadata.
        metadata = pl.from_dicts(metadata_dicts if len(metadata_dicts) > 0 else [{}])
        if use_csv:
            metadata.write_csv(output_dir / "metadata.csv")
        else:
            metadata.write_parquet(output_dir / "metadata.parquet")
        return metadata

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"


def format_level_index(index: int, available: list[int]) -> int:
    """Check if a `index` exists in `available` and format it correctly."""
    if index < 0:
        if abs(index) > len(available):
            raise LevelNotFoundError(index, available)
        return available[index]
    if index in available:
        return index
    raise LevelNotFoundError(index, available)


def level_from_max_dimension(
    maximum: int, level_dimensions: dict[tuple[float, float]]
) -> int:
    """Find level where both dimensions are leq to maximum or the smallest level."""
    for level, (level_h, level_w) in level_dimensions.items():
        if level_h <= maximum and level_w <= maximum:
            return level
    return -1
