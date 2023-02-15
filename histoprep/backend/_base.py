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
from histoprep.data import TileCoordinates, TissueMask, TMASpotCoordinates

from ._save import prepare_output_dir, worker_init, worker_save_region


class SlideReadingError(Exception):
    """Exception class for failures during slide reading."""


ERROR_BACKGROUND_PERCENTAGE = "Background percentage should be between [0, 1]."
ERROR_NO_TISSUE_MASK = (
    "Background percentage requires that `tissue_mask` is defined. "
    "If you wish not to filter coordinates based on background percentage, "
    "set `max_backgroud=None`"
)
ERROR_WRONG_TYPE = "Expected '{}' to be of type {}, not {}."
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
        slide_height, slide_width = self.level_dimensions[level]
        # Check allowed width and height.
        x, y, w, h = xywh
        if y > slide_height and x > slide_width:
            # Both out of bounds, return empty image.
            return np.zeros((h, w), dtype=np.uint8) + 255
        if y + h > slide_height or x + w > slide_width:
            # Either w or h goes out of bounds, read allowed area.
            allowed_h = max(0, min(slide_height - y, h))
            allowed_w = max(0, min(slide_width - x, w))
            data = self._read_region(xywh=(x, y, allowed_w, allowed_h), level=level)
            # Pad image to requested size.
            image = np.zeros((h, w, *data.shape[2:]), dtype=np.uint8) + 255
            image[:allowed_h, :allowed_w] = data
            return image
        return self._read_region(xywh=(x, y, w, h), level=level)

    def get_tissue_mask(
        self,
        *,
        level: Optional[int] = None,
        max_dimension: int = 4096,
        threshold: Optional[int] = None,
        multiplier: float = 1.05,
        sigma: float = 0.0,
        ignore_white: bool = True,
        ignore_black: bool = True,
    ) -> TissueMask:
        """Detect tissue from slide level image.

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
            sigma: Sigma for gaussian blurring. Defaults to 0.0.
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
        max_background: float = 0.95,
        out_of_bounds: bool = True,
    ) -> TileCoordinates:
        """Generate tile coordinates.

        Args:
            tissue_mask: `TissueMask` for filtering tiles with too much background.
            width: Width of a tile.
            height: Height of a tile. If None, will be set to `width`. Defaults to None.
            overlap: Overlap between neighbouring tiles. Defaults to 0.0.
            max_background: Maximum amount of background in tiles. Defaults to 0.95.
            out_of_bounds: Keep tiles which contain regions outside of the image.
                Defaults to True.

        Returns:
            `TileCoordinates` instance.
        """
        # Check arguments.
        if not isinstance(tissue_mask, TissueMask):
            raise TypeError(
                ERROR_WRONG_TYPE.format("tissue_mask", TissueMask, type(tissue_mask))
            )
        if not 0 <= max_background <= 1:
            raise ValueError(ERROR_BACKGROUND_PERCENTAGE)
        # Filter tiles based on background.
        tile_coords = []
        for xywh in F.get_tile_coordinates(
            dimensions=self.dimensions,
            width=width,
            height=height,
            overlap=overlap,
            out_of_bounds=out_of_bounds,
        ):
            tile_mask = tissue_mask.read_region(
                xywh=F.multiply_xywh(xywh, tissue_mask.level_downsample)
            )
            if (
                tile_mask.size > 0
                or (tile_mask == 0).sum() / tile_mask.size <= max_background
            ):
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
        self, tissue_mask: TissueMask, min_area: float = 0.2, max_area: float = 2.0
    ) -> TMASpotCoordinates:
        """Dearray tissue microarray -spots based on a tissue mask of the spots.

        Tissue mask can be obtained with `get_tissue_mask` method and by increasing the
        sigma value removes most of the unwanted tissue fragements/artifacts. Rest
        can be handled with `min_area` and `max_area` arguments.

        Args:
            tissue_mask: Tissue mask of TMA-slide.
            min_area: Minimum contour area, defined by `median(areas) * min_area`.
                Defaults to 0.2.
            max_area: Maximum contour area, defined by `median(areas) * max_area`.
                Defaults to 2.0.

        Returns:
            `TMASpotCoordinates` instance.
        """
        # Get spot info.
        spot_info = F.dearray_tma(
            tissue_mask.mask, min_area=min_area, max_area=max_area
        )
        spot_names = list(spot_info.keys())
        # Generate thumbnails.
        thumbnail = Image.fromarray(self.read_level(level=tissue_mask.level))
        thumbnail_spots = F.draw_tiles(
            image=thumbnail,
            coordinates=list(spot_info.values()),
            text_items=spot_names,
            downsample=tissue_mask.level_downsample,
        )
        thumbnail_tissue = tissue_mask.to_pil()
        # Upsample coords to get lvl zero xywh.
        lvl_upsample = [1 / x for x in tissue_mask.level_downsample]
        lvl_zero_coords = [F.multiply_xywh(x, lvl_upsample) for x in spot_info.values()]
        return TMASpotCoordinates(
            num_spots=len(spot_info),
            coordinates=lvl_zero_coords,
            names=spot_names,
            tissue_threshold=tissue_mask.threshold,
            tissue_sigma=tissue_mask.sigma,
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
        save_metrics: bool = False,
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
            level: Slide level for extracting XYWH-regions. Defaults to 0.
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

        Returns:
            Polars dataframe with metadata.
        """
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
        save_metrics: bool = False,
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
            coordinates: `TMASpotCoordinates` instance.
            level: Slide level for extracting XYWH-regions. Defaults to 0.
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

        Returns:
            Polars dataframe with metadata.
        """
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
        level = self._check_and_format_level(level)
        output_dir = prepare_output_dir(parent_dir, self.slide_name, overwrite)
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
            image_dir=image_dir,
            raise_exception=raise_exception,
        )
        # Save regions and collect metadata.
        with WorkerPool(n_jobs=num_workers, use_worker_state=True) as pool:
            progbar = tqdm.tqdm(
                pool.imap(save_fn, coordinates.coordinates, worker_init=init_fn),
                desc=self.slide_name,
                disable=not verbose,
                total=len(coordinates),
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
                        result = {"name": image_names[idx], **result}
                    metadata_dicts.append(result)
        # Save metadata.
        metadata = pl.from_dicts(metadata_dicts)
        if use_csv:
            metadata.write_csv(output_dir / "metadata.csv")
        else:
            metadata.write_parquet(output_dir / "metadata.parquet")
        return metadata

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"
