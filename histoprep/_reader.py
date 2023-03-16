from __future__ import annotations

__all__ = ["SlideReader"]

import functools
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from PIL import Image

import histoprep.functional as F
from histoprep._helpers import (
    close_pool,
    load_region_data,
    prepare_output_dir,
    prepare_worker_pool,
    read_slide,
    read_tile,
    save_regions,
)
from histoprep.backend import CziBackend, OpenSlideBackend, PillowBackend
from histoprep.data import SpotCoordinates, TileCoordinates

ERROR_WRONG_TYPE = "Expected '{}' to be of type {}, not {}."
ERROR_NO_THRESHOLD = "Threshold argument is required to save masks/metrics."


class SlideReader:
    def __init__(
        self,
        path: str | Path,
        backend: str | OpenSlideBackend | PillowBackend | CziBackend | None = None,
    ) -> None:
        """Reader class for histological slide images, with a lot of useful
        functionalty.

        Args:
            path: Path to slide image.
            backend: Backend to use for reading slide images. If None, attempts to
                assing the correct backend based on file extension. Defaults to None.
        """
        super().__init__()
        if backend is None or isinstance(backend, str):
            self.backend = read_slide(path=path, backend=backend)
        else:
            self.backend = backend(path=path)

    @property
    def path(self) -> str:
        """Resolved path to slide."""
        return str(self.backend.path.resolve())

    @property
    def name(self) -> str:
        """Path basename without an extension."""
        return self.backend.name

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        """xywh-coordinates at `level=0` defining the area containing data."""
        return self.backend.data_bounds

    @property
    def dimensions(self) -> tuple[int, int]:
        """Image dimensions (height, width) at `level=0`."""
        return self.backend.dimensions

    @property
    def level_count(self) -> int:
        """Number of slide levels."""
        return self.backend.level_count

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        """Image dimensions (height, width) for each level."""
        return self.backend.level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        """Image downsample factors (height, width) for each level."""
        return self.backend.level_downsamples

    def read_level(self, level: int) -> np.ndarray:
        """Read level to memory.

        Args:
            level: Image level to read.

        Raises:
            ValueError: Invalid level argument.

        Returns:
            Array containing image data for the level.
        """
        return self.backend.read_level(level=level)

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        """Read region based on xywh-coordinates.

        Args:
            xywh: Coordinates for the region.
            level: Slide level to read from.

        Raises:
            ValueError: Invalid level argument.

        Returns:
            Array containing image data from the region.
        """
        return self.backend.read_region(xywh=xywh, level=level)

    def level_from_max_dimension(self, max_dimension: int = 4096) -> int:
        """Find level with both dimensions less or equal to `max_dimension`.

        Args:
            max_dimension: Maximum dimension for the level. Defaults to 4096.

        Returns:
            Level with both dimensions less than `max_dimension`, or the smallest level.
        """
        return F._level_from_max_dimension(
            max_dimension=max_dimension, level_dimensions=self.level_dimensions
        )

    def level_from_dimensions(self, dimensions: tuple[int, int]) -> int:
        """Find level which is closest to `dimensions`.

        Args:
            dimensions: Height and width to match.

        Returns:
            Level which is closest to `dimensions`.
        """
        return F._level_from_dimensions(
            dimensions=dimensions, level_dimensions=self.level_dimensions
        )

    def get_tissue_mask(
        self,
        *,
        level: int | None = None,
        threshold: int | None = None,
        multiplier: float = 1.05,
        sigma: float = 0.0,
    ) -> tuple[int, np.ndarray]:
        """Detect tissue from slide level image.

        Args:
            level: Slide level to use for tissue detection. If None, uses the
                `level_from_max_dimension` method. Defaults to None.
            threshold: Threshold for tissue detection. If set, will detect tissue by
                global thresholding, and otherwise Otsu's method is used to find
                a threshold. Defaults to None.
            multiplier: Otsu's method is used to find an optimal threshold by
                minimizing the weighted within-class variance. This threshold is
                then multiplied with `multiplier`. Ignored if `threshold` is not None.
                Defaults to 1.0.
            sigma: Sigma for gaussian blurring. Defaults to 0.0.

        Raises:
            ValueError: Threshold not between 0 and 255.

        Returns:
            Threshold and tissue mask.
        """
        level = (
            self.level_from_max_dimension()
            if level is None
            else F._format_level(level, available=list(self.level_dimensions))
        )
        return F.get_tissue_mask(
            image=self.read_level(level),
            threshold=threshold,
            multiplier=multiplier,
            sigma=sigma,
        )

    def get_tile_coordinates(
        self,
        tissue_mask: np.ndarray | None,
        width: int,
        *,
        height: int | None = None,
        overlap: float = 0.0,
        max_background: float = 0.95,
        out_of_bounds: bool = True,
    ) -> TileCoordinates:
        """Generate tile coordinates.

        Args:
            tissue_mask: Tissue mask for filtering tiles with too much background. Set
                to None if you wish to skip filtering.
            width: Width of a tile.
            height: Height of a tile. If None, will be set to `width`. Defaults to None.
            overlap: Overlap between neighbouring tiles. Defaults to 0.0.
            max_background: Maximum proportion of background in tiles. Ignored if
                `tissue_mask` is None. Defaults to 0.95.
            out_of_bounds: Keep tiles which contain regions outside of the image.
                Defaults to True.

        Raises:
            ValueError: Height and/or width are smaller than 1.
            ValueError: Height and/or width is larger than dimensions.
            ValueError: Overlap is not in range [0, 1).

        Returns:
            `TileCoordinates` dataclass.
        """
        tile_coordinates = F.get_tile_coordinates(
            dimensions=self.dimensions,
            width=width,
            height=height,
            overlap=overlap,
            out_of_bounds=out_of_bounds,
        )
        if tissue_mask is not None:
            all_backgrounds = F.get_background_percentages(
                tile_coordinates=tile_coordinates,
                tissue_mask=tissue_mask,
                downsample=F.get_downsample(tissue_mask, self.dimensions),
            )
            filtered_coordinates = []
            for xywh, background in zip(tile_coordinates, all_backgrounds):
                if background <= max_background:
                    filtered_coordinates.append(xywh)
            tile_coordinates = filtered_coordinates
        return TileCoordinates(
            coordinates=tile_coordinates,
            width=width,
            height=width if height is None else height,
            overlap=overlap,
            max_background=None if tissue_mask is None else max_background,
            tissue_mask=tissue_mask,
        )

    def get_spot_coordinates(
        self,
        tissue_mask: np.ndarray,
        *,
        min_area_pixel: int = 10,
        max_area_pixel: int | None = None,
        min_area_relative: float = 0.2,
        max_area_relative: float | None = 2.0,
    ) -> SpotCoordinates:
        """Generate tissue microarray spot coordinates.

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
            tissue_mask=tissue_mask,
            min_area_pixel=min_area_pixel,
            max_area_pixel=max_area_pixel,
            min_area_relative=min_area_relative,
            max_area_relative=max_area_relative,
        )
        # Dearray spots.
        spot_info = F.dearray_tma(spot_mask)
        spot_coordinates = [  # upsample to level zero.
            F._multiply_xywh(x, F.get_downsample(tissue_mask, self.dimensions))
            for x in spot_info.values()
        ]
        return SpotCoordinates(
            coordinates=spot_coordinates,
            spot_names=list(spot_info.keys()),
            spot_mask=spot_mask,
        )

    def get_annotated_thumbnail(
        self,
        image: np.ndarray,
        coordinates: Iterator[tuple[int, int, int, int]],
    ) -> Image.Image:
        """Generate annotated thumbnail from coordinates.

        Args:
            image: Input image.
            coordinates: Coordinates to annotate.

        Returns:
            Annotated thumbnail.
        """
        kwargs = {
            "image": image,
            "downsample": F.get_downsample(image, self.dimensions),
        }
        if isinstance(coordinates, SpotCoordinates):
            text_items = ([x.lstrip("spot_") for x in coordinates.spot_names],)
            kwargs.update(
                {"coordinates": coordinates.coordinates, "text_items": text_items}
            )
        elif isinstance(coordinates, TileCoordinates):
            kwargs.update(
                {"coordinates": coordinates.coordinates, "highlight_first": True}
            )
        return F.draw_tiles(**kwargs)

    def yield_regions(
        self,
        coordinates: Iterator[tuple[int, int, int, int]],
        *,
        level: int = 0,
        transform: Callable[[np.ndarray], Any] | None = None,
        num_workers: int = 1,
        return_exception: bool = False,
    ) -> Iterator[tuple[np.ndarray | Exception | Any, tuple[int, int, int, int]]]:
        """Yield tile images and corresponding xywh coordinates.

        Args:
            coordinates: List of xywh-coordinates.
            level: Slide level for reading tile image. Defaults to 0.
            transform: Transform function for tile image. Defaults to None.
            num_workers: Number of worker processes. Defaults to 1.
            return_exception: Whether to return exception in case there is a failure to
                read region, instead of raising the exception. Defaults to False.

        Yields:
            Tuple of (possibly transformed) tile image and corresponding
            xywh-coordinate.
        """
        pool, iterable = prepare_worker_pool(
            worker_fn=functools.partial(
                read_tile,
                level=level,
                transform=transform,
                return_exception=return_exception,
            ),
            reader=self,
            iterable_of_args=((x,) for x in coordinates),
            iterable_length=len(coordinates),
            num_workers=num_workers,
        )
        yield from zip(iterable, coordinates)
        close_pool(pool)

    def get_mean_and_std(
        self,
        coordinates: Iterator[tuple[int, int, int, int]],
        level: int = 0,
        max_samples: int = 1000,
        num_workers: int = 1,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Calculate mean and std for each image channel.

        Args:
            coordinates: `TileCoordinates` instance or a list of xywh-coordinates.
            level: Slide level for reading tile image. Defaults to 0.
            max_samples: Maximum tiles to load. Defaults to 1000.
            num_workers: Number of worker processes for yielding tiles. Defaults to 1.

        Returns:
            Tuples of mean and std values for each image channel.
        """
        if isinstance(coordinates, TileCoordinates):
            coordinates = coordinates.coordinates
        if len(coordinates) > max_samples:
            rng = np.random.default_rng()
            coordinates = rng.choice(
                coordinates, size=max_samples, replace=False
            ).tolist()
        return F._get_mean_and_std(
            images=self.yield_regions(
                coordinates=coordinates, level=level, num_workers=num_workers
            )
        )

    def save_regions(
        self,
        parent_dir: str | Path,
        coordinates: Iterator[tuple[int, int, int, int]],
        *,
        level: int = 0,
        threshold: int | None = None,
        overwrite: bool = False,
        save_metrics: bool = False,
        save_masks: bool = False,
        save_thumbnails: bool = True,
        thumbnail_level: int | None = None,
        image_format: str = "jpeg",
        quality: int = 80,
        use_csv: bool = False,
        num_workers: int = 1,
        raise_exception: bool = True,
        verbose: bool = True,
    ) -> pl.DataFrame:
        """Save regions from an iterable of xywh-coordinates.

        Args:
            parent_dir: Parent directory for output. All output is saved to
                `parent_dir/{self.name}/`.
            coordinates: Iterator of xywh-coordinates.
            level: Slide level for extracting xywh-regions. Defaults to 0.
            threshold: Tissue detection threshold. Required when either `save_masks` or
                `save_metrics` is True. Defaults to None.
            overwrite: Overwrite everything in `parent_dir/{slide_name}/` if it exists.
                Defaults to False.
            save_metrics: Save image metrics to metadata, requires that threshold is
                set. Defaults to False.
            save_masks: Save tissue masks as `png` images, requires that threshold is
                set. Defaults to False.
            save_thumbnails: Save slide thumbnail with and without region annotations.
                Defaults to True.
            thumbnail_level: Level for thumbnail images. If None, uses the
                `level_from_max_dimension` method. Ignored when `save_thumbnails=False`.
                Defaults to None.
            image_format: File format for `Pillow` image writer. Defaults to "jpeg".
            quality: JPEG compression quality if `format="jpeg"`. Defaults to 80.
            use_csv: Save metadata to csv-files instead of parquet-files. Defaults to
                False.
            num_workers: Number of data saving workers. Defaults to 1.
            raise_exception: Whether to raise an exception if there are problems with
                reading tile regions. Defaults to True.
            verbose: Enables `tqdm` progress bar. Defaults to True.

        Raises:
            ValueError: Invalid level argument.
            ValueError: Threshold is not between 0 and 255.

        Returns:
            Polars dataframe with metadata.
        """
        if (save_metrics or save_masks) and threshold is None:
            raise ValueError(ERROR_NO_THRESHOLD)
        level = F._format_level(level, available=list(self.level_dimensions))
        parent_dir = parent_dir if isinstance(parent_dir, Path) else Path(parent_dir)
        output_dir = prepare_output_dir(parent_dir / self.name, overwrite=overwrite)
        image_dir = "spots" if isinstance(coordinates, SpotCoordinates) else "tiles"
        # Save properties.
        if isinstance(coordinates, TileCoordinates):
            with (output_dir / "properties.json").open("w") as f:
                json.dump(
                    coordinates.get_properties(
                        level=level, level_downsample=self.level_downsamples[level]
                    ),
                    f,
                )
        # Save thumbnails.
        if save_thumbnails:
            if thumbnail_level is None:
                thumbnail_level = self.level_from_max_dimension()
            thumbnail = self.read_level(thumbnail_level)
            thumbnail_regions = self.get_annotated_thumbnail(thumbnail, coordinates)
            Image.fromarray(thumbnail).save(output_dir / "thumbnail.jpeg")
            thumbnail_regions.save(output_dir / f"thumbnail_{image_dir}.jpeg")
        metadata = save_regions(
            output_dir=output_dir,
            iterable=self.yield_regions(
                coordinates=coordinates,
                level=level,
                transform=functools.partial(
                    load_region_data,
                    save_masks=save_masks,
                    save_metrics=save_metrics,
                    threshold=threshold,
                ),
                num_workers=num_workers,
                return_exception=not raise_exception,
            ),
            desc=self.name,
            total=len(coordinates),
            quality=quality,
            image_format=image_format,
            image_dir=image_dir,
            verbose=verbose,
        )
        if use_csv:
            metadata.write_csv(output_dir / "metadata.csv")
        else:
            metadata.write_parquet(output_dir / "metadata.parquet")
        return metadata

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(path={self.path}, dimensions={self.dimensions},"
            f"backend={self.backend.BACKEND_NAME})"
        )
