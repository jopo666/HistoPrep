import functools
import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import polars as pl
import tqdm
from PIL import Image

import histoprep.functional as F
from histoprep._backend import CziBackend, OpenSlideBackend, PillowBackend
from histoprep._data import SpotCoordinates, TileCoordinates
from histoprep.functional._concurrent import close_pool, prepare_worker_pool
from histoprep.functional._level import format_level
from histoprep.functional._tiles import _multiply_xywh

ERROR_WRONG_TYPE = "Expected '{}' to be of type {}, not {}."
ERROR_NO_THRESHOLD = "Threshold argument is required to save masks/metrics."
ERROR_AUTOMATIC_BACKEND = (
    "Could not automatically assing reader for path: '{}'. Please choose from {}."
)
ERROR_BACKEND_NAME = "Backend '{}' does not exist, choose from: {}."
ERROR_OUTPUT_DIR_IS_FILE = "Output directory exists but it is a file."
ERROR_CANNOT_OVERWRITE = "Output directory exists, but `overwrite=False`."
AVAILABLE_BACKENDS = ("PILLOW", "OPENSLIDE", "CZI")
OPENSLIDE_READABLE_FORMATS = (
    "svs",
    "vms",
    "vmu",
    "ndpi",
    "scn",
    "mrxs",
    "tiff",
    "svslide",
    "tif",
    "bif",
)


class SlideReader:
    """Reader class for histological slide images."""

    def __init__(
        self,
        path: Union[str, Path],
        backend: Optional[str] = None,
    ) -> None:
        """Initialize `SlideReader` instance.

        Args:
            path: Path to slide image.
            backend: Backend to use for reading slide images. If None, attempts to
                assing the correct backend based on file extension. Defaults to None.

        Raises:
            FileNotFoundError: Path does not exist.
            ValueError: Cannot automatically assign backend for reader.
            ValueError: Backend name not recognised.
        """
        super().__init__()
        self._backend = _read_slide(path=path, backend=backend)

    @property
    def path(self) -> str:
        """Full slide filepath."""
        return self._backend.path

    @property
    def name(self) -> str:
        """Slide filename without an extension."""
        return self._backend.name

    @property
    def suffix(self) -> str:
        """Slide file-extension."""
        return self._backend.suffix

    @property
    def backend_name(self) -> str:
        """Name of the slide reader backend."""
        return self._backend.BACKEND_NAME

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        """Data bounds defined by `xywh`-coordinates at `level=0`.

        Some image formats (eg. `.mrxs`) define a bounding box where image data resides,
        which may differ from the actual image dimensions. `HistoPrep` always uses the
        full image dimensions, but other software (such as `QuPath`) uses the image
        dimensions defined by this data bound.
        """
        return self._backend.data_bounds

    @property
    def dimensions(self) -> tuple[int, int]:
        """Image dimensions (height, width) at `level=0`."""
        return self._backend.dimensions

    @property
    def level_count(self) -> int:
        """Number of slide pyramid levels."""
        return self._backend.level_count

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        """Image dimensions (height, width) for each pyramid level."""
        return self._backend.level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        """Image downsample factors (height, width) for each pyramid level."""
        return self._backend.level_downsamples

    def read_level(self, level: int) -> np.ndarray:
        """Read full pyramid level data.

        Args:
            level: Slide pyramid level to read.

        Raises:
            ValueError: Invalid level argument.

        Returns:
            Array containing image data from `level`.
        """
        return self._backend.read_level(level=level)

    def read_region(
        self, xywh: tuple[int, int, int, int], level: int = 0
    ) -> np.ndarray:
        """Read region based on `xywh`-coordinates.

        Args:
            xywh: Coordinates for the region.
            level: Slide pyramid level to read from. Defaults to 0.

        Raises:
            ValueError: Invalid `level` argument.

        Returns:
            Array containing image data from `xywh`-region.
        """
        return self._backend.read_region(xywh=xywh, level=level)

    def level_from_max_dimension(self, max_dimension: int = 4096) -> int:
        """Find pyramid level with *both* dimensions less or equal to `max_dimension`.
        If one isn't found, return the last pyramid level.

        Args:
            max_dimension: Maximum dimension for the level. Defaults to 4096.

        Returns:
            Slide pyramid level.
        """
        for level, (level_h, level_w) in self.level_dimensions.items():
            if level_h <= max_dimension and level_w <= max_dimension:
                return level
        return list(self.level_dimensions.keys())[-1]

    def level_from_dimensions(self, dimensions: tuple[int, int]) -> int:
        """Find pyramid level which is closest to `dimensions`.

        Args:
            dimensions: Height and width.

        Returns:
            Slide pyramid level.
        """
        height, width = dimensions
        available = []
        distances = []
        for level, (level_h, level_w) in self.level_dimensions.items():
            available.append(level)
            distances.append(abs(level_h - height) + abs(level_w - width))
        return available[distances.index(min(distances))]

    def get_tissue_mask(
        self,
        *,
        level: Optional[int] = None,
        threshold: Optional[int] = None,
        multiplier: float = 1.05,
        sigma: float = 0.0,
    ) -> tuple[int, np.ndarray]:
        """Detect tissue from slide pyramid level image.

        Args:
            level: Slide pyramid level to use for tissue detection. If None, uses the
                `level_from_max_dimension` method. Defaults to None.
            threshold: Threshold for tissue detection. If set, will detect tissue by
                global thresholding. Otherwise Otsu's method is used to find a
                threshold. Defaults to None.
            multiplier: Otsu's method finds an optimal threshold by minimizing the
                weighted within-class variance. This threshold is then multiplied with
                `multiplier`. Ignored if `threshold` is not None. Defaults to 1.0.
            sigma: Sigma for gaussian blurring. Defaults to 0.0.

        Raises:
            ValueError: Threshold not between 0 and 255.

        Returns:
            Threshold and tissue mask.
        """
        level = (
            self.level_from_max_dimension()
            if level is None
            else format_level(level, available=list(self.level_dimensions))
        )
        return F.get_tissue_mask(
            image=self.read_level(level),
            threshold=threshold,
            multiplier=multiplier,
            sigma=sigma,
        )

    def get_tile_coordinates(
        self,
        tissue_mask: Optional[np.ndarray],
        width: int,
        *,
        height: Optional[int] = None,
        overlap: float = 0.0,
        max_background: float = 0.95,
        out_of_bounds: bool = True,
    ) -> TileCoordinates:
        """Generate tile coordinates.

        Args:
            tissue_mask: Tissue mask for filtering tiles with too much background. If
                None, the filtering is disabled.
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
        max_area_pixel: Optional[int] = None,
        min_area_relative: float = 0.2,
        max_area_relative: Optional[float] = 2.0,
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
        spot_info = F.get_spot_coordinates(spot_mask)
        spot_coordinates = [  # upsample to level zero.
            _multiply_xywh(x, F.get_downsample(tissue_mask, self.dimensions))
            for x in spot_info.values()
        ]
        return SpotCoordinates(
            coordinates=spot_coordinates,
            spot_names=list(spot_info.keys()),
            tissue_mask=spot_mask,
        )

    def get_annotated_thumbnail(
        self,
        image: np.ndarray,
        coordinates: Iterator[tuple[int, int, int, int]],
        linewidth: int = 1,
    ) -> Image.Image:
        """Generate annotated thumbnail from coordinates.

        Args:
            image: Input image.
            coordinates: Coordinates to annotate.
            linewidth: Width of rectangle lines.

        Returns:
            Annotated thumbnail.
        """
        kwargs = {
            "image": image,
            "downsample": F.get_downsample(image, self.dimensions),
            "rectangle_width": linewidth,
        }
        if isinstance(coordinates, SpotCoordinates):
            text_items = [x.lstrip("spot_") for x in coordinates.spot_names]
            kwargs.update(
                {"coordinates": coordinates.coordinates, "text_items": text_items}
            )
        elif isinstance(coordinates, TileCoordinates):
            kwargs.update(
                {"coordinates": coordinates.coordinates, "highlight_first": True}
            )
        else:
            kwargs.update({"coordinates": coordinates})
        return F.get_annotated_image(**kwargs)

    def yield_regions(
        self,
        coordinates: Iterator[tuple[int, int, int, int]],
        *,
        level: int = 0,
        transform: Optional[Callable[[np.ndarray], Any]] = None,
        num_workers: int = 1,
        return_exception: bool = False,
    ) -> Iterator[tuple[Union[np.ndarray, Exception, Any], tuple[int, int, int, int]]]:
        """Yield tile images and corresponding xywh coordinates.

        Args:
            coordinates: List of xywh-coordinates.
            level: Slide pyramid level for reading tile images. Defaults to 0.
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
                _read_tile,
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
        *,
        level: int = 0,
        max_samples: int = 1000,
        num_workers: int = 1,
        raise_exception: bool = True,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Calculate mean and std for each image channel.

        Args:
            coordinates: `TileCoordinates` instance or a list of xywh-coordinates.
            level: Slide pyramid level for reading tile images. Defaults to 0.
            max_samples: Maximum tiles to load. Defaults to 1000.
            num_workers: Number of worker processes for yielding tiles. Defaults to 1.
            raise_exception: Whether to raise an exception if there are problems with
                reading tile regions. Defaults to True.

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
        iterable = self.yield_regions(
            coordinates=coordinates,
            level=level,
            num_workers=num_workers,
            return_exception=not raise_exception,
        )
        return F.get_mean_and_std_from_images(
            images=(tile for tile, __ in iterable if not isinstance(tile, Exception))
        )

    def save_regions(
        self,
        parent_dir: Union[str, Path],
        coordinates: Iterator[tuple[int, int, int, int]],
        *,
        level: int = 0,
        threshold: Optional[int] = None,
        overwrite: bool = False,
        save_metrics: bool = False,
        save_masks: bool = False,
        save_thumbnails: bool = True,
        thumbnail_level: Optional[int] = None,
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
            level: Slide pyramid level for extracting xywh-regions. Defaults to 0.
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
            thumbnail_level: Slide pyramid level for thumbnail images. If None, uses the
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
            ValueError: Invalid `level` argument.
            ValueError: Threshold is not between 0 and 255.

        Returns:
            Polars dataframe with metadata.
        """
        if (save_metrics or save_masks) and threshold is None:
            raise ValueError(ERROR_NO_THRESHOLD)
        level = format_level(level, available=list(self.level_dimensions))
        parent_dir = parent_dir if isinstance(parent_dir, Path) else Path(parent_dir)
        output_dir = _prepare_output_dir(parent_dir / self.name, overwrite=overwrite)
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
            Image.fromarray(thumbnail).save(output_dir / "thumbnail.jpeg")
            thumbnail_regions = self.get_annotated_thumbnail(thumbnail, coordinates)
            thumbnail_regions.save(output_dir / f"thumbnail_{image_dir}.jpeg")
            if (
                isinstance(coordinates, (TileCoordinates, SpotCoordinates))
                and coordinates.tissue_mask is not None
            ):
                Image.fromarray(255 - 255 * coordinates.tissue_mask).save(
                    output_dir / "thumbnail_tissue.jpeg"
                )
        metadata = _save_regions(
            output_dir=output_dir,
            iterable=self.yield_regions(
                coordinates=coordinates,
                level=level,
                transform=functools.partial(
                    _load_region_data,
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
            file_prefixes=coordinates.spot_names
            if isinstance(coordinates, SpotCoordinates)
            else None,
            verbose=verbose,
        )
        if use_csv:
            metadata.write_csv(output_dir / "metadata.csv")
        else:
            metadata.write_parquet(output_dir / "metadata.parquet")
        return metadata

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(path={self.path}, "
            f"backend={self._backend.BACKEND_NAME})"
        )


@dataclass
class RegionData:
    """Dataclass representing data for a slide region."""

    image: np.ndarray
    mask: Optional[np.ndarray]
    metrics: dict[str, float]

    def save_data(
        self,
        *,
        image_dir: Path,
        mask_dir: Path,
        xywh: tuple[int, int, int, int],
        quality: int,
        image_format: str,
        prefix: Optional[str],
    ) -> dict[str, float]:
        """Save image (and mask) and return region metadata."""
        metadata = dict(zip("xywh", xywh))
        filename = "x{}_y{}_w{}_h{}".format(*xywh)
        if prefix is not None:
            filename = f"{prefix}_{filename}"
        # Save image.
        image_path = image_dir / f"{filename}.{image_format}"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.image).save(image_path, quality=quality)
        metadata["path"] = str(image_path.resolve())
        # Save mask.
        if self.mask is not None:
            mask_path = mask_dir / f"{filename}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(self.mask).save(mask_path)
            metadata["mask_path"] = str(mask_path.resolve())
        return {**metadata, **self.metrics}


def _read_slide(  # noqa
    path: Union[str, Path], backend: Optional[str] = None
) -> Union[CziBackend, OpenSlideBackend, PillowBackend]:
    """Read slide with the requested backend.

    Args:
        backend: Backend to use for reading slide images. If None, attempts to
            assing the correct backend based on file extension. Defaults to None.

    Raises:
        FileNotFoundError: Path does not exist.
        ValueError: Cannot automatically assign backend for reader.
        ValueError: Backend name not recognised.

    Returns:
        Slide reader backend.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path.resolve()))
    if backend is None:
        # Based on file-extension.
        if path.name.endswith(OPENSLIDE_READABLE_FORMATS):
            return OpenSlideBackend(path)
        if path.name.endswith(("jpeg", "jpg")):
            return PillowBackend(path)
        if path.name.endswith("czi"):
            return CziBackend(path)
        raise ValueError(ERROR_AUTOMATIC_BACKEND.format(path, AVAILABLE_BACKENDS))
    if isinstance(backend, str):
        # Based on backend argument.
        if "PIL" in backend.upper():
            return PillowBackend(path)
        if "OPEN" in backend.upper():
            return OpenSlideBackend(path)
        if "CZI" in backend.upper() or "ZEISS" in backend.upper():
            return CziBackend(path)
    if isinstance(
        backend, (type(CziBackend), type(OpenSlideBackend), type(PillowBackend))
    ):
        return backend(path=path)
    raise ValueError(ERROR_BACKEND_NAME.format(backend, AVAILABLE_BACKENDS))


def _read_tile(
    worker_state: dict,
    xywh: tuple[int, int, int, int],
    *,
    level: int,
    transform: Optional[Callable[[np.ndarray], Any]],
    return_exception: bool,
) -> Union[np.ndarray, Exception, Any]:
    """Parallisable tile reading function."""
    reader = worker_state["reader"]
    try:
        tile = reader.read_region(xywh=xywh, level=level)
    except KeyboardInterrupt:
        raise KeyboardInterrupt from None
    except Exception as catched_exception:  # noqa
        if not return_exception:
            raise catched_exception  # noqa
        return catched_exception
    if transform is not None:
        return transform(tile)
    return tile


def _prepare_output_dir(output_dir: Union[str, Path], *, overwrite: bool) -> Path:
    """Prepare output directory for saving regions."""
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if output_dir.exists():
        if output_dir.is_file():
            raise NotADirectoryError(ERROR_OUTPUT_DIR_IS_FILE)
        if len(list(output_dir.iterdir())) > 0 and not overwrite:
            raise ValueError(ERROR_CANNOT_OVERWRITE)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_region_data(
    image: np.ndarray,
    *,
    save_masks: bool,
    save_metrics: bool,
    threshold: Optional[int],
) -> RegionData:
    """Helper transform to add tissue mask and image metrics during tile loading."""
    tissue_mask = None
    metrics = {}
    if save_masks or save_metrics:
        __, tissue_mask = F.get_tissue_mask(image=image, threshold=threshold)
    if save_metrics:
        metrics = F.get_image_metrics(image=image, tissue_mask=tissue_mask)
    return RegionData(
        image=image, mask=tissue_mask if save_masks else None, metrics=metrics
    )


def _save_regions(
    output_dir: Path,
    iterable: Iterator[RegionData, tuple[int, int, int, int]],
    *,
    desc: str,
    total: int,
    quality: int,
    image_format: str,
    image_dir: str,
    file_prefixes: list[str],
    verbose: bool,
) -> pl.DataFrame:
    """Save region data to output directory.

    Args:
        output_dir: Output directory.
        iterable: Iterable yieldin RegionData and xywh-coordinates.
        desc: For the progress bar.
        total: For the progress bar.
        quality: Quality of jpeg-compression.
        image_format: Image extension.
        image_dir: Image directory name
        file_prefixes: List of file prefixes.
        verbose: Enable progress bar.

    Returns:
        Polars dataframe with metadata.
    """
    progress_bar = tqdm.tqdm(
        iterable=iterable,
        desc=desc,
        disable=not verbose,
        total=total,
    )
    rows = []
    num_failed = 0
    for i, (region_data, xywh) in enumerate(progress_bar):
        if isinstance(region_data, Exception):
            num_failed += 1
            progress_bar.set_postfix({"failed": num_failed}, refresh=False)
        rows.append(
            region_data.save_data(
                image_dir=output_dir / image_dir,
                mask_dir=output_dir / "masks",
                xywh=xywh,
                quality=quality,
                image_format=image_format,
                prefix=None if file_prefixes is None else file_prefixes[i],
            )
        )
    return pl.DataFrame(rows)
