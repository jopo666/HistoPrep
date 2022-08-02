import functools
import logging
import os
import warnings
from typing import Callable, Dict, Iterable, List, Tuple, Union

import numpy
import pandas
from PIL import Image

from .. import functional as F
from ..helpers._files import get_extension, remove_extension
from ..helpers._multiprocess import multiprocess_loop
from ..helpers._verbose import progress_bar, verbose_fn
from ._backend import (
    OPENSLIDE_READABLE,
    PILLOW_READABLE,
    READABLE_FORMATS,
    ZEISS_READABLE,
    OpenSlideBackend,
    PillowBackend,
    ZeissBackend,
)
from ._utils import (
    annotate_thumbnail,
    check_level,
    check_region,
    find_level,
    prepare_paths_and_directories,
    save_spot_worker,
    save_tile_worker,
    worker_initializer_fn,
)

__all__ = ["SlideReader"]

ALLOWED_IMAGE_FORMATS = ("png", "jpeg", "jpg")


class SlideReader(object):
    def __init__(
        self,
        path: str,
        verbose: bool = False,
        preferred_dimension: int = 4096,
        threshold: int = None,
        threshold_multiplier: float = 1.05,
        max_dimension: int = 2**14,
    ):
        """Reader for large slide images with extended functionality such as
        tissue detection, tile coordinate extraction, dearraying TMA spots and
        saving tiles.

        Args:
            path: Path to slide image.
            verbose: Set to True to get verbose output. Defaults to False.
            preferred_dimension: Preferred maximum dimension for the for the
                generated thumbnail. Defaults to 4096.
            threshold: Treshold for tissue detection. If set, will detect tissue
                by global thresholding, and otherwise Otsu's method is used to
                find a threshold. Defaults to None.
            threshold_multiplier: Otsu's method is used to find an optimal
                threshold by minimizing the weighted within-class variance. This
                threshold is then multiplied with `threshold_multiplier`. Used
                only if `threshold` is None. Defaults to 1.05.
            max_dimension: Maximum dimension for the generated thumbnail.
                Smaller thumbnails are preferred, but in the absence of these,
                the first level with all dimensions less than `max_dimension`
                is chosen. Defaults to 16_384.

        Raises:
            FileNotFoundError: File not found.
            NotADirectoryError: Scratch directory exists, but isn't a directory.
            IOError: File not readable by HistoPrep.

        Usage:
        ```python
        from histoprep import SlideReader

        # Load tile images.
        reader = SlideReader("/path/to/slide.tiff")
        thumbnail = reader.get_thumbnail()
        thresh, tissue_mask = reader.detect_tissue()
        coords = reader.get_tile_coordinates(
            width=512, overlap=0.1, tissue_mask=tissue_mask
        )
        tiles = [reader.read_region(xywh) for xywh in coords]
        ```
        """
        # Check path.
        if not os.path.exists(path):
            raise FileNotFoundError("{} does not exist.".format(path))
        elif not os.path.isfile(path):
            raise IOError("{} is not a file.".format(path))
        # Load slide with the correct backend.
        extension = get_extension(path)
        if extension is None:
            raise IOError("{} file has no extension.".format(path))
        elif extension in OPENSLIDE_READABLE:
            logging.debug("Using OPENSLIDE backend.")
            self.__backend = OpenSlideBackend(path)
        elif extension in PILLOW_READABLE:
            logging.debug("Using PILLOW backend.")
            self.__backend = PillowBackend(path)
        elif extension in ZEISS_READABLE:
            logging.debug("Using ZEISS backend.")
            self.__backend = ZeissBackend(path)
            self.__max_workers = max(1, os.cpu_count() // 2)
        else:
            raise ValueError(
                "Extension {} is not readable with HistoPrep. Supported file "
                "formats are: {}".format(extension, READABLE_FORMATS)
            )
        # Save attributes.
        self.__path = path
        self.__slide_name = remove_extension(os.path.basename(path))
        self.__max_workers = os.cpu_count()
        self.__max_dimension = max_dimension
        # Define verbose function.
        self.__verbose = functools.partial(
            verbose_fn, verbose=verbose, desc=self.__slide_name, color=True
        )
        # Generate thumbnail and detect tissue.
        self.get_thumbnail(preferred_dimension)
        self.detect_tissue(threshold=threshold, multiplier=threshold_multiplier)
        self.__annotated_thumbnail_tiles = None
        self.__annotated_thumbnail_spots = None
        self.__tile_metadata = None
        self.__spot_metadata = None
        self.__spot_mask = None
        self.__spot_info = None

    @property
    def MAX_DIMENSION(self) -> int:
        return self.__max_dimension

    # a setter function
    @MAX_DIMENSION.setter
    def MAX_DIMENSION(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Maximum dimension should be an integer.")
        if value < 0:
            raise ValueError("Maximum dimension should be positive.")
        self.__max_dimension = value

    @property
    def path(self) -> str:
        return self.__path

    @property
    def slide_name(self) -> str:
        return self.__slide_name

    @property
    def backend(self) -> str:
        return self.__backend

    @property
    def channel_order(self) -> str:
        return "XYWH"

    @property
    def dimension_order(self) -> str:
        return "HW"

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Slide image dimensions."""
        return self.__backend.dimensions

    @property
    def level_downsamples(self) -> Dict[int, Tuple[float, float]]:
        """Dimensions for each level."""
        return self.__backend.level_downsamples

    @property
    def level_dimensions(self) -> Dict[int, Tuple[int, int]]:
        """Downsamples for each level."""
        return self.__backend.level_dimensions

    @property
    def tissue_threshold(self) -> int:
        """Threshold for tissue detection."""
        return self.__threshold

    @property
    def thumbnail(self) -> Image.Image:
        """Thumbnail image of the slide."""
        return self.__thumbnail

    @property
    def thumbnail_downsample(self) -> Image.Image:
        """Downsample for the thumbnail image."""
        return self.level_downsamples[self.__thumbnail_level]

    @property
    def tissue_mask(self) -> Image.Image:
        """Tissue mask (0=background, 1=tissue)."""
        return F.arr2pil(1 - self.__tissue_mask, equalize=True)

    @property
    def spot_mask(self) -> Image.Image:
        """TMA spot mask (0=background, 1=spot)."""
        if self.__spot_mask is not None:
            return F.arr2pil(1 - self.__spot_mask, equalize=True)

    @property
    def annotated_thumbnail_tiles(self) -> Image.Image:
        """Thumbnail image annotated with the tile coordinates."""
        return self.__annotated_thumbnail_tiles

    @property
    def annotated_thumbnail_spots(self) -> Image.Image:
        """Thumbnail image annotated with the spot coordinates."""
        return self.__annotated_thumbnail_spots

    @property
    def tile_metadata(self) -> Image.Image:
        """Metadata for saved tiles."""
        return self.__tile_metadata

    @property
    def spot_metadata(self) -> Image.Image:
        """Metadata for TMA spots."""
        return self.__spot_metadata

    def read_region(
        self,
        xywh: Tuple[int, int, int, int],
        level: int = 0,
        return_arr: bool = False,
        fill: int = 255,
    ) -> Union[Image.Image, numpy.ndarray]:
        """Read region from slide.

        Args:
            xywh: Region defined by X, Y, width and height.
            return_arr: Return region as an uint8 array instead of a Pil image.
                Defaults to False.
            fill: Fill overbound areas with this value. If None, raises an error
                when the region goes out of bounds. Defaults to 255.

        Raises:
            ValueError: XYWH, level or fill are not integers.
            ValueError: Negative coordinates.
            ValueError: Width or height less than 1.
            ValueError: Region overbound without a fill value.

        Returns:
            Image of the region.

        Example:
            ```python
            from histoprep import SlideReader

            slide = SlideReader("/path/to/slide.tiff")
            tile_image = slide.read_region(x=0, y=0, w=1024, h=1024)
            ```
        """
        if isinstance(xywh, numpy.ndarray):
            xywh = xywh.tolist()
        if any(not isinstance(z, int) for z in xywh if z):
            raise TypeError("XYWH coordinates should be integers.")
        x, y, width, height = xywh
        if x < 0 or y < 0:
            raise ValueError(
                "Coordinates should be positive, not ({},{}).".format(x, y)
            )
        if width < 1 or height < 1:
            raise ValueError("Width and height should be above zero.")
        if fill and (not isinstance(fill, int) or not 0 <= fill <= 255):
            raise ValueError("Fill value should be an integer in range [0, 255].")
        if level not in self.level_downsamples.keys():
            raise ValueError("Slide does not contain level {}.".format(level))
        # Check XYWH region.
        out_of_bounds, padding = check_region(xywh, self.level_dimensions[level])
        if out_of_bounds:
            # Tile goes out-of-bounds.
            if fill is None:
                raise ValueError(
                    "Region ({}) overbound and fill is None (dimensions: "
                    "{}).".format((x, y, width, height), self.level_dimensions[level])
                )
            if padding is None:
                # Empty tile.
                tile = numpy.zeros((height, width), dtype=numpy.uint8) + fill
            else:
                # Read allowed region.
                allowed_h, allowed_w = padding
                tile = self.__backend.read_region(
                    XYWH=(x, y, allowed_w, allowed_h),
                    level=level,
                )
                # Pad tile.
                tile = numpy.pad(
                    tile,
                    ((0, height - allowed_h), (0, width - allowed_w), (0, 0)),
                    constant_values=fill,
                )
        else:
            # Read region.
            tile = self.__backend.read_region(XYWH=(x, y, width, height), level=level)
        if not return_arr:
            tile = Image.fromarray(tile)
        # Return tile
        return tile

    def get_thumbnail(
        self,
        preferred_dimension: int = 4096,
        level: int = None,
        return_arr: bool = False,
        fill: int = 255,
        return_level: bool = False,
    ) -> Union[numpy.ndarray, Image.Image]:
        """Get thumbnail image of the slide.

        Running this method updates the `thumbnail` attribute.

        Args:
            preferred_dimension: Selects the first level which has both
                dimensions smaller than `preferred_dimension`. If not found,
                selects the smallest thumbnail with all dimensions less than
                `MAX_DIMENSION` attribute. Defaults to 4096.
            level: Overwrites max_dimension and loads a specific level. Defaults
                to None.
            return_arr: Return an uint8 array instead of PIL Image. Defaults to
                False.
            fill: Fill out of bounds regions. Defaults to 255.
            return_level: Return the thumbnail level. Defaults to False.

        Raises:
            ValueError: No levels with dimensions less than `MAX_DIMENSION`.
            ValueError: Level does not exist.

        Returns:
            Thumbnail image.

        Example:
            ```python
            from histoprep import SlideReader

            reader = Slidereader("/path/to/slide.tiff")
            thumbnail = reader.get_thumbnail()
            ```
        """
        if level is None and preferred_dimension is None:
            raise ValueError("Either level or max_dimension must be defined.")
        if level is None:
            level = find_level(
                preferred_dimension,
                self.MAX_DIMENSION,
                self.level_dimensions,
            )
        else:
            check_level(level, self.level_dimensions)
        # Load thumbnail.
        thumbnail = self.__backend.get_thumbnail(level)
        if fill is not None:
            # Fill completely black pixels.
            thumbnail[thumbnail.sum(-1) == 0] = fill
        if not return_arr:
            # Convert to PIL.
            thumbnail = Image.fromarray(thumbnail)
        downsample = self.level_downsamples[level]
        if isinstance(downsample, (tuple, list)):
            downsample = "({})".format(
                ", ".join(["{:.3f}".format(x) for x in downsample])
            )
        else:
            downsample = "{:.3f}".format(downsample)
        self.__verbose(
            "Thumbnail created with downsample {}.".format(downsample),
        )
        # Cache.
        self.__thumbnail = thumbnail
        self.__thumbnail_level = level
        if return_level:
            return level, thumbnail
        else:
            return thumbnail

    def detect_tissue(
        self,
        threshold: int = None,
        multiplier: float = 1.05,
        sigma: float = 1.0,
        remove_white: bool = True,
    ):
        """Detect tissue from slide image.

        Running this method updates the `tissue_mask` attribute.

        Args:
            threshold: Treshold for tissue detection. If set, will detect tissue
                by global thresholding, and otherwise Otsu's method is used to
                find a threshold. Defaults to None.
            multiplier: Otsu's method is used to find an optimal threshold by
                minimizing the weighted within-class variance. This threshold is
                then multiplied with `multiplier`. Used only if `threshold` is
                None. Defaults to 1.0.
            sigma: Sigma for gaussian blurring. Defaults to 1.05.
            remove_white: Changes completely white regions to the second highest
                pixel value, before detecting background. This way Otsu's method
                isn't affected by possibly extreme white areas. Defaults to
                True.

        Returns:
            Binary mask with 0=background and 1=tissue.

        Example:
            ```python
            from histoprep import SlideReader

            reader = SlideReader("path/to/slide")
            thresh, tissue_mask = reader.detect_tissue()
            ```
        """
        # Detect tissue.
        threshold, tissue_mask = F.detect_tissue(
            image=self.__thumbnail,
            threshold=threshold,
            multiplier=multiplier,
            sigma=sigma,
            remove_white=remove_white,
        )
        self.__verbose(
            "Tissue detected with threshold {}.".format(threshold),
        )
        # Cache.
        self.__tissue_mask = tissue_mask
        self.__threshold = threshold
        return threshold, tissue_mask

    def get_tile_coordinates(
        self,
        width: int,
        overlap: float = 0.0,
        height: int = None,
        level: int = 0,
        tissue_mask: numpy.ndarray = None,
        max_background: float = 0.95,
    ) -> List[Tuple[int, int, int, int]]:
        """Get tile coordinates for the slide.

        Divides the slide into tiles based on width, height and overlap. These
        tiles are then filtered based on the amount of background. Running this
        method updates the `annotated_thumbnail_tiles` attribute.

        Args:
            width: Tile width.
            overlap: Overlap between neighbouring tiles. Defaults to 0.0.
            level: Slide level for calculating tile coordinates. Defaults to 0.
            height: Height of a tile. If None, will be set to width. Defaults to
                None.
            tissue_mask: Tissue mask. If None, uses a cached tissue mask.
                Defaults to None.
            max_background: Maximum amount of background in tile. Defaults to
                0.95.

        Returns:
            XYWH coordinates for each tile.
        """
        if level not in self.level_dimensions.keys():
            raise ValueError("Slide does not contain level {}.".format(level))
        # Get dimensions and downsample.
        dimensions = self.level_dimensions[level]
        h_d, w_d = self.level_downsamples[level]
        if height is None:
            height = width
        # Get coordinates.
        coordinates = F.tile_coordinates(
            dimensions,
            width=width,
            height=height,
            overlap=overlap,
        )
        # Filter tiles.
        if tissue_mask is None:
            # Use cached.
            tissue_mask = self.__tissue_mask
        # Get downsample between mask and coords.
        mask_downsample = [b / a for a, b in zip(tissue_mask.shape[:2], dimensions)]
        # Filter coordinates.
        filtered = F.filter_coordinates(
            coordinates=coordinates,
            tissue_mask=tissue_mask,
            max_background=max_background,
            downsample=mask_downsample,
        )
        self.__verbose(
            "Detected {} tiles with <= {:.1f}{} background "
            "(filtered {:.1f}{} of all tiles).".format(
                len(filtered),
                100 * max_background,
                "%",
                100 - 100 * len(filtered) / len(coordinates),
                "%",
            )
        )
        # Get downsample between thumbnail and coords.
        thumbnail_downsample = [
            b / a for a, b in zip(self.thumbnail.size[::-1], dimensions)
        ]
        # Annotate thumbnail and cache it.
        self.__annotated_thumbnail_tiles = annotate_thumbnail(
            thumbnail=self.thumbnail,
            downsample=thumbnail_downsample,
            coordinates=filtered,
        )
        return filtered

    def dearray(
        self,
        kernel_size: int = 5,
        iterations: int = 3,
        min_area: float = 0.25,
        max_area: float = 3.0,
    ) -> pandas.DataFrame:
        """Dearray TMA spots.

        Running this method updates the `annotated_thumbnail_spots` and
        `spot_mask` attributes.

        Args:
            kernel_size: Kernel size for dilate. Defaults to 5.
            iterations: Dilate iterations. Defaults to 3.
            min_area: Minimum area for a spot. Defaults to 0.1 and calculated
                with: `median(areas) * min_area`
            max_area: Maximum area for a spot. Similar to min_area. Defaults to
                3.0.

        Returns:
            Spot metadata.
        """
        # Dearray.
        self.__spot_mask, bboxes, spot_numbers = F.dearray(
            tissue_mask=self.__tissue_mask,
            kernel_size=kernel_size,
            iterations=iterations,
            min_area=min_area,
            max_area=max_area,
        )
        # Figure out downsample.
        h_d, w_d = [
            b / a for a, b in zip(self.__tissue_mask.shape[:2], self.dimensions)
        ]
        # Multiply bboxes to get actual coordinates.
        spot_coordinates = []
        for (x, y, w, h) in bboxes:
            # Multiply with downsample.
            spot_coordinates.append(
                (round(x * w_d), round(y * h_d), round(w * w_d), round(h * h_d))
            )
        # Annotate thumbnail.
        self.__annotated_thumbnail_spots = annotate_thumbnail(
            thumbnail=self.thumbnail,
            downsample=self.thumbnail_downsample,
            coordinates=spot_coordinates,
            numbers=spot_numbers,
        )
        # Create metadata.
        rows = []
        for xywh, number in zip(spot_coordinates, spot_numbers):
            rows.append((self.slide_name, *xywh, number))
        self.__spot_metadata = pandas.DataFrame(rows)
        self.__spot_metadata.columns = [
            "slide_name",
            "x",
            "y",
            "w",
            "h",
            "spot_number",
        ]
        # Create simple info for saving.
        self.__spot_info = []
        for __, row in self.spot_metadata.iterrows():
            self.__spot_info.append(row[["x", "y", "w", "h", "spot_number"]].tolist())
        return self.__spot_metadata

    def __prepare_for_saving_images(
        self,
        output_dir: str,
        overwrite: bool = False,
        image_format: str = "jpeg",
        quality: int = 95,
        num_workers: int = None,
    ):
        # Check image formats.
        if image_format.lower() not in ALLOWED_IMAGE_FORMATS:
            raise ValueError(
                "Image format %s not allowed. Select from %s."
                % (image_format, ALLOWED_IMAGE_FORMATS)
            )
        # Check quality.
        if image_format.lower() in {"jpeg", "jpg"} and not 0 < quality <= 100:
            raise ValueError("Image quality should be in range (0,100].")
        # Define number of workers.
        num_workers = os.cpu_count() if num_workers is None else num_workers
        num_workers = max(min(self.__max_workers, num_workers), 1)
        # Define outputs paths and directories.
        output_paths = prepare_paths_and_directories(
            output_dir=output_dir,
            slide_name=self.slide_name,
            overwrite=overwrite,
        )
        return num_workers, output_paths

    def __save_images(
        self,
        worker_function: Callable,
        iterable: Iterable,
        num_workers: int,
        num_retries: int,
        csv_path: str,
        unreadable_path: str,
        display_progress: bool,
    ) -> pandas.DataFrame:
        # Initialize lists for metadata and unreadable regions.
        metadata = []
        failed_regions = []
        # Initialise the main worker.
        worker_initializer_fn(self)
        # Define loop.
        if num_workers < 2:
            # Single process.
            loop = map(worker_function, iterable)
        else:
            # Multiple processes.
            loop = multiprocess_loop(
                func=worker_function,
                iterable=iterable,
                num_workers=num_workers,
                initializer=worker_initializer_fn,
                initializer_args=(self,),
            )
        # Init metadata and failed regions.
        metadata = []
        failed_regions = []
        unreadable = []
        # Save images.
        self.__verbose("Saving images with {} workers.".format(num_workers))
        for logs, (info, meta, elapsed) in progress_bar(
            iterable=loop,
            total=len(iterable),
            log_values=True,
            suppress=not display_progress,
        ):
            if isinstance(meta, Exception):
                # Region couldn't be read, preprocessed or saved.
                logging.debug("Region ({}) raised exception: {}".format(info[:4], meta))
                failed_regions.append((info, meta))
                logs["failed"] = len(failed_regions)
                continue
            # Append to metadata.
            metadata.append(meta)
            # Log elapsed time.
            if elapsed < 1:
                logs["per_image"] = "{:.1f}ms".format(1000 * elapsed)
            else:
                logs["per_image"] = "{:.1f}s".format(elapsed)
        # Attempt to read unreadable regions again.
        if len(failed_regions) > 0 and num_retries > 0:
            self.__verbose(
                "Attempting to read {} failed regions again.".format(
                    len(failed_regions)
                )
            )
            num_workers = num_workers // 2
            if num_workers < 2:
                # Single process.
                retry_loop = map(
                    worker_function,
                    (info for info, __ in failed_regions),
                )
            else:
                # Multiple processes.
                retry_loop = multiprocess_loop(
                    func=worker_function,
                    iterable=(info for info, __ in failed_regions),
                    num_workers=num_workers,
                    initializer=worker_initializer_fn,
                    initializer_args=(self,),
                )
            # Try to save images again.
            for info, meta, __ in progress_bar(
                iterable=retry_loop,
                total=len(failed_regions),
                suppress=not display_progress,
                desc="Retrying failed regions",
            ):
                if isinstance(meta, Exception):
                    for i in range(num_retries):
                        __, meta, __ = worker_function(info)
                        if isinstance(meta, dict):
                            # Success!
                            break
                if isinstance(meta, Exception):
                    unreadable.append((info[:4], meta))
                    logs["unreadable"] = len(unreadable)
                    continue
                # Append to metadata if reading succeeded.
                metadata.append(meta)
            if len(unreadable) > 0:
                self.__verbose(
                    "Couldn't read {} regions. Saving information to {}".format(
                        len(unreadable), unreadable_path
                    )
                )
                unreadable = pandas.DataFrame(unreadable)
                unreadable.columns = ["xywh", "exception"]
                # Save.
                unreadable.to_csv(unreadable_path, index=False)
        # Combine metadata.
        if len(metadata) > 0:
            metadata = pandas.DataFrame(metadata)
            metadata = metadata.sort_values(by=["x", "y"])
            # Save metadata.
            self.__verbose("Saving metadata to {}".format(csv_path))
            metadata.to_csv(csv_path, index=False)
            return metadata
        else:
            warnings.warn("No tiles saved from {}.".format(self.slide_name))

    def save_spots(
        self,
        output_dir: str,
        overwrite: bool = False,
        image_format: str = "jpeg",
        quality: int = 95,
        num_workers: int = os.cpu_count(),
        num_retries: int = 10,
        display_progress: bool = True,
    ) -> pandas.DataFrame:
        """Save TMA spot images from a slide.

        Args:
            output_dir: Output directory for the processed images.
            overwrite: Overwrite any existing images. Defaults to False.
            image_format: Image format. Defaults to "jpeg".
            quality: Quality of JPEG compression. Defaults to 95.
            num_workers: Number of image saving workers. Defaults to the number
                of CPU cores.
            num_retries: Number retries loading any regions, which couldn't be
                saved the first time. Defaults to 10.
            display_progress: Display progress bar. Defaults to True.

        Returns:
            Dataframe with spot bounding boxes.
        """
        # Prepare paths and number of workers.
        num_workers, output_paths = self.__prepare_for_saving_images(
            output_dir=output_dir,
            overwrite=overwrite,
            image_format=image_format,
            quality=quality,
            num_workers=num_workers,
        )
        if output_paths is None:
            warnings.warn(
                "Output directory exists but overwrite=False. " "Not saving images."
            )
            return
        # Load spot metadata.
        if self.__spot_info is None:
            self.__verbose("Dearraying slide with default settings.")
            self.dearray()
        # Create output directory.
        os.makedirs(output_paths["spot_directory"], exist_ok=True)
        self.__verbose("Saving annotated spot thumbnail and spot mask.")
        # Save thumbnail and mask.
        self.thumbnail.save(output_paths["thumbnail"])
        self.annotated_thumbnail_spots.save(output_paths["annotated_spots"])
        self.spot_mask.save(output_paths["spot_mask"])
        # Save spots.
        self.__spot_metadata = self.__save_images(
            worker_function=functools.partial(
                save_spot_worker,
                **{
                    "spot_directory": output_paths["spot_directory"],
                    "image_format": image_format,
                    "quality": quality,
                }
            ),
            iterable=self.__spot_info,
            num_workers=min(num_workers, len(self.__spot_info)),
            num_retries=num_retries,
            csv_path=output_paths["spot_metadata"],
            unreadable_path=output_paths["unreadable_spots"],
            display_progress=display_progress,
        )
        return self.__spot_metadata

    def save_tiles(
        self,
        output_dir: str,
        coordinates: List[Tuple[int, int, int, int]],
        level: int = 0,
        preprocess_metrics: Callable = F.PreprocessMetrics(),
        overwrite: bool = False,
        image_format: str = "jpeg",
        quality: int = 95,
        num_workers: int = os.cpu_count(),
        num_retries: int = 10,
        display_progress: bool = True,
    ):
        """Save tile images from a slide.

        Args:
            output_dir: Output directory for the processed images.
            coordinates: A list of XYWH coordinates for the tile regions.
                Can be easily created with the `SlideReader.get_tile_coordinates()`
                method. **Please note that the level used to create coordinates
                should, match the `level` argument for this function.**
            level: Slide level to read tile images from. This should match the
                level used to create the coordinates! Defaults to 0.
            preprocess_metrics: Any Callable which takes an image as an input
                and returns a dictionary. It is recommended to use the
                `histoprep.PreprocessMetrics` class, which can be modified with
                custom functions. Defaults to `histoprep.PreprocessMetrics()`.
            overwrite: Overwrite any existing images. Defaults to False.
            image_format: Image format. Defaults to "jpeg".
            quality: Quality of JPEG compression. Defaults to 95.
            num_workers: Number of image saving workers. Defaults to the number
                of CPU cores.
            num_retries: Number retries loading any regions, which couldn't be
                saved the first time. Defaults to 10.
            display_progress: Display progress bar. Defaults to True.

        Returns:
            Tile metadata and preprocessing metrics.
        """
        # Check coordinates.
        if len(coordinates) < 1:
            warnings.warn("Empty coordinates.")
            return
        # Prepare paths and number of workers.
        num_workers, output_paths = self.__prepare_for_saving_images(
            output_dir=output_dir,
            overwrite=overwrite,
            image_format=image_format,
            quality=quality,
            num_workers=num_workers,
        )
        if output_paths is None:
            # Directory exists.
            warnings.warn(
                "Output directory exists but overwrite=False. " "Not saving images."
            )
            return
        # Create output directory.
        os.makedirs(output_paths["tile_directory"], exist_ok=True)
        self.__verbose("Saving thumbnails, tissue mask and slide summary.")
        # Save thumbnail and mask.
        self.thumbnail.save(output_paths["thumbnail"])
        self.tissue_mask.save(output_paths["tissue_mask"])
        # Save annotated thumbnail.
        downsample = [
            b / a
            for a, b in zip(self.thumbnail.size[::-1], self.level_dimensions[level])
        ]
        annotate_thumbnail(
            thumbnail=self.thumbnail,
            downsample=downsample,
            coordinates=coordinates,
        ).save(output_paths["annotated_tiles"])
        # Save images.
        self.__tile_metadata = self.__save_images(
            worker_function=functools.partial(
                save_tile_worker,
                **{
                    "level": level,
                    "tile_directory": output_paths["tile_directory"],
                    "tissue_threshold": self.tissue_threshold,
                    "preprocess_metrics": preprocess_metrics,
                    "image_format": image_format,
                    "quality": quality,
                }
            ),
            iterable=coordinates,
            num_workers=num_workers,
            num_retries=num_retries,
            csv_path=output_paths["tile_metadata"],
            unreadable_path=output_paths["unreadable_tiles"],
            display_progress=display_progress,
        )
        return self.__tile_metadata

    def __repr__(self):
        return (
            "{}:"
            "\n  Dimensions: {}"
            "\n  Downsamples: {}"
            "\n  Channel order: {}"
            "\n  Dimension order: {}"
            "\n  Tissue threshold: {}"
            "\n  Backend: {}".format(
                self.slide_name,
                self.dimensions,
                [2**level for level in self.level_downsamples.keys()],
                self.channel_order,
                self.dimension_order,
                self.tissue_threshold,
                self.backend,
            )
        )
