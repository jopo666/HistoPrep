import sys
import os
from os.path import dirname, join, basename, exists
from typing import List, Tuple, Callable
import itertools
import multiprocessing as mp
from functools import partial
import warnings

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from openslide import OpenSlide

from ._functional import (
    get_thumbnail,
    get_downsamples,
    try_thresholds,
    resize,
    detect_spots,
    get_spots,
    get_theta
)
from .preprocess.functional import preprocess, tissue_mask
from .helpers._utils import remove_extension, remove_images, flatten

__all__ = [
    'Dearrayer'
]


class Dearrayer(object):
    """
    Cut TMA spots from a TMA array slide.

    Args:
        slide_path (str): Path to the TMA slide array. All formats that are
            supported by openslide can be used.
        threshold (int, optional): Threshold value for tissue detection.
            Can be left undefined, in which case Otsu's binarization is used.
            This is not recommended! Values can easily be searched with
            Cutter.try_thresholds() function. Defaults to None.
        downsample (int, optional): Downsample used for the thumbnail. The user
            might have to tweak this value depending on the magnification of the
            slide. The TMA spot detection method is optimized for downsamlpe=64
            when the magnification is 20x. If no spots are found, try adjusting
            the downsample or tweak the spot detection variables with
            ``Dearrayer.try_spot_mask()`` function.  Defaults to 64.
        min_area_multiplier (float, optional): Remove all detected contours that
            have an area smaller than ``median_area*min_area_multiplier``.
            Defaults to 0.2.
        max_area_multiplier (float, optional): Remove all detected contours that
            have an area larger than ``median_area*max_area_multiplier``.
            Defaults to None.
        kernel_size (Tuple[int], optional): Kernel size used during spot
            detection. Defaults to (8, 8).
        create_thumbnail (bool, optional):  Create a thumbnail if downsample is
            not available. Defaults to False.

    Raises:
        IOError: slide_path not found.
        ValueError: downsample is not available and create_thumbnail=False.
    """

    def __init__(
            self,
            slide_path: str,
            threshold: int = None,
            downsample: int = 64,
            min_area_multiplier: float = 0.2,
            max_area_multiplier: float = None,
            kernel_size: Tuple[int] = (8, 8),
            create_thumbnail: bool = False):
        super().__init__()
        # Define openslide reader.
        if not exists(slide_path):
            raise IOError(f'{slide_path} not found.')
        self.openslide_reader = OpenSlide(slide_path)
        # Assing basic stuff that user can see/check.
        self.slide_path = slide_path
        self.slide_name = remove_extension(basename(slide_path))
        self.dimensions = self.openslide_reader.dimensions
        self.downsample = downsample
        self.threshold = threshold
        self.min_area_multiplier = min_area_multiplier
        self.max_area_multiplier = max_area_multiplier
        self.kernel_size = kernel_size
        self._spots_saved = False
        # Get spots.
        self._thumbnail = get_thumbnail(
            slide_path=self.slide_path,
            downsample=self.downsample,
            create_thumbnail=create_thumbnail
        )
        if self._thumbnail is None:
            # Downsample not available.
            raise ValueError(
                f'Thumbnail not available for downsample {self.downsample}. '
                'Please set create_thumbnail=True or select downsample from\n\n'
                f'{self._downsamples()}'
            )
        self.threshold, self._tissue_mask = tissue_mask(
            image=self._thumbnail,
            threshold=self.threshold,
            return_threshold=True
        )
        self._spot_mask = detect_spots(
            mask=self._tissue_mask,
            min_area_multiplier=self.min_area_multiplier,
            max_area_multiplier=self.max_area_multiplier,
            kernel_size=self.kernel_size,
        )
        self._numbers, self._bounding_boxes = get_spots(
            spot_mask=self._spot_mask,
            downsample=self.downsample,
        )
        if self._numbers is None or self._bounding_boxes is None:
            print(
                'No spots detected from the slide! Please try and adjust, '
                'the kernel_size, min_area_multiplier and max_area_multiplier '
                'parameters using the dearrayer.try_spot_mask() function.'
            )
            self.spot_metadata = None
            self._annotate()
        else:
            self.spot_metadata = pd.DataFrame(
                np.hstack((self._numbers.reshape(-1, 1), self._bounding_boxes)))
            self.spot_metadata.columns = ['number', 'x', 'y', 'width', 'height']
            self._annotate()
            if len([x for x in self._numbers if '_' in x]) > 0:
                print(
                    'Some spots were assinged the same number. Please check the '
                    f'annotated thumbnail for slide {self.slide_name}.'
                )

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __len__(self):
        return len(self._bounding_boxes)

    def summary(self):
        """Returns a summary of the dearraying process."""
        print(self._summary())

    def _summary(self, cut=False):
        summary =  (
            f"{self.slide_name}"
            f"\n  Number of TMA spots: {len(self._bounding_boxes)}"
            f"\n  Downsample: {self.downsample}"
            f"\n  Threshold: {self.threshold}"
            f"\n  Min area multiplier: {self.min_area_multiplier}"
            f"\n  Max area multiplier: {self.max_area_multiplier}"
            f"\n  Kernel size: {self.kernel_size}"
            f"\n  Dimensions: {self.dimensions}"
        )
        if cut:
            summary += (
                f"\n  Tile width: {self.width}"
                f"\n  Tile overlap: {self.overlap}"
                f"\n  Max background: {self.max_background}"
            )
        return summary

    def get_thumbnail(self, max_pixels: int = 1_000_000) -> Image.Image:
        """
        Returns an Pillow Image of the thumbnail for inspection.

        Args:
            max_pixels (int, optional): Downsample the image until the image
                has less than max_pixles pixels. Defaults to 1_000_000.

        Returns:
            Image.Image: Thumbnail.
        """
        return resize(self._thumbnail, max_pixels)

    def get_annotated_thumbnail(self,
                                max_pixels: int = 5_000_000) -> Image.Image:
        """
        Returns an Pillow Image of the annotated thumbnail for inspection.

        Args:
            max_pixels (int, optional): Downsample the image until the image
                has less than max_pixles pixels. Defaults to 5_000_000.

        Returns:
            Image.Image: Annotated thumbnail.
        """
        return resize(self._annotated_thumbnail, max_pixels)

    def get_tissue_mask(self, max_pixels: int = 1_000_000) -> Image.Image:
        """
        Returns an Pillow Image of the tissue mask for inspection.

        Args:
            max_pixels (int, optional): Downsample the image until the image
                has less than max_pixles pixels. Defaults to 1_000_000.

        Returns:
            Image.Image: Tissue mask.
        """
        mask = self._tissue_mask
        # Flip for a nicer image.
        mask = 1 - mask
        mask = mask/mask.max()*255
        mask = Image.fromarray(mask.astype(np.uint8))
        return resize(mask, max_pixels)

    def get_spot_mask(self, max_pixels: int = 1_000_000) -> Image.Image:
        """
        Returns an Pillow Image of the TMA spot mask for inspection.

        Args:
            max_pixels (int, optional): Downsample the image until the image
                has less than max_pixles pixels. Defaults to 1_000_000.

        Returns:
            Image.Image: Spot mask.
        """
        mask = self._spot_mask
        # Flip for a nicer image.
        mask = 1 - mask
        mask = mask/mask.max()*255
        mask = Image.fromarray(mask.astype(np.uint8))
        return resize(mask, max_pixels)

    def _annotate(self):
        """Draw bounding boxes and numbers to the thumbnail."""
        fontsize = (self.spot_metadata.width.median()/6000)*70/self.downsample
        self._annotated_thumbnail = self._thumbnail.copy()
        if self.spot_metadata is None:
            return
        else:
            annotated = ImageDraw.Draw(self._annotated_thumbnail)
            # Bounding boxes.
            for i in range(len(self)):
                x, y, w, h = self._bounding_boxes[i]/self.downsample
                annotated.rectangle(
                    [x, y, x+w, y+h], outline='red', width=round(fontsize*5))
            arr = np.array(self._annotated_thumbnail)
            # Numbers.
            for i in range(len(self)):
                x, y, w, h = self._bounding_boxes[i]/self.downsample
                arr = cv2.putText(
                    arr,
                    str(self._numbers[i]),
                    (int(x+10), int(y-10+h)),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontsize,
                    (0, 0, 255),
                    round(fontsize*3),
                    cv2.LINE_AA
                )
            self._annotated_thumbnail = Image.fromarray(arr)

    def try_thresholds(
            self,
            thresholds: List[int] = [250, 240, 230,
                                     220, 200, 190, 180, 170, 160, 150, 140],
            max_pixels=1_000_000) -> Image.Image:
        """
        Try out different thresholds for tissue detection.

        The function prepares tissue masks with given thresholds and slaps them
        all together in one summary image.

        Args:
            thresholds (List[int], optional): Thresholds to try. Defaults to
                [250, 240, 230, 220, 200, 190, 180, 170, 160, 150, 140].
            max_pixels (int, optional): Downsample the image until the image
                has less than max_pixles pixels. Defaults to 1_000_000.

        Returns:
            Image.Image: [description]
        """
        return try_thresholds(thumbnail=self._thumbnail, thresholds=thresholds)

    def try_spot_mask(
            self,
            min_area_multiplier: float = 0.1,
            max_area_multiplier: float = 2,
            kernel_size: Tuple[int] = (5, 5),
            max_pixels: int = 1_000_000) -> Image.Image:
        """
        Try out different values for TMA spot detection.

        Args:
            min_area_multiplier (float, optional): Increase if some of the small
                shit is detected as a spot. Decrease if some spots are missed.
                Defaults to 0.1.
            max_area_multiplier (float, optional): Increase if some spots are
                missed.  Decrease if some large elements are detected as spots.
                Defaults to 2.
            kernel_size (Tuple[int], optional): Increase with a small downsample
             and vice versa. Defaults to (5, 5).
            max_pixels (int, optional): Downsample the image until the image
                has less than max_pixles pixels. Defaults to 1_000_000.

        Returns:
            Image.Image: TMA spot massk.
        """
        mask = detect_spots(
            image=self._thumbnail,
            mask=self._tissue_mask,
            min_area_multiplier=min_area_multiplier,
            max_area_multiplier=max_area_multiplier,
            kernel_size=kernel_size,
        )
        # Flip for a nicer image.
        mask = 1 - mask
        mask = mask/mask.max()*255
        mask = Image.fromarray(mask.astype(np.uint8))
        return resize(mask, max_pixels)

    def _prepare_directories(self, output_dir: str) -> None:
        out_dir = join(output_dir, self.slide_name)
        # Save paths.
        self._thumb_path = join(out_dir, f'thumbnail_{self.downsample}.jpeg')
        self._annotated_path = join(out_dir, 'thumbnail_annotated.jpeg')
        self._spot_meta_path = join(out_dir, 'spot_metadata.csv')
        self._tile_meta_path = join(out_dir, 'metadata.csv')
        self._image_dir = join(out_dir, 'spots')
        self._tile_dir = join(out_dir, 'tiles')
        self._summary_path = join(out_dir, 'summary.txt')

        # Make dirs.
        os.makedirs(self._image_dir, exist_ok=True)

    def save_spots(
            self,
            output_dir: str,
            overwrite: bool = False,
            image_format: str = 'jpeg',
            quality: int = 95) -> pd.DataFrame:
        """
        Save TMA spots, coordinates and spot numbering.

        Args:
            output_dir (str): Parent directory for all output.
            overwrite (bool, optional): This will **remove** all saved images,
                thumbnail and metadata and save images again.. Defaults to
                False.
            image_format (str, optional): Format can be jpeg or png. Defaults
                to 'jpeg'.
            quality (int, optional): For jpeg compression. Defaults to 95.

        Raises:
            ValueError: Invalid image format.

        Returns:
            pd.DataFrame: Coordinates and spot numbers.
        """
        allowed_formats = ['jpeg', 'png']
        if image_format not in allowed_formats:
            raise ValueError(
                'Image format {} not allowed. Select from {}'.format(
                    image_format, allowed_formats
                ))
        self._prepare_directories(output_dir)
        # Check if slide has been cut before.
        if exists(self._thumb_path) and not overwrite:
            print(
                'Spots have already been cut! Please set overwrite=True if you '
                'wish to save them again.'
                )
            self.spot_metadata = pd.read_csv(self._spot_meta_path)
            self._spots_saved = True
            return self.spot_metadata
        elif exists(self._thumb_path) and overwrite:
            # Remove all previous files.
            os.remove(self._annotated_path)
            remove_images(self._image_dir)
        # Save text summary.
        with open(self._summary_path, "w") as f:
            f.write(self._summary())
        # Save both thumbnails.
        self._thumbnail.save(self._thumb_path, quality=95)
        self._annotated_thumbnail.save(self._annotated_path, quality=95)
        # Wrap the saving function so it can be parallized.
        func = partial(save_spot, **{
            'slide_path': self.slide_path,
            'image_dir': self._image_dir,
            'image_format': image_format,
            'quality': quality,
        })
        # Multiprocessing to speed things up!
        data = list(zip(self._numbers, self._bounding_boxes))
        spot_paths = []
        with mp.Pool(processes=os.cpu_count()) as p:
            for filepath in tqdm(
                p.imap(func, data),
                total=len(data),
                desc=self.slide_name,
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            ):
                spot_paths.append(filepath)
        # Finally save metadata.
        self.spot_metadata['path'] = spot_paths
        self.spot_metadata.to_csv(self._spot_meta_path, index=False)
        self._spots_saved = True
        return self.spot_metadata

    def save_tiles(
        self,
        width: int,
        overlap: float = 0.0,
        max_background: float = 0.999,
        overwrite: bool = False,
        image_format: str = 'jpeg',
        quality: int = 95,
        custom_preprocess: Callable[[Image.Image], dict] = None
    ) -> pd.DataFrame:
        """
        Cut tiles from dearrayed TMA spots.

        Args:
            width (int): Tile width.
            overlap (float, optional): Overlap between neighbouring tiles.
                Defaults to 0.0.
            max_background (float, optional): Maximum amount of background
                allowed for a tile. Defaults to 0.999.
            overwrite (bool, optional): This will **remove** the tiles directory
                completely and save all the tiles again. Defaults to False.
            image_format (str, optional): Format can be jpeg or png. Defaults
                to 'jpeg'.
            quality (int, optional): For jpeg compression. Defaults to 95.
            custom_preprocess (Callable[[Image.Image], dict], optional): This is
                intended for users that want to define their own preprocessing
                function. The function must take a Pillow image as an input and
                return a dictionary of desired metrics. Defaults to None.

        Raises:
            ValueError: Invalid image format.
            IOError: Spots have not been saved first with Dearrayer.save().
            IOError: No spot paths found.

        Returns:
            pd.DataFrame: Metadata.
        """
        allowed_formats = ['jpeg', 'png']
        if image_format not in allowed_formats:
            raise ValueError(
                'Image format {} not allowed. Select from {}'.format(
                    image_format, allowed_formats
                ))
        if not self._spots_saved:
            raise IOError('Please save the spots first with Dearrayer.save()')
        if exists(self._tile_dir) and overwrite == False:
            print(
                f'{self._tile_dir} already exists! If you want to save tiles '
                'again please set overwrite=True.'
            )
            return pd.read_csv(self._tile_meta_path)
        else:
            # Create the tiles directory
            os.makedirs(self._tile_dir, exist_ok=True)
            # Update summary.txt
            self.width = width
            self.overlap = overlap
            self.max_background = max_background
            with open(self._summary_path, "w") as f:
                f.write(self._summary(cut=True))
        # Let's collect all spot paths.
        spot_paths = self.spot_metadata['path'].tolist()
        # Remove nan paths.
        spot_paths = [x for x in spot_paths if isinstance(x, str)]
        if len(spot_paths) == 0:
            raise IOError('No spot paths found!')
        # Wrap the saving function so it can be parallized.
        func = partial(save_tile, **{
            'image_dir': self._tile_dir,
            'width': width,
            'overlap': overlap,
            'threshold': self.threshold,
            'max_background': max_background,
            'image_format': image_format,
            'quality': quality,
            'custom_preprocess': custom_preprocess
        })
        # Multiprocessing to speed things up!
        metadata = []
        with mp.Pool(processes=os.cpu_count()) as p:
            for results in tqdm(
                p.imap(func, spot_paths),
                total=len(spot_paths),
                desc='Cutting tiles',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            ):
                metadata.append(results)
        metadata = list(filter(None, metadata))
        metadata = flatten(metadata)
        if len(metadata) == 0:
            print(f'No tiles saved from any of the spots!')
            return None
        # Save metadata.
        self.tile_metadata = pd.DataFrame(metadata)
        self.tile_metadata.to_csv(self._tile_meta_path, index=False)
        return self.tile_metadata


def save_spot(
        data: Tuple[int, Tuple[int, int, int, int]],
        slide_path: str,
        image_dir: str,
        image_format: str,
        quality: int) -> dict:
    """Saves spot as an image (parallizable)."""
    # Unpack variables
    number, (x, y, w, h) = data
    # Load slide as it can't be pickled...
    reader = OpenSlide(slide_path)
    # Prepare filename.
    filepath = join(image_dir, f'spot-{number}')
    if image_format == 'png':
        filepath = filepath + '.png'
    else:
        filepath = filepath + '.jpeg'
    # Load image.
    try:
        image = reader.read_region((x, y), 0, (w, h)).convert('RGB')
    except:
        warnings.warn('Broken slide!')
        return
    # Save image.
    if not exists(filepath):
        image.save(filepath, quality=quality)
    return filepath


def save_tile(
        path: str,
        image_dir: str,
        width: int,
        overlap: float,
        threshold: int,
        max_background: float,
        image_format: str,
        quality: int,
        custom_preprocess: Callable[[Image.Image], dict]) -> dict:
    """Saves tiles from a TMA spot (parallizable)."""
    # Load spot image.
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Collect all the tile coordinates.
    coords = _get_all_coordidates(dimensions=image.shape[:2],
                                  width=width, overlap=overlap)
    # Discard based on tissue_mask
    mask = tissue_mask(image=image, threshold=threshold)
    coords = _filter_coordinates(coords, mask, width, max_background)
    if len(coords) == 0:
        return
    # Prepare filename prefix.
    prefix = remove_extension(basename(path))
    # Create spot dir.
    image_dir = join(image_dir, prefix)
    os.makedirs(image_dir, exist_ok=True)
    # Load and save each tile.
    spot_metadata = []
    for (x, y), background in coords:
        # Init empty image.
        tile = np.ones((width,width,3)) * 255
        # Add data (this way we have padding!).
        tmp = image[y:y+width, x:x+width, :]
        tile[:tmp.shape[0], :tmp.shape[1]] = tmp
        # Turn to PIL image.
        tile = Image.fromarray(tile.astype(np.uint8))
        # Define path.
        tile_path = join(image_dir, f'x-{x}_y-{y}.{image_format}')
        # Collect basic metadata.
        metadata = {
            'path': tile_path,
            'spot_path': path,
            'x': x,
            'y': y,
            'width': width,
            'background': background,
        }
        # Update metadata with preprocessing metrics.
        metadata.update(preprocess(image=tile, threshold=threshold))
        # Add custom metrics.
        if custom_preprocess is not None:
            metadata.update(custom_preprocess(image))
        # Add to spot metadata.
        spot_metadata.append(metadata)
        # Save tile.
        tile.save(tile_path, quality=quality)
    return spot_metadata


def _get_all_coordidates(dimensions, width, overlap):
    """Return all coordinates."""
    x = [0]
    y = [0]
    overlap_px = int(width*overlap)
    while x[-1] < dimensions[0]:
        x.append(x[-1] + width - overlap_px)
    x = x[:-1]
    while y[-1] < dimensions[1]:
        y.append(y[-1] + width - overlap_px)
    y = y[:-1]
    coordinates = list(itertools.product(x, y))
    return coordinates


def _filter_coordinates(coords, mask, width, max_background):
    """Discard coordinates with too much background."""
    filtered = []
    for x, y in coords:
        tile_mask = mask[y:y+width, x:x+width]
        if tile_mask.size == 0:
            continue
        bg_perc = 1 - tile_mask.sum()/tile_mask.size
        if bg_perc < max_background:
            filtered.append(((x, y), bg_perc))
    return filtered
