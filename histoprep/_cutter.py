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
from PIL import Image, ImageDraw
from openslide import OpenSlide

from ._functional import (
    get_thumbnail,
    get_downsamples,
    try_thresholds,
    resize
)
from .preprocess.functional import preprocess, tissue_mask
from ._czi_reader import OpenSlideCzi
from .helpers._utils import (
    remove_extension,
    remove_images,
    save_pickle
)

__all__ = [
    'Cutter',
    'TMACutter'
]


class Cutter(object):
    """
    Cut tiles from histological images.

    This class detectct tissue on the slide and cuts tiles of desired width
    from the image.

    Args:
        slide_path (str): Path to the slide image. All formats that are 
            supported by openslide can be used.
        width (int): Tile width.
        overlap (float, optional): Overlap between neighbouring tiles. Defaults 
            to 0.0.
        threshold (int, optional): Threshold value for tissue detection.
            Can be left undefined, in which case Otsu's binarization is used. 
            This is not recommended! Values can easily be searched with 
            Cutter.try_thresholds() function. Defaults to None.
        downsample (int, optional): Downsample used for the thumbnail.
            When a lower downsample is used, the thumbnail-based background 
            detection is more accurate but slower. Good results are achieved 
            with downsample=16. Defaults to 16.
        max_background (float, optional): Maximum amount of background allowed 
            for a tile. Due to the thumbnail-based background detection, tiles 
            with higher background percentage may pass through but rarely the 
            other way around. Defaults to 0.999.
        create_thumbnail (bool, optional):  Create a thumbnail if downsample is 
            not available. Defaults to False.
        thumbnail_path (str, optional): Load a created thumbnail from a file.
            Defaults to None.

    Raises:
        IOError: slide_path not found.
        ValueError: downsample is not available and create_thumbnail=False.
        IOError: thumbnail_path not found.

    Example::

        import histoprep as hp
        cutter = hp.Cutter(slide_path='path/to/slide', width=512, overlap=0.2)
        metadata = cutter.save('/path/to/output_dir')

    """

    def __init__(
            self,
            slide_path: str,
            width: int,
            overlap: float = 0.0,
            threshold: int = None,
            downsample: int = 16,
            max_background: float = 0.999,
            create_thumbnail: bool = False,
            thumbnail_path: str = None):
        super().__init__()
        # Define slide reader.
        if not exists(slide_path):
            raise IOError(f'{slide_path} not found.')
        if slide_path.endswith('czi'):
            warnings.warn(
                "Support for czi-files is in alpha phase! If "
                "you run into errors, please submit an issue to "
                "https://github.com/jopo666/HistoPrep/issues"
            )
            self.reader = OpenSlideCzi(slide_path)
            self._czi = True
        else:
            self.reader = OpenSlide(slide_path)
            self._czi = False
        # Assing basic stuff that user can see/check.
        self.slide_path = slide_path
        self.slide_name = remove_extension(basename(slide_path))
        self.dimensions = self.reader.dimensions
        self.downsample = downsample
        self.width = width
        self.overlap = overlap
        self.threshold = threshold
        # Warn about Otsu's thresholding.
        # if self.threshold is None:
        #     warnings.warn(
        #         "No threshold defined for tissue detection! Otsu's method will "
        #         "be used to select a threshold which is not always optimal. "
        #         "Different thresholds can be easily tried with the "
        #         "Cutter.try_tresholds() command."
        #     )
        self.max_background = max_background
        self.all_coordinates = self._get_all_coordinates()
        # Filter coordinates.
        if thumbnail_path is not None:
            if not exists(thumbnail_path):
                raise IOError(f'{thumbnail_path} not found.')
            self.thumbnail = Image.open(thumbnail_path).convert('RGB')
        else:
            self.thumbnail = get_thumbnail(
                slide_path=self.slide_path,
                downsample=self.downsample,
                create_thumbnail=create_thumbnail
            )
        if self.thumbnail is None:
            # Downsample not available.
            raise ValueError(
                f'Thumbnail not available for downsample {self.downsample}. '
                'Please set create_thumbnail=True or select downsample from\n\n'
                f'{self._downsamples()}'
            )
        self.threshold, self._tissue_mask = tissue_mask(
            image=self.thumbnail,
            threshold=self.threshold,
            return_threshold=True
        )
        self.filtered_coordinates = self._filter_coordinates()
        # Annotate thumbnail
        self._annotate()

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __len__(self):
        return len(self.filtered_coordinates)

    def available_downsamples(self):
        """
        Returns available downsamples for the slide.
        """
        print(self._downsamples())

    def _downsamples(self):
        string = 'Downsample  Dimensions'
        if self._czi:
            d = {1: self.dimensions}
        else:
            d = get_downsamples(self.slide_path)
        for item, val in d.items():
            string += f'\n{str(item).ljust(12)}{val}'
        return string

    def summary(self):
        """Returns a summary of the cutting process."""
        print(self._summary())

    def _summary(self):
        return (
            f"{self.slide_name}"
            f"\n  Tile width: {self.width}"
            f"\n  Tile overlap: {self.overlap}"
            f"\n  Threshold: {self.threshold}"
            f"\n  Max background: {self.max_background}"
            f"\n  Thumbnail downsample: {self.downsample}"
            f"\n  Total number of tiles: {len(self.all_coordinates)}"
            f"\n  After background filtering: {len(self.filtered_coordinates)}"
        )

    def get_annotated_thumbnail(self,
                                max_pixels: int = 1_000_000) -> Image.Image:
        """
        Returns an Pillow Image of the annotated thumbnail for inspection.

        Args:
            max_pixels (int, optional): Downsample the image until the image 
                has less than max_pixles pixels. Defaults to 1_000_000.

        Returns:
            Image.Image: Annotated thumbnail.
        """
        return resize(self._annotated_thumbnail, max_pixels)

    def get_thumbnail(self, max_pixels: int = 1_000_000) -> Image.Image:
        """
        Returns an Pillow Image of the thumbnail for inspection.

        Args:
            max_pixels (int, optional): Downsample the image until the image 
                has less than max_pixles pixels. Defaults to 1_000_000.

        Returns:
            Image.Image: Thumbnail.
        """
        return resize(self.thumbnail, max_pixels)

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
        # Flip for a nicer image
        mask = 1 - mask
        mask = mask/mask.max()*255
        mask = Image.fromarray(mask.astype(np.uint8))
        return resize(mask, max_pixels)

    def _prepare_directories(self, output_dir: str) -> None:
        out_dir = join(output_dir, self.slide_name)
        # Save paths.
        self._meta_path = join(out_dir, 'metadata.csv')
        self._thumb_path = join(out_dir, f'thumbnail_{self.downsample}.jpeg')
        self._annotated_path = join(out_dir, 'thumbnail_annotated.jpeg')
        self._param_path = join(out_dir, 'parameters.p')
        self._summary_path = join(out_dir, 'summary.txt')
        self._image_dir = join(out_dir, 'tiles')
        # Make dirs.
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(self._image_dir, exist_ok=True)

    def _annotate(self) -> None:
        # Draw tiles to the thumbnail.
        self._annotated_thumbnail = self.thumbnail.copy()
        annotated = ImageDraw.Draw(self._annotated_thumbnail)
        w = h = int(self.width/self.downsample)
        for (x, y), __ in self.filtered_coordinates:
            x_d = round(x/self.downsample)
            y_d = round(y/self.downsample)
            annotated.rectangle([x_d, y_d, x_d+w, y_d+h],
                                outline='red', width=4)

    def try_thresholds(
        self,
        thresholds: List[int] = [250, 240, 230,
                                 220, 200, 190, 180, 170, 160, 150, 140],
        max_pixels: int = 1_000_000
    ) -> Image.Image:
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
        return try_thresholds(thumbnail=self.thumbnail, thresholds=thresholds)

    def save(
        self,
        output_dir: str,
        overwrite: bool = False,
        image_format: str = 'jpeg',
        quality: int = 95,
        custom_preprocess: Callable[[Image.Image], dict] = None
    ) -> pd.DataFrame:
        """
        Save tile images and metadata.

        The function saves all the detected tiles in the desired format. When
        the acutal image is loaded into memory, basic preprocessing metrics are
        computed and added to metadata for preprocessing.

        Args:
            output_dir (str): Parent directory for all output.
            overwrite (bool, optional): This will **remove** all saved images, 
                thumbnail and metadata and save images again.. Defaults to 
                False.
            image_format (str, optional): Format can be jpeg or png. Defaults 
                to 'jpeg'.
            quality (int, optional): For jpeg compression. Defaults to 95.
            custom_preprocess (Callable[[Image.Image], dict], optional): This is
                intended for users that want to define their own preprocessing 
                function. The function must take a Pillow image as an input and 
                return a dictionary of desired metrics. Defaults to None.

        Raises:
            ValueError: Invalid image format.

        Returns:
            pd.DataFrame: Metadata.
        """

        allowed_formats = ['jpeg', 'png']

        if image_format not in allowed_formats:
            raise ValueError(
                'Image format {} not allowed. Select from {}'.format(
                    image_format, allowed_formats
                ))
        self._prepare_directories(output_dir)
        # Check if slide has been cut before.
        if exists(self._meta_path) and not overwrite:
            print(f'Slide has already been cut!')
            return pd.read_csv(self._meta_path)
        elif exists(self._meta_path) and overwrite:
            # Remove all previous files.
            os.remove(self._annotated_path)
            os.remove(self._meta_path)
            remove_images(self._image_dir)
        # Save both thumbnails.
        self.thumbnail.save(self._thumb_path, quality=95)
        self._annotated_thumbnail.save(self._annotated_path, quality=95)
        # Save used parameters. NOTE: Can't remember where I would need these...
        # self._save_parameters()
        # Save text summary.
        with open(self._summary_path, "w") as f:
            f.write(self._summary())
        # Wrap the saving function so it can be parallized.
        func = partial(save_tile, **{
            'slide_path': self.slide_path,
            'slide_name': self.slide_name,
            'image_dir': self._image_dir,
            'width': self.width,
            'threshold': self.threshold,
            'image_format': image_format,
            'quality': quality,
            'custom_preprocess': custom_preprocess,
        })
        # Multiprocessing to speed things up!
        metadata = []
        with mp.Pool(processes=os.cpu_count()-1) as p:
            for result in tqdm(
                p.imap(func, self.filtered_coordinates),
                total=len(self.filtered_coordinates),
                desc=self.slide_name,
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            ):
                metadata.append(result)
        metadata = list(filter(None, metadata))
        if len(metadata) == 0:
            print(f'No tiles saved from slide {self.slide_path}!')
            return
        # Save metadata.
        self.metadata = pd.DataFrame(metadata)
        self.metadata.to_csv(self._meta_path, index=False)
        return self.metadata

    def _get_all_coordinates(self):
        """Return tile coordinates over the whole slide."""
        x = [0]
        y = [0]
        overlap_px = int(self.width*self.overlap)
        while x[-1] < self.dimensions[0]:
            x.append(x[-1] + self.width - overlap_px)
        x = x[:-1]
        while y[-1] < self.dimensions[1]:
            y.append(y[-1] + self.width - overlap_px)
        y = y[:-1]
        coordinates = list(itertools.product(x, y))
        return coordinates

    def _filter_coordinates(self):
        """Filter out coordinates with too much background."""
        filtered = []
        width_d = np.ceil(self.width/self.downsample).astype(int)
        for x, y in self.all_coordinates:
            y_d = int(y/self.downsample)
            x_d = int(x/self.downsample)
            mask = self._tissue_mask[y_d:y_d+width_d, x_d:x_d+width_d]
            if mask.size == 0:
                continue
            bg_perc = 1 - mask.sum()/mask.size
            if bg_perc < self.max_background:
                filtered.append(((x, y), bg_perc))
        return filtered


def save_tile(
        coords: Tuple[int, int, float],
        slide_path: str,
        slide_name: str,
        image_dir: str,
        width: int,
        threshold: int,
        image_format: str,
        quality: int,
        custom_preprocess: Callable[[Image.Image], dict] = None
) -> dict:
    """Saves a tile and returns metadata (parallizable)."""
    # Load slide as it can't be pickled...
    if slide_path.endswith('czi'):
        reader = OpenSlideCzi(slide_path)
    else:
        reader = OpenSlide(slide_path)
    (x, y), bg_estimate = coords
    # Prepare filename.
    filepath = join(image_dir, f'x-{x}_y-{y}')
    if image_format == 'png':
        filepath = filepath + '.png'
    else:
        filepath = filepath + '.jpeg'
    # Collect basic metadata.
    metadata = {
        'path': filepath,
        'slide_name': slide_name,
        'x': x,
        'y': y,
        'width': width,
        'background_estimate': bg_estimate
    }
    # Load image.
    try:
        image = reader.read_region((x, y), 0, (width, width)).convert('RGB')
    except:
        # Sometimes parts of the slide are corrupt or something...
        return
    # Update metadata with preprocessing metrics.
    metadata.update(preprocess(image=image, threshold=threshold))
    # Add custom metrics.
    if custom_preprocess is not None:
        metadata.update(custom_preprocess(image))
    # Save image.
    if not exists(filepath):
        image.save(filepath, quality=quality)
    return metadata
