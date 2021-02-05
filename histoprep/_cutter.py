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
from ._helpers import (
    remove_extension,
    remove_images,
    save_data
)


class Cutter(object):
    """Cut tiles from histological images.

    Parameters:
        slide_path:
            Path to the slide image. All formats that are supported by openslide
            can be used.
        width: 
            Width of the square tiles to be cut.
        overlap: 
            Proportion of overlap between neighboring tiles
        threshold: 
            Threshold value for tissue detection. Can be left 
            undefined, in which case Otsu's binarization is used. This is not
            recommended! Values can easily be searched with 
            Cutter.try_thresholds() function.
        downsample: 
            Downsample used for the thumbnail. When a lower downsample is used, 
            the thumbnail-based background detection is more accurate but 
            slower. Good results are achieved with downsample=16.
        max_background:
            Maximum amount of background allowed for a tile. Due to the 
            thumbnail-based background detection, tiles with higher background
            percentage may pass through but rarely the other way.
        create_thumbnail:
            Create a thumbnail if downsample is not available.
    """

    def __init__(
        self,
        slide_path: str,
        width: int,
        overlap: float = 0.0,
        threshold: int = None,
        downsample: int = 16,
        max_background: float = 0.999,
        create_thumbnail: bool = False
    ):
        super().__init__()
        # Define openslide reader.
        if not exists(slide_path):
            raise IOError(f'{slide_path} not found.')
        self.openslide_reader = OpenSlide(slide_path)
        # Make it global so cutting is faster.
        global __READER__
        __READER__ = self.openslide_reader
        # Assing basic stuff that user can see/check.
        self.slide_path = slide_path
        self.slide_name = remove_extension(basename(slide_path))
        self.dimensions = self.openslide_reader.dimensions
        self.downsample = downsample
        self.width = width
        self.overlap = overlap
        self.threshold = threshold
        if self.threshold is None:
            warnings.warn(
                "No threshold defined for tissue detection! Otsu's method will "
                "be used to select a threshold which is not always optimal. "
                "Different thresholds can be easily tried with the "
                "Cutter.try_tresholds() command."
            )
        self.max_background = max_background
        self.all_coordinates = self._get_all_coordinates()
        # Filter coordinates.
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
        self.filtered_coordinates = self._filter_coordinates()
        # Annotate thumbnail
        self._annotate()

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __len__(self):
        return len(self.filtered_coordinates)

    def available_downsamples(self):
        print(self._downsamples())

    def _downsamples(self):
        string = 'Downsample  Dimensions'
        d = get_downsamples(self.slide_path)
        for item, val in d.items():
            string += f'\n{str(item).ljust(12)}{val}'
        return string

    def summary(self):
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

    def _save_parameters(self):
        save_data(
            data={
                'slide_path': self.slide_path,
                'width': self.width,
                'overlap': self.overlap,
                'downsample': self.downsample,
                'threshold': self.threshold,
                'max_background': self.max_background
            },
            path=self._param_path
        )

    def get_annotated_thumbnail(self, max_pixels=1_000_000) -> Image.Image:
        return resize(self._annotated_thumbnail, max_pixels)

    def get_thumbnail(self, max_pixels=1_000_000) -> Image.Image:
        return resize(self._thumbnail, max_pixels)

    def plot_tissue_mask(self, max_pixels=1_000_000) -> Image.Image:
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
        self._thumb_path = join(out_dir, 'thumbnail.jpeg')
        self._annotated_path = join(out_dir, 'thumbnail_annotated.jpeg')
        self._param_path = join(out_dir, 'parameters.p')
        self._summary_path = join(out_dir, 'summary.txt')
        self._image_dir = join(out_dir, 'images')
        # Make dirs.
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(self._image_dir, exist_ok=True)

    def _annotate(self) -> None:
        # Draw tiles to the thumbnail.
        self._annotated_thumbnail = self._thumbnail.copy()
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
        max_pixels=1_000_000
    ) -> Image.Image:
        """Returns a summary image of different thresholds."""
        return try_thresholds(thumbnail=self._thumbnail, thresholds=thresholds)

    def save(
            self,
            output_dir: str,
            overwrite: bool = False,
            image_format: str = 'jpeg',
            quality: int = 95,
            custom_preprocess: Callable[[Image.Image], dict] = None
    ) -> pd.DataFrame:
        """Cut and save tile images and metadata.

        Arguments:
            output_dir: save all information here
            overwrite: This will REMOVE all saved images,thumbnail and metadata
                and save images again.
            image_format: jpeg or png.
            quality: For jpeg saving.
            custom_preprocess: This is intended for users that want to define 
                their own preprocessing function. Function must take a Pillow
                image as an input and return a dictionary of desired metrics.
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
            os.remove(self._thumb_path)
            os.remove(self._meta_path)
            remove_images(self._image_dir)
        # Warn about Otsu's thresholding.
        if self.threshold is None:
            warnings.warn(
                "Otsu's binarization will be used which might lead to errors "
                "in tissue detection (seriously it takes a few seconds to "
                "check for a good value with Cutter.try_thresholds() "
                "function...)"
            )
            # Save both thumbnails.
        self._thumbnail.save(self._thumb_path, quality=95)
        self._annotated_thumbnail.save(self._annotated_path, quality=95)
        # Save used parameters.
        self._save_parameters()
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
        with mp.Pool(processes=os.cpu_count()) as p:
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
        metadata = pd.DataFrame(metadata)
        metadata.to_csv(self._meta_path, index=False)
        return metadata

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
    """Saves tile and returns metadata (parallizable)."""
    # Load slide from global.
    reader = __READER__
    (x, y), bg_estimate = coords
    # Prepare filename.
    filepath = join(image_dir, f'{slide_name}_x-{x}_y-{y}')
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
    if custom_preprocess is None:
        metadata.update(preprocess(image=image, threshold=threshold))
    else:
        metadata.update(custom_preprocess(image))
    # Save image.
    image.save(filepath, quality=quality)
    return metadata
