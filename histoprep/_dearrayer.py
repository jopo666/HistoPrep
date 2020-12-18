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
)
from .preprocess.functional import preprocess, tissue_mask
from ._helpers import remove_extension, remove_images


class Dearrayer(object):
    """Cut TMA spots from a TMA array slide.

    Parameters:
        slide_path:
            Path to the TMA slide array. All formats that are supported by 
            openslide can be used.
        sat_thresh: 
            Saturation threshold for tissue detection. Can be left 
            undefined, in which case Otsu's binarization is used. This is not
            recommended! Values can easily be searched with 
            Dearrayer.try_thresholds() function.
        downsample: 
            Downsample used for the thumbnail. When a lower downsample is used, 
            the thumbnail-based background detection is more accurate but 
            slower. Good results are achieved with downsample=16.
        min_area:
            Used to detect small shit from the image that isn't a TMA spot.
                min_area = median_spot_area * min_area
        max_area:
            Used to detect big shit from the image that isn't a TMA spot.
            max_area = median_spot_area * max_area
        kernel_size: 
            Sometimes the default doesn't work for large/small downsamples.
        fontsize: 
            For annotating the thumbnail.
        create_thumbnail:
            Create a thumbnail if downsample is not available.
    """

    def __init__(
        self,
        slide_path: str,
        sat_thresh: int = None,
        downsample: int = 64,
        min_area: float = 0.1,
        max_area: float = 3,
        kernel_size: Tuple[int] = (5, 5),
        fontsize: int = 2,
        create_thumbnail: bool = False,
    ):
        super().__init__()
        # Define openslide reader.
        if not exists(slide_path):
            raise IOError(f'{slide_path} not found.')
        self._reader = OpenSlide(slide_path)
        # Make it global so cutting is faster.
        global __READER__
        __READER__ = self._reader
        # Assing basic stuff that user can see/check.
        self.slide_path = slide_path
        self.slide_name = remove_extension(basename(slide_path))
        self.dimensions = self._reader.dimensions
        self.downsample = downsample
        self.sat_thresh = sat_thresh
        self.min_area = min_area
        self.max_area = max_area
        self.kernel_size = kernel_size
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
        self.sat_thresh, self._tissue_mask = tissue_mask(
            image=self._thumbnail, 
            sat_thresh=self.sat_thresh,
            return_threshold=True
        )
        self._spot_mask = detect_spots(
            image=self._thumbnail,
            mask=self._tissue_mask,
            min_area=self.min_area,
            max_area=self.max_area,
            kernel_size=self.kernel_size,
        )
        self._numbers, self._boxes = get_spots(
            image=self._thumbnail,
            spot_mask=self._spot_mask,
            downsample=self.downsample
        )
        self._annotate(fontsize)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __len__(self):
        return len(self._boxes)

    def summary(self):
        print(self._summary())

    def _summary(self):
        return (
            f"{self.slide_name}"
            f"\n  Saturation threshold: {self.sat_thresh}"
            f"\n  Number of TMA spots: {len(self._boxes)}"
        )

    def plot_thumbnail(self, max_pixels=1_000_000) -> Image.Image:
        return resize(self._thumbnail, max_pixels)

    def plot_tissue_mask(self, max_pixels=1_000_000) -> Image.Image:
        mask = self._tissue_mask
        # Flip for a nicer image
        mask = 1 - mask
        mask = mask/mask.max()*255
        mask = Image.fromarray(mask.astype(np.uint8))
        return resize(mask, max_pixels)

    def plot_spot_mask(self, max_pixels=1_000_000) -> Image.Image:
        mask = self._spot_mask
        # Flip for a nicer image.
        mask = 1 - mask
        mask = mask/mask.max()*255
        mask = Image.fromarray(mask.astype(np.uint8))
        return resize(mask, max_pixels)

    def plot_bounding_boxes(self, max_pixels=5_000_000) -> Image.Image:
        return resize(self._annotated_thumbnail, max_pixels)

    def _annotate(self, fontsize):
        """Draw bounding boxes and numbers to the thumbnail."""
        self._annotated_thumbnail = self._thumbnail.copy()
        annotated = ImageDraw.Draw(self._annotated_thumbnail)
        # Bounding boxes
        for i in range(len(self)):
            x,y,w,h = self._boxes[i]/self.downsample
            annotated.rectangle([x,y,x+w,y+h],outline='red',width=10)
        arr = np.array(self._annotated_thumbnail)
        # Numbers.
        for i in range(len(self)):
            x,y,w,h = self._boxes[i]/self.downsample
            arr = cv2.putText(
                arr,
                str(self._numbers[i]),
                (int(x+10),int(y-10+h)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontsize,
                (0, 0, 255),
                5,
                cv2.LINE_AA
                )
        self._annotated_thumbnail = Image.fromarray(arr)

    def try_thresholds(
        self,
        thresholds: List[int] = [5, 10, 15,
                                    20, 30, 40, 50, 60, 80, 100, 120],
        max_pixels=1_000_000
    ) -> Image.Image:
        """Returns a summary image of different thresholds."""
        return try_thresholds(thumbnail=self._thumbnail,thresholds=thresholds)

    def try_spot_mask(
        self,
        min_area: float = 0.1,
        max_area: float = 3,
        kernel_size: Tuple[int] = (5, 5),
        max_pixels: int = 1_000_000
    ) -> Image.Image:
        """Returns a spot mask with given arguments.
        
        Arguments:
            min_area: 
                Increase if some of the small shit is detected as a spot.
                Decrease if some spots are missed.
            max_area: 
                Increase if some spots are missed.
                Decrease if some large elements are detected as spots.
            kernel_size:
                Increase if using a small downsample.
                Decrease if using a large downsample.
        """
        mask = detect_spots(
                image=self._thumbnail,
                mask=self._tissue_mask,
                min_area=min_area,
                max_area=max_area,
                kernel_size=kernel_size,
            )
        # Flip for a nicer image.
        mask = 1 - mask
        mask = mask/mask.max()*255
        mask = Image.fromarray(mask.astype(np.uint8))
        return resize(mask, max_pixels)
    
    def _prepare_directories(self, parent_dir: str) -> None:
        out_dir = join(parent_dir, self.slide_name)
        # Save paths.
        #self._meta_path = join(out_dir, 'metadata.csv')
        self._thumb_path = join(out_dir, 'thumbnail.jpeg')
        self._image_dir = join(out_dir, 'images')
        # Make dirs.
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(self._image_dir, exist_ok=True)
    
    def save(
        self,
        parent_dir: str,
        overwrite: bool = False,
        image_format: str = 'jpeg',
        quality: int = 95,
    ) -> None:
        """Cut and save tile images and metadata.

        Arguments:
            parent_dir: 
                Save all information here.
            overwrite: 
                This will REMOVE all saved in parent_dir before saving 
                everything again.
            image_format: 
                jpeg or png.
            quality: 
                For jpeg saving.
        """
        allowed_formats = ['jpeg', 'png']
        if image_format not in allowed_formats:
            raise ValueError(
                'Image format {} not allowed. Select from {}'.format(
                    image_format, allowed_formats
                ))
        self._prepare_directories(parent_dir)
        # Check if slide has been cut before.
        if exists(self._thumb_path) and not overwrite:
            print(f'Slide has already been cut!')
            return None
        elif exists(self._thumb_path) and overwrite:
            # Remove all previous files.
            os.remove(self._thumb_path)
            remove_images(self._image_dir)
        # Save annotated thumbnail.
        self._annotated_thumbnail.save(self._thumb_path, quality=95)
        # Wrap the saving function so it can be parallized.
        func = partial(save_spot, **{
            'image_dir': self._image_dir,
            'image_format': image_format,
            'quality': quality,
        })
        # Multiprocessing to speed things up!
        data = list(zip(self._numbers, self._boxes))
        with mp.Pool(processes=os.cpu_count()) as p:
            for result in tqdm(
                p.imap(func, self.filtered_coordinates),
                total=len(self.filtered_coordinates),
                desc=self.slide_name,
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            ):
                continue
        metadata = list(filter(None, metadata))
        if len(metadata) == 0:
            print(f'No tiles saved from slide {self.slide_path}!')
            return
        # Save metadata
        metadata = pd.DataFrame(metadata)
        metadata.to_csv(self._meta_path, index=False)
        return metadata
     

def save_spot(
        data: Tuple[int,int,int,int],
        image_dir: str,
        image_format: str,
        quality: int,
) -> dict:
    """Saves spot as an image (parallizable)."""
    # Unpack variables
    number, (x, y, w, h) = data
    # Load slide from global.
    reader = __READER__
    # Prepare filename.
    filepath = join(image_dir, f'{slide_name}_spot-{number}')
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
    image.save(filepath, quality=quality)