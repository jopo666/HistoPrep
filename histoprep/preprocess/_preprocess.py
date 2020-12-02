from typing import Union, List, Dict, Any
import warnings

import numpy as np
import cv2
from PIL import Image

__all__ = [
    'PIL_to_array',
    'array_to_PIL',
    'mask_to_PIL',
    'tissue_mask',
    'artifact',
    'data_loss',
    'sharpness',
    'preprocess'
]


def PIL_to_array(image: Image.Image) -> np.ndarray:
    """Convert Pillow image to numpy array."""
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    else:
        raise TypeError('Excpected {} not {}.'.format(Image.Image, type(image)))


def array_to_PIL(image: np.ndarray) -> Image.Image:
    """Convert numpy array to Pillow Image."""
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8))
    else:
        raise TypeError('Excpected {} not {}.'.format(np.ndarray, type(image)))


def mask_to_PIL(mask: np.ndarray) -> Image.Image:
    """Normalize a numpy mask between 0 and 255 and convert to PIL image."""
    if isinstance(mask, np.ndarray):
        # Normalize between 0-255.
        mask = (mask/mask.max()) * 255
        return Image.fromarray(mask.astype(np.uint8))
    else:
        raise TypeError('Excpected {} not {}.'.format(np.ndarray, type(mask)))


def tissue_mask(
        image: Union[np.ndarray,Image.Image], 
        threshold: int = None, 
        blur: bool = True,
        mean_saturation_threshold: int = None,
        return_threshold: bool = False
        ) -> np.ndarray:
    """ Generate a tissue mask for image.

    Two methods are implemented. 
    
    Otsu's binarization:
        Otsu's binarization finds an optimal threshold by minimizing the 
        weighted within-class variance. Due to this, a relatively high threshold
        for tissue detection is often found and binarization is forced even for
        images full of background. This means that tissue is found on background
        only images and actual tissue is misclassified as background.

    Adaptive gaussian thresholding:
        Requires a threshold to be given but performs better than otsu. This
        is automatically implemented if threshold is given.
    """
    if isinstance(image, Image.Image):
        image = PIL_to_array(image)
    elif not isinstance(image, np.ndarray):
        raise TypeError('Excpected {} or {} not {}.'.format(
            np.ndarray, Image.Image, type(image)
            ))
    # HSV stands for hue, SATURATION, value.
    saturation = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)[:,:,1]
    # Blur if asked.
    if blur:
        saturation = cv2.GaussianBlur(saturation,(5,5),1)
    # Then do thresholding.
    if threshold is None:
        thresh, mask = cv2.threshold(
            src=saturation,
            thresh=None,
            maxval=1,
            type=cv2.THRESH_BINARY+cv2.THRESH_OTSU
            )
    elif isinstance(threshold,int):
        thresh, mask = cv2.threshold(
            src=saturation,
            thresh=threshold,
            maxval=1,
            type=cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            )
        mask = 1-mask
    else:
        raise TypeError(f'Excpected {int} not {type(threshold)}.')
    if return_threshold:
        return int(thresh), mask
    else:
        return mask


def artifact(
        image: Union[np.ndarray,Image.Image],
        mask: np.ndarray = None,
        quantiles: List[float] = [.01,.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]
        ) -> Dict[int,int]:
    """Detect artifacts with HSV color transformation.

    Returns selected quantile values for HUE and VALUE for tissue, which are 
    useful in the detection of artifacts.
    """
    if isinstance(image, Image.Image):
        image = PIL_to_array(image)
    elif not isinstance(image, np.ndarray):
        raise TypeError('Excpected {} or {} not {}.'.format(
            np.ndarray, Image.Image, type(image)
            ))
    if mask is None:
        mask = tissue_mask(image)
    HSV = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    hue = [np.quantile(HSV[mask==1,0],q) for q in quantiles]
    val = [np.quantile(HSV[mask==1,2],q) for q in quantiles]
    results = dict(zip(
        [f'hue_{x}' for x in quantiles] + [f'val_{x}' for x in quantiles],
        hue + val
    ))
    return results


def data_loss(image: Union[np.ndarray,Image.Image]) -> Dict[float,float]:
    """Detect data_loss.
    
    Returns the percentages of completely black and white pixels.
    """
    if isinstance(image, Image.Image):
        image = PIL_to_array(image)
    elif not isinstance(image, np.ndarray):
        raise TypeError('Excpected {} or {} not {}.'.format(
            np.ndarray, Image.Image, type(image)
            ))
    if len(image.shape) > 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return {
        'black_pixels': gray[gray==0].size/gray.size,
        'white_pixels': gray[gray==255].size/gray.size
    }


def sharpness(
        image: Union[np.ndarray,Image.Image],
        downsample: int = 2,
        reduce: str = 'max'
        ) -> float:
    """
    Sharpness detection with Laplacian variance.

    Applies sliding window to the image and calculates the laplacian variance of
    each window. The returned value is reduced with the reduce method given.
    Sliding window can be disabled wtih downsample 0.
    Reduce methods:
        max: Unlikely to be affected by
    """
    if isinstance(image, Image.Image):
        image = PIL_to_array(image)
    elif not isinstance(image, np.ndarray):
        raise TypeError('Excpected {} or {} not {}.'.format(
            np.ndarray, Image.Image, type(image)
            ))
    # Laplacian variance is defined for greyscale images.
    if len(image.shape) > 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    values = [] 
    for window in sliding_window(image):
        values.append(cv2.Laplacian(window, cv2.CV_32F).var())
    # Then reduce
    if reduce == 'max':
        return np.max(values)
    elif reduce == 'median':
        return np.median(values)
    elif reduce == 'mean':
        return np.mean(values)
    elif reduce == 'min':
        return np.min(values)
    else:
        raise ValueError('Reduce {} not recognised. Select from {}.'.format(
            reduce,['max','median','mean','min']
            ))


def sliding_window(
        image: Union[np.ndarray,Image.Image], 
        downsample: int = 2
        ) -> List[np.ndarray]:
    """Sliding window with 0.5 overlap."""
    if isinstance(image, Image.Image):
        image = PIL_to_array(image)
    elif not isinstance(image, np.ndarray):
        raise TypeError('Excpected {} or {} not {}.'.format(
            np.ndarray, Image.Image, type(image)
            ))
    w = int(min(x/downsample for x in image.shape))
    rows,cols = [int(x/(w/2)-1) for x in image.shape]
    windows = []
    for row in range(rows):
        for col in range(cols):
            r = int(row*w/2)
            c = int(col*w/2)
            windows.append(image[r:r+w,c:c+w])
    return windows


def preprocess(
        image: Union[np.ndarray,Image.Image],
        sat_thresh: int = None,
        quantiles: List[float] = [.01,.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99],
        reduce: str = 'max'
        ) -> Dict[float,float]:
    """Preprocessing metrics for a histological image.

    Arguments:
        image: Image to be preprocessed
        sat_thresh: If not defined Otsu's binarization will be used (which) may
            fail for images with data loss or only background.
        quantiles: For artifact() function.
        reduce: For sharpness() function.
    """
    if isinstance(image, Image.Image):
        image = PIL_to_array(image)
    elif not isinstance(image, np.ndarray):
        raise TypeError('Excpected {} or {} not {}.'.format(
            np.ndarray, Image.Image, type(image)
            ))
    # Initialize results.
    results = {}
    # Background percentage.
    mask = tissue_mask(image, threshold=sat_thresh)
    results['background'] = (mask==0).sum()/mask.size
    # Sharpness.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    results['sharpness'] = sharpness(gray,reduce=reduce)
    # Data loss.
    results.update(data_loss(gray))
    # Artifacts.
    results.update(artifact(image, mask=mask, quantiles=quantiles))
    return results
