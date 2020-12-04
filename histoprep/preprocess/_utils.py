from typing import Union, List, Dict, Any
import warnings

import numpy as np
import cv2
from PIL import Image

__all__ = [
    'tissue_mask',
    'artifact',
    'data_loss',
    'sharpness',
    'preprocess',
    'sliding_window'
]

def tissue_mask(
        image: Union[np.ndarray,Image.Image], 
        sat_thresh: int = None, 
        blur: bool = True,
        return_threshold: bool = False
        ) -> np.ndarray:
    """ Generate a tissue mask for image.

    Arguments:
        image: Input image.
        sat_thresh: Saturation threshold for tissue detection (see the method
            explanation below).
        blur: Whether to blur the image before thresholding (recommended).
        return_threshold: Whether to return the used sat_thresh in the case of
            Otsu's method.
        
    Return:
        np.ndarray: A tissue mask with 1 incidating tissue.

    
    Two methods are implemented.
    
    Otsu's binarization:
        Otsu's method is used to find an optimal saturation threshold by
        minimizing the weighted within-class variance. Due to this, a relatively
        high threshold for tissue detection is often selected and actual tissue
        is misclassified as background. Binarization is also forced even for 
        tiles with only background, causing the detection of non-existent 
        tissue.

    Adaptive gaussian thresholding:
        Requires a saturation threshold to be given but performs better than 
        Otsu's method. This is automatically implemented if a saturation 
        threshold is given.
    """
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
    else:
        raise TypeError('Excpected {} or {} not {}.'.format(
            np.ndarray, Image.Image, type(image)
            ))
    # HSV stands for hue, SATURATION, value.
    saturation = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)[:,:,1]
    # Blur if asked.
    if blur:
        saturation = cv2.GaussianBlur(saturation,(5,5),1)
    # Then do thresholding.
    if sat_thresh is None:
        thresh, mask = cv2.threshold(
            src=saturation,
            thresh=None,
            maxval=1,
            type=cv2.THRESH_BINARY+cv2.THRESH_OTSU
            )
    else:
        try:
            sat_thresh = int(sat_thresh)
        except:
            raise TypeError(f'Excpected {int} not {type(sat_thresh)}.')
        thresh, mask = cv2.threshold(
            src=saturation,
            thresh=sat_thresh,
            maxval=1,
            type=cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            )
        mask = 1 - mask
    if return_threshold:
        return int(thresh), mask
    else:
        return mask


def artifact(
        image: Union[np.ndarray,Image.Image],
        quantiles: List[float] = [.01,.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99],
        mask: np.ndarray = None
        ) -> Dict[int,int]:
    """Detect artifacts with HSV color transformation.

    Arguments:
        image: Input image.
        mask: Tissue mask for the input image. Will be generated if not defined.
        quantiles: The quantiles of hue and value to be reported for tissue
            areas.
    
    Return:
        dict: A dictionary of the quantiles of hue and value for tissue areas.
    """
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
    else:
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

    Arguments:
        image: Input image.
    
    Return:
        dict: Percentage of completely black (0) and white (255) pixels.
    """
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
    else:
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
        divider: int = 2,
        reduction: Union[str,List[str]] = 'max'
        ) -> Dict[float]:
    """Sharpness detection with Laplacian variance.
    
    Arguments:
        image: Input image.
        divider: Divider argument for the sliding_window() function. Window size
            is defined as min(heigh,width)/divider
        reduction: Reduction method(s) for the Laplacian variance values for each
            window.
    
    Return:
        dict: Sharpness values for each defined reduction method.
    """
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
    else:
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
    # Then reduction(s).
    if isinstance(reduction,str):
        reduction = [reduction]
    results = dict(zip(
        ['sharpness_'+method for method in reduction],
        [reduce_values(values,method) for method in reduction]
    ))
    return results


def reduce_values(values: list, method: str):
    """Reduce values of values with given method."""
    allowed_methods = ['max','median','mean','min']
    if method not in allowed_methods:
        raise ValueError('Reduction {} not recognised. Select from {}.'.format(
            reduction,allowed_methods
            ))
    if method == 'max':
        return np.max(values)
    elif method == 'median':
        return np.median(values)
    elif method == 'mean':
        return np.mean(values)
    elif method == 'min':
        return np.min(values)


def sliding_window(
        image: Union[np.ndarray,Image.Image], 
        divider: int = 2,
        ) -> List[np.ndarray]:
    """Sliding window with 0.5 overlap.
    
    Arguments:
        image: Input image.
        divider: Window size is defined as min(height,width)/divider.
            For square images, divider values will produce:
                1: original image
                2: 3x3=9 windows
                3: 5x5=25 windows
                4: 7x7=49 windows
                ...
    Return:
        list: List of window images as numpy arrays.
    """
    if isinstance(image, Image.Image):
        image = np.array(image, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
    else:
        raise TypeError('Excpected {} or {} not {}.'.format(
            np.ndarray, Image.Image, type(image)
            ))
    if divider < 2:
        return [image]
    w = int(min(x/divider for x in image.shape[:2]))
    rows,cols = [int(x/(w/2)-1) for x in image.shape[:2]]
    windows = []
    for row in range(rows):
        for col in range(cols):
            r = int(row*w/divider)
            c = int(col*w/divider)
            windows.append(image[r:r+w,c:c+w])
    return windows


def preprocess(
        image: Union[np.ndarray,Image.Image],
        sat_thresh: int = None,
        quantiles: List[float] = [.01,.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99],
        reduction: Union[str,List[str]] = ['max','median','mean','min']
        ) -> Dict[float,float]:
    """Preprocessing metrics for a histological image.

    Arguments:
        image: Image to be preprocessed
        sat_thresh: If not defined Otsu's binarization will be used (which) may
            fail for images with data loss or only background.
        quantiles: For artifact() function.
        reduction: For sharpness() function.
    """
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
    else:
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
    results['sharpness'] = sharpness(gray,reduction=reduction)
    # Data loss.
    results.update(data_loss(gray))
    # Artifacts.
    results.update(artifact(image, mask=mask, quantiles=quantiles))
    return results
