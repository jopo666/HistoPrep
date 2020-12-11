from typing import Union, List, Dict, Any, Tuple

import numpy as np
import cv2
from PIL import Image

__all__ = [
    'tissue_mask',
    'HSV_quantiles',
    'RGB_quantiles',
    'data_loss',
    'sharpness',
    'preprocess',
    'sliding_window',
    'PIL_to_array',
    'array_to_PIL',
    'mask_to_PIL',
]


def PIL_to_array(image: Image.Image) -> np.ndarray:
    """Convert Pillow image to numpy array."""
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    else:
        raise TypeError('Excpected {} not {}.'.format(
            Image.Image, type(image)))


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
        image: Union[np.ndarray, Image.Image],
        sat_thresh: int = None,
        blur_kernel: Tuple[int, int] = (5, 5),
        blur_iterations: int = 1,
        return_threshold: bool = False
) -> np.ndarray:
    """Generate a tissue mask for image.

    Arguments:
        image: Input image.
        sat_thresh: Saturation threshold for tissue detection (see the method
            explanation below).
        blur_kernel: Kernel to be used in Gaussian Blur. Set to None to disable.
        blur_iterations: How many iterations to blur with kernel.
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
    saturation = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]
    # Blur if asked.
    if blur_kernel is not None:
        saturation = cv2.GaussianBlur(saturation, blur_kernel, blur_iterations)
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


def RGB_quantiles(
        image: Union[np.ndarray, Image.Image],
        gray: np.ndarray = None,
        quantiles: List[float] = [.01, .05, 0.1,
                                  0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        mask: np.ndarray = None,
        resize: int = None,
        sat_thresh: int = None
) -> Dict[int, int]:
    """Color channel quantiles including grayscale.

    Arguments:
        image: Input image.
        gray: Grayscale of the input image. Will be created if None.
        mask: Tissue mask for the input image. Will be generated if not defined.
        quantiles: The quantiles  of color channel values for tissue areas.
        resize: Resize the image to resize x resize. The function can become 
            really slow with large images, in these situations just use this
            option.
        sat_thresh: For tissue_mask() function. Ignored if mask is defined.

    Return:
        dict: A dictionary of the quantiles of color channel values for tissue 
            areas.

    Useful in the detection of misclassified tissue and artifacts.
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
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif not isinstance(gray, np.ndarray):
        raise TypeError('Excpected grayscale to be {} not {}.'.format(
            np.ndarray, type(gray)
        ))
    if resize is not None:
        image = cv2.resize(image, (resize, resize), cv2.INTER_LANCZOS4)
        gray = cv2.resize(gray, (resize, resize), cv2.INTER_LANCZOS4)
    if mask is None:
        mask = tissue_mask(image, sat_thresh=sat_thresh)
    elif mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (resize, resize), cv2.INTER_LANCZOS4)
    # Collect quantiles.
    red = [np.quantile(image[mask == 1, 0], q) for q in quantiles]
    green = [np.quantile(image[mask == 1, 1], q) for q in quantiles]
    blue = [np.quantile(image[mask == 1, 2], q) for q in quantiles]
    gray = [np.quantile(gray[mask == 1], q) for q in quantiles]
    keys = (
        [f'red_{x}' for x in quantiles] +
        [f'green_{x}' for x in quantiles] +
        [f'blue_{x}' for x in quantiles] +
        [f'gray_{x}' for x in quantiles]
    )
    results = dict(zip(keys, red + green + blue + gray))
    return results


def HSV_quantiles(
        image: Union[np.ndarray, Image.Image],
        quantiles: List[float] = [.01, .05, 0.1,
                                  0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        mask: np.ndarray = None,
        resize: int = None,
        sat_thresh: int = None,
) -> Dict[int, int]:
    """Detect artifacts with HSV color transformation.

    Arguments:
        image: Input image.
        mask: Tissue mask for the input image. Will be generated if not defined.
        quantiles: The quantiles of hue, sat and value values for tissue
            areas.
        resize: Resize the image to resize x resize. The function can become 
            really slow with large images, in these situations just use this
            option.
        sat_thresh: For tissue_mask() function. Ignored if mask is defined.

    Return:
        dict: A dictionary of the quantiles of hue, sat and value values for 
            tissue areas.

    Useful in the detection of artifacts.
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
    if resize is not None:
        image = cv2.resize(image, (resize, resize), cv2.INTER_LANCZOS4)
    if mask is None:
        mask = tissue_mask(image, sat_thresh=sat_thresh)
    elif mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (resize, resize), cv2.INTER_LANCZOS4)
    HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = [np.quantile(HSV[mask == 1, 0], q) for q in quantiles]
    sat = [np.quantile(HSV[mask == 1, 1], q) for q in quantiles]
    val = [np.quantile(HSV[mask == 1, 2], q) for q in quantiles]
    keys = (
        [f'hue_{x}' for x in quantiles] +
        [f'sat_{x}' for x in quantiles] +
        [f'val_{x}' for x in quantiles]
    )
    results = dict(zip(keys, hue + sat + val))
    return results


def data_loss(image: Union[np.ndarray, Image.Image]) -> Dict[float, float]:
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
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return {
        'black_pixels': gray[gray == 0].size/gray.size,
        'white_pixels': gray[gray == 255].size/gray.size
    }


def sharpness(
        image: Union[np.ndarray, Image.Image],
        divider: int = 2,
        reduction: Union[str, List[str]] = 'max'
) -> dict:
    """Sharpness detection with Laplacian variance.

    Arguments:
        image: Input image.
        divider: Divider argument for the sliding_window() function. Window size
            is defined as min(heigh,width)/divider
        reduction: Reduction method(s) for the Laplacian variance values for 
            each window.

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
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    values = []
    for window in sliding_window(image):
        values.append(cv2.Laplacian(window, cv2.CV_32F).var())
    # Then reduction(s).
    if isinstance(reduction, str):
        reduction = [reduction]
    results = dict(zip(
        ['sharpness_'+method for method in reduction],
        [reduce_values(values, method) for method in reduction]
    ))
    return results


def reduce_values(values: list, method: str):
    """Reduce values of values with given method."""
    allowed_methods = ['max', 'median', 'mean', 'min']
    if method not in allowed_methods:
        raise ValueError('Reduction {} not recognised. Select from {}.'.format(
            reduction, allowed_methods
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
        image: Union[np.ndarray, Image.Image],
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
    rows, cols = [int(x/(w/2)-1) for x in image.shape[:2]]
    windows = []
    for row in range(rows):
        for col in range(cols):
            r = int(row*w/divider)
            c = int(col*w/divider)
            windows.append(image[r:r+w, c:c+w])
    return windows


def preprocess(
        image: Union[np.ndarray, Image.Image],
        sat_thresh: int = None,
        resize: int = 64,
        quantiles: List[float] = [.01, .05, 0.1,
                                  0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        reduction: Union[str, List[str]] = ['max', 'median', 'mean', 'min']
) -> dict:
    """Preprocessing metrics for a histological image.

    Arguments:
        image: Image to be preprocessed
        sat_thresh: If not defined Otsu's binarization will be used (which) may
            fail for images with data loss or only background.
        resize: For artifact() function (it's slow as fuck if tiles are not
            resized to 64x64).
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
    # Initialize results and other shit.
    results = {}
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = tissue_mask(image, sat_thresh=sat_thresh)
    # Background percentage.
    results['background'] = (mask == 0).sum()/mask.size
    # Sharpness.
    results.update(sharpness(gray, reduction=reduction))
    # Data loss.
    results.update(data_loss(gray))
    # Artifacts.
    # Resize
    small_img = cv2.resize(image, (resize, resize), cv2.INTER_LANCZOS4) 
    small_gray = cv2.resize(image, (resize, resize), cv2.INTER_LANCZOS4)
    small_mask = cv2.resize(mask, (resize, resize), cv2.INTER_LANCZOS4)
    results.update(HSV_quantiles(
        small_img, mask=small_mask, quantiles=quantiles))
    results.update(RGB_quantiles(
        small_img, mask=small_mask, quantiles=quantiles, gray=small_gray))
    
    return results
