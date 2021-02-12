from typing import Union, List, Dict, Any, Tuple

import numpy as np
import cv2
from PIL import Image

__all__ = [
    'tissue_mask',
    'HSV_quantiles',
    'RGB_quantiles',
    'data_loss',
    'laplacian_variance',
    'sharpness',
    'preprocess',
    'sliding_window',
    'PIL_to_array',
    'array_to_PIL',
    'mask_to_PIL',
]


def PIL_to_array(image: Image.Image) -> np.ndarray:
    """
    Convert Pillow image to numpy array.

    Args:
        image (Image.Image): Input image.

    Raises:
        TypeError: Invalid input type for ``image``.

    Returns:
        np.ndarray: ``image`` as an numpy array.
    """
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    else:
        raise TypeError('Excpected {} not {}.'.format(
            Image.Image, type(image)))


def array_to_PIL(image: np.ndarray) -> Image.Image:
    """
    Convert numpy array to Pillow image,

    [extended_summary]

    Args:
        image (np.ndarray): Input image.

    Raises:
        TypeError: Invalid input type for ``image``.

    Returns:
        Image.Image: ``image`` as a Pillow image.
    """
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8))
    else:
        raise TypeError('Excpected {} not {}.'.format(np.ndarray, type(image)))

    """"""


def mask_to_PIL(mask: np.ndarray) -> Image.Image:
    """
    Normalize a numpy mask between 0 and 255 and convert to PIL image.

    Args:
        mask (np.ndarray): Mask as an numpy array.

    Raises:
        TypeError: Invalid mask type.

    Returns:
        Image.Image: Mask image.
    """
    if isinstance(mask, np.ndarray):
        # Normalize between 0-255.
        if mask.max() != 0:
            mask = (mask/mask.max()) * 255
        return Image.fromarray(mask.astype(np.uint8))
    else:
        raise TypeError('Excpected {} not {}.'.format(np.ndarray, type(mask)))


def tissue_mask(
        image: Union[np.ndarray, Image.Image],
        threshold: int = None,
        blur_kernel: Tuple[int, int] = (5, 5),
        blur_iterations: int = 1,
        return_threshold: bool = False) -> np.ndarray:
    """
    Generate a tissue mask for image.

    Two methods are implemented.

    Otsu's binarization:
        Otsu's method is used to find an optimal threshold by minimizing the 
        weighted within-class variance. Due to this, a relatively high 
        threshold for tissue detection is often selected and actual tissue
        is misclassified as background. Binarization is also forced even for 
        tiles with only background, causing the detection of non-existent 
        tissue.

    Adaptive gaussian thresholding:
        Requires a threshold to be given but performs better than Otsu's method.
        This is automatically implemented if a threshold is given.

    Args:
        image (Union[np.ndarray, Image.Image]): Input image.
        threshold (int, optional): Threshold for tissue detection (see the 
            method explanation above). Defaults to None.
        blur_kernel (Tuple[int, int], optional): Kernel to be used in Gaussian 
            Blur. Set to None to disable. Defaults to (5, 5).
        blur_iterations (int, optional): How many iterations to blur with 
            kernel. Defaults to 1.
        return_threshold (bool, optional): Whether to return the used threshold
            in the case of Otsu's method. Defaults to False.

    Raises:
        TypeError: Invalid type for ``image`` or ``threshold``.

    Returns:
        np.ndarray: tissue mask.
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
    # Turn RGB to GRAY
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Blur if asked.
    if blur_kernel is not None:
        gray = cv2.GaussianBlur(gray, blur_kernel, blur_iterations)
    # Then do thresholding.
    if threshold is None:
        thresh, mask = cv2.threshold(
            src=gray,
            thresh=None,
            maxval=1,
            type=cv2.THRESH_BINARY+cv2.THRESH_OTSU
        )
        mask = 1 - mask
    else:
        try:
            threshold = int(threshold)
        except:
            raise TypeError(f'Excpected {int} not {type(threshold)}.')
        thresh, mask = cv2.threshold(
            src=gray,
            thresh=threshold,
            maxval=1,
            type=cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        )
    if return_threshold:
        return int(thresh), mask
    else:
        return mask


def RGB_quantiles(
        image: Union[np.ndarray, Image.Image],
        quantiles: List[float] = [.01, .05, 0.1,
                                  0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        gray: np.ndarray = None,
        mask: np.ndarray = None,
        resize: int = None,
        threshold: int = None) -> Dict[int, int]:
    """
    Measure color channel quantiles.

    Useful in the detection of misclassified tissue and artifacts.

    Args:
        image (Union[np.ndarray, Image.Image]): Input image.
        quantiles (List[float], optional): Quantiles to be collected. Defaults 
            to [.01, .05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99].
        gray (np.ndarray, optional): Grayscale image of the input image. Will be
            generated if not defined. Defaults to None.
        mask (np.ndarray, optional): Tissue mask for the input image. 
            Will be generated if not defined. Defaults to None.
        resize (int, optional): Resize the image to resize x resize. The 
            function can become really slow with large images as we have to sort
            every pixel in the image. In these situations just use this
            option. Defaults to None.
        threshold (int, optional): For tissue_mask() function. Ignored if mask 
            is defined. Defaults to None.

    Raises:
        TypeError: Invalid type for ``image``.

    Returns:
        Dict[int, int]: Dictionary of color channel quantiles.
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
        mask = tissue_mask(image, threshold=threshold)
    elif mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (resize, resize), cv2.INTER_LANCZOS4)
    if mask.sum() == 0:
        # No tissue, return empty dict
        return {}
    # Collect channels and sort.
    red = np.sort(image[mask == 1, 0])
    green = np.sort(image[mask == 1, 1])
    blue = np.sort(image[mask == 1, 2])
    # Collect quantiles.
    red = [np.quantile(red, q) for q in quantiles]
    green = [np.quantile(green, q) for q in quantiles]
    blue = [np.quantile(blue, q) for q in quantiles]
    keys = (
        [f'red_{x}' for x in quantiles] +
        [f'green_{x}' for x in quantiles] +
        [f'blue_{x}' for x in quantiles]
    )
    results = dict(zip(keys, red + green + blue))
    return results

    """HSV channel quantiles.

    Useful in the detection of misclassified tissue and artifacts.

    Arguments:
        image: 
            Input image.
        mask: 
            Tissue mask for the input image. Will be generated if not defined.
        quantiles: 
            The quantiles of hue, sat and value values for tissue areas.
        resize: 
            Resize the image to resize x resize. The function can become 
            really slow with large images, in these situations just use this
            option.
        threshold: 
            For tissue_mask() function. Ignored if mask is defined.

    Return:
        dict: A dictionary of the quantiles of hue, sat and value values for 
            tissue areas.
    """


def HSV_quantiles(
        image: Union[np.ndarray, Image.Image],
        quantiles: List[float] = [.01, .05, 0.1,
                                  0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        mask: np.ndarray = None,
        resize: int = None,
        threshold: int = None) -> Dict[int, int]:
    """
    Measure HSV channels quantiles.

    Useful in the detection of misclassified tissue and artifacts.

    Args:
        image (Union[np.ndarray, Image.Image]): Input image.
        quantiles (List[float], optional): Quantiles to be collected. Defaults 
            to [.01, .05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99].
        mask (np.ndarray, optional): Tissue mask for the input image. 
            Will be generated if not defined. Defaults to None.
        resize (int, optional): Resize the image to resize x resize. The 
            function can become really slow with large images as we have to sort
            every pixel in the image. In these situations just use this
            option. Defaults to None.
        threshold (int, optional): For tissue_mask() function. Ignored if mask 
            is defined. Defaults to None.

    Raises:
        TypeError: Invalid type for ``image``.

    Returns:
        Dict[int, int]: Dictionary of HSV channels quantiles.
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
        mask = tissue_mask(image, threshold=threshold)
    elif mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (resize, resize), cv2.INTER_LANCZOS4)
    if mask.sum() == 0:
        # No tissue, return empty dict
        return {}
    # Collect channels and sort.
    HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = np.sort(HSV[mask == 1, 0])
    sat = np.sort(HSV[mask == 1, 1])
    val = np.sort(HSV[mask == 1, 2])
    # Collect quantiles.
    hue = [np.quantile(hue, q) for q in quantiles]
    sat = [np.quantile(sat, q) for q in quantiles]
    val = [np.quantile(val, q) for q in quantiles]
    keys = (
        [f'hue_{x}' for x in quantiles] +
        [f'sat_{x}' for x in quantiles] +
        [f'val_{x}' for x in quantiles]
    )
    results = dict(zip(keys, hue + sat + val))
    return results


def data_loss(image: Union[np.ndarray, Image.Image]) -> Dict[float, float]:
    """
    Detect data loss.

    Calculates the percentage of completely white and black pixels.

    Args:
        image (Union[np.ndarray, Image.Image]): Input image.

    Raises:
        TypeError: Invalid type for ``image``.

    Returns:
        Dict[float, float]: Percentages of completely black (0) and white (255)
            pixels.
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

def laplacian_variance(image: Union[np.ndarray, Image.Image]):
    """
    Return the laplacian variance of the image.

    Args:
        image (Union[np.ndarray, Image.Image]): input image.

    Raises:
        TypeError: Invalid type for ``image``.
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
    return cv2.Laplacian(gray, cv2.CV_32F).var()

def sharpness(
        image: Union[np.ndarray, Image.Image],
        divider: int = 2,
        reduction: Union[str, List[str]] = 'max') -> dict:
    """
    Sharpness detection with Laplacian variance.

    Divides the image into 9, 25, 49, ... tiles with 50% overlap based on 
    ``divider``, calculates the Laplacian sharpness for each tile and returns
    the value based on ``reduction``.

    Args:
        image (Union[np.ndarray, Image.Image]): Input image.
        divider (int, optional): Divider argument for the ``sliding_window()`` 
            function. Defaults to 2.
        reduction (Union[str, List[str]], optional): Reduction method(s) for the
            Laplacian variance values for each window. Defaults to 'max'.

    Raises:
        TypeError: Invalid type for ``image``.

    Returns:
        dict: Laplacian variance values. 
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


    """
    Sliding window with 0.5 overlap.

    :param image: Input image.
    :type image: Union[np.ndarray, Image.Image]
    :param divider: Window size is defined as min(height,width)/divider, where
        divider defaults to 2. For square images, divider values will produce:
            1: original image
            2: 3x3=9 windows
            3: 5x5=25 windows
            4: 7x7=49 windows
    :type divider: int, optional
    :raises TypeError: If image is in a wrong format.
    :return: List of window images.
    :rtype: List[np.ndarray]
    """

def sliding_window(
        image: Union[np.ndarray, Image.Image],
        divider: int = 2) -> List[np.ndarray]:
    """
    Sliding window with 0.5 overlap.


    Args:
        image (Union[np.ndarray, Image.Image]): Input image.
        divider (int, optional): Window size is defined as 
            min(height,width)/divider, where divider defaults to 2. For square 
            images, divider values will produce:

            1: original image
            2: 3x3=9 windows
            3: 5x5=25 windows
            4: 7x7=49 windows.

    Raises:
        TypeError: Invalid type for ``image``.

    Returns:
        List[np.ndarray]: List of window images.
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
        threshold: int = None,
        resize: int = 64,
        quantiles: List[float] = [.01, .05, 0.1,
                                  0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        reduction: Union[str, List[str]] = ['max', 'median', 'mean', 'min']
) -> dict:
    """
    Basic preprocessing metrics for a histological image.

    Args:
        image (Union[np.ndarray, Image.Image]):  Input image.
        threshold (int, optional): Threshold for tissue detection. If not 
            defined Otsu's binarization will be used (which) may fail for images
            with data loss or only background. Defaults to None.
        resize (int, optional): For artifact() function. Defaults to 64.
        quantiles (List[float], optional): For HSV_quantiles() and RGB_quantiles
            functions. Defaults to 
            [.01, .05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99].
        reduction (Union[str, List[str]], optional): Reduction methods for
            sharpness() function. Defaults to ['max', 'median', 'mean', 'min'].

    Raises:
        TypeError: Invalid type for ``image``.

    Returns:
        dict: Dictionary of basic preprocessing metrics.
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
    mask = tissue_mask(image, threshold=threshold)
    # Background percentage.
    results['background'] = (mask == 0).sum()/mask.size
    # Sharpness.
    results.update(sharpness(gray, reduction=reduction))
    # Data loss.
    results.update(data_loss(gray))
    # Artifacts.
    small_img = cv2.resize(image, (resize, resize), cv2.INTER_LANCZOS4)
    small_mask = cv2.resize(mask, (resize, resize), cv2.INTER_LANCZOS4)
    results.update(HSV_quantiles(
        small_img, mask=small_mask, quantiles=quantiles))
    results.update(RGB_quantiles(
        small_img, mask=small_mask, quantiles=quantiles))

    return results
