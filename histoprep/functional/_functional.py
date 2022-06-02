import cv2
import numpy


def _thresholding(
    gray: numpy.ndarray,
    threshold: int = None,
    multiplier: float = 1.0,
    remove_white: bool = False,
) -> numpy.ndarray:
    if gray.ndim != 2:
        raise ValueError("Thresholding expects grayscale images.")
    if threshold is None:
        # Use Otsu.
        if remove_white:
            # Get threshold without white pixels.
            threshold, __ = cv2.threshold(
                gray[gray != 255],
                None,
                1,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
        else:
            # Normal thresholding.
            threshold, __ = cv2.threshold(
                gray, None, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        # Multiply threshold.
        threshold = max(min(255, int(threshold * multiplier)), 0)
    # Global thresholding.
    threshold, mask = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY_INV)
    return int(threshold), mask


def _gaussian_blur(
    image: numpy.ndarray, sigma: float, truncate: float = 3.5
) -> numpy.ndarray:
    """Gaussian blur."""
    # Define kernel size based on sigma and truncate.
    if sigma > 0:
        ksize = int(truncate * sigma + 0.5)
        if ksize % 2 == 0:
            ksize += 1
        kernel_size = (ksize, ksize)
        # Gaussian blurrrr.
        return cv2.GaussianBlur(image, ksize=kernel_size, sigmaX=sigma, sigmaY=sigma)
    else:
        return image


def _to_grayscale(image: numpy.ndarray) -> numpy.ndarray:
    """Convert image to grayscale."""
    if image.ndim == 2:
        return image
    else:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _to_laplacian(gray: numpy.ndarray) -> numpy.ndarray:
    """Apply laplacian filter to the image."""
    if gray.ndim != 2:
        gray = _to_grayscale(gray)
    return cv2.Laplacian(gray, cv2.CV_32F)


def _rgb2hsv(image: numpy.ndarray) -> numpy.ndarray:
    """Convert RGB to HSV."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
