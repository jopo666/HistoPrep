from typing import Callable, Dict, List, Tuple, Union

import numpy
from PIL import Image

from ._functional import _rgb2hsv, _to_grayscale, _to_laplacian
from ._helpers import _five_crop, _reduce_values, _to_array, resize_image
from ._tissue import detect_tissue

DEFAULT_QUANTILES = [0.05, 0.1, 0.5, 0.9, 0.95]

__all__ = [
    "PreprocessMetrics",
    "data_loss",
    "sharpness",
    "channel_quantiles",
    "channel_std",
]


class PreprocessMetrics:
    """Class to calculate basic preprocessing metrics for a histological image.

    Args:
        channel_resize: Image size for `F.channel_quantiles()`. Defaults to 128.
        quantiles: RGB and HSV quantiles. Defaults to
            [0.05, 0.1, 0.5, 0.9, 0.95].
        sharpness_reduce: Sharpness reduction method. Defaults to "max".
        custom_callback: Custom callback for to calculate new metrics. Defaults
            to None.

    Example:
        ```python
        import histoprep.functional as F
        from histoprep.helpers import read_image

        image = read_image("path/to/image.jpeg")
        threshold, tissue_mask = F.detect_tissue(image)
        metric_fn = PreprocessMetrics(threshold)
        results = metric_fn(image)
        ```
    """

    def __init__(
        self,
        channel_resize: int = 64,
        quantiles: List[int] = DEFAULT_QUANTILES,
        sharpness_reduce: Union[str, list] = "max",
        custom_callback: Callable = None,
    ):
        self.__custom_callback = custom_callback
        self.__quantiles = quantiles
        self.__channel_resize = channel_resize
        self.__sharpness_reduce = sharpness_reduce

    def __call__(
        self,
        image: Union[numpy.ndarray, Image.Image],
        tissue_threshold: int,
    ) -> Dict[str, float]:
        """Calculate preprocessing metrics for an image,

        Args:
            image: Input image.
            threshold: Tissue detection threshold for the tile images. Otsu's
                method cannot not be used here as some of the tiles may not
                contain background, and the method would still find an "optimal"
                threshold and classify tissue as background.

        Returns:
            Preprocessing metrics.
        """
        # Convert to array.
        image = _to_array(image)
        # Detect tissue.
        __, tissue_mask = detect_tissue(image, threshold=tissue_threshold)
        # Initialize metrics with background.
        preprocessing_metrics = {
            "background": (tissue_mask == 0).sum() / tissue_mask.size
        }
        # Channel mean and std.
        preprocessing_metrics.update(channel_mean(image))
        preprocessing_metrics.update(channel_std(image))
        # Data loss.
        preprocessing_metrics.update(data_loss(image))
        # Sharpness.
        preprocessing_metrics.update(sharpness(image, reduce=self.__sharpness_reduce))
        # Resize image and mask for channel quantiles.
        resized_image = resize_image(
            image, self.__channel_resize, return_arr=True, fast_resize=True
        )
        resized_mask = resize_image(
            tissue_mask,
            self.__channel_resize,
            return_arr=True,
            fast_resize=True,
        )
        # Channel quantiles.
        preprocessing_metrics.update(
            channel_quantiles(resized_image, resized_mask, self.__quantiles)
        )
        # Call custom callback.
        if self.__custom_callback is not None:
            output = self.__custom_callback(image)
            if not isinstance(output, dict):
                raise TypeError("Custom callback should return a dictionary.")
            preprocessing_metrics.update(self.__custom_callback(image))
        return preprocessing_metrics

    def __repr__(self):
        callback_name = None
        if self.__custom_callback:
            callback_name = self.__custom_callback.__name__
        return (
            "{}(channel_resize={}, quantiles={}, "
            "sharpness_reduce={}, custom_callback={})".format(
                self.__class__.__name__,
                self.__channel_resize,
                self.__quantiles,
                self.__sharpness_reduce,
                callback_name,
            )
        )


def channel_quantiles(
    image: Union[numpy.ndarray, Image.Image],
    tissue_mask: numpy.ndarray = None,
    quantiles: List[float] = DEFAULT_QUANTILES,
) -> Dict[str, List[int]]:
    """Calculate image channel quantiles which is useful in the detection of
    artifacts and shitty images. If the input image is RGB, HSV quantiles are
    also automatically calculated.

    Args:
        image: Input image.
        tissue_mask: Tissue mask for the input image. Helpful for discarding
            background in quantile evaluation. Defaults to None.
        quantiles: Quantiles to be collected. Defaults to
            [0.05, 0.1, 0.5, 0.9, 0.95].

    Returns:
        Quantiles for each image channel.

    Example:
        ```python
        import histoprep.functional as F
        from histoprep.helpers import read_image

        image = read_image("path/to/image.jpeg")
        results = F.channel_quantiles(image)
        ```
    """
    if any(x < 0 or x > 1 for x in quantiles):
        raise ValueError("Quantiles should be between 0 and 1.")
    image = _to_array(image)
    if tissue_mask is not None:
        if tissue_mask.shape != image.shape[:2]:
            raise ValueError(
                "Tissue mask shape ({}) should match image shape ({}).".format(
                    image.shape, tissue_mask.shape
                )
            )
        selection = tissue_mask == 1
        if (selection).sum() == 0:
            selection = tissue_mask >= 0
    else:
        selection = image >= 0 if image.ndim == 2 else image[..., 0] >= 0
    # Collect quantiles.
    results = []
    if image.ndim == 2:
        # Only one channel.
        bins = numpy.cumsum(numpy.bincount(image[selection].flatten(), minlength=256))
        for q in quantiles:
            results.append(int(numpy.argwhere(bins > int(q * selection.sum()))[0]))
        return {"q={}".format(q): x for q, x in zip(quantiles, results)}
    else:
        # Loop each channel.
        for img in (image, _rgb2hsv(image)):
            for c in range(3):
                bins = numpy.cumsum(
                    numpy.bincount(img[selection, c].flatten(), minlength=256)
                )
                tmp = []
                for q in quantiles:
                    tmp.append(int(numpy.argwhere(bins > int(q * selection.sum()))[0]))
                results.append(tmp)
    results_dict = {}
    for c_name, q_values in zip(
        ["red", "green", "blue", "hue", "saturation", "brightness"], results
    ):
        for q_name, q_val in zip(quantiles, q_values):
            results_dict["{}_q={}".format(c_name, q_name)] = q_val
    return results_dict


def sharpness(
    image: Union[numpy.ndarray, Image.Image],
    reduce: Union[str, List[str]] = "max",
) -> Dict[str, float]:
    """
    The method takes five crops from the image and calculates the standard
    deviation of the crops after a Laplace transformation. These values then
    reduced with the selected method.

    Args:
        image: Input image.
        reduction: Reduction method(s) for the Laplacian variance values for
            each crop. Defaults to "max".
    Returns:
        Reduced laplacian standard deviation of the crops.

    Example:
        ```python
        import histoprep.functional as F
        from histoprep.helpers import read_image

        image = read_image("path/to/image.jpeg")
        sharpness = F.sharpness(image, "median")
        ```
    """
    image = _to_array(image)
    laplacian = _to_laplacian(image)
    # Collect stds (tirsk).
    values = []
    for crop in _five_crop(laplacian):
        values.append(crop.std())
    # Then reduction(s).
    if isinstance(reduce, str):
        reduce = [reduce]
    results = dict(
        zip(
            ["sharpness_" + method for method in reduce],
            [_reduce_values(values, method) for method in reduce],
        )
    )
    return results


def data_loss(
    image: Union[numpy.ndarray, Image.Image],
) -> Dict[str, float]:
    """
    Calculates the percentage of completely white and black pixels.

    Args:
        image: Input image.

    Returns:
        Percentages of completely black (0) and white (255) pixels.

    Example:
        ```python
        import histoprep.functional as F
        from histoprep.helpers import read_image

        image = read_image("path/to/image.jpeg")
        black_pixels, white_pixels = F.data_loss(image).values()
        ```
    """
    image = _to_array(image)
    if image.ndim != 2:
        image = _to_grayscale(image)
    return {
        "black_pixels": (image == 0).sum() / image.size,
        "white_pixels": (image == 255).sum() / image.size,
    }


def channel_mean(image: Union[numpy.ndarray, Image.Image]) -> Dict[str, float]:
    """Calculate standard deviation in each image channel.

    If the image is grayscale, only the total standard deviation is returned.
    Else both RGB and HSV channel standard deviations are evaluated.

    Args:
        image: Input image.

    Returns:
        Channel standard deviations.

    Example:
        ```python
        import histoprep.functional as F
        from histoprep.helpers import read_image

        image = read_image("path/to/image.jpeg")
        channe_mean = F.channel_mean(image)
        ```
    """
    image = _to_array(image)
    if image.ndim == 2:
        return {"gray_mean": image.mean().round(3)}
    # Start with gray.
    results = {"gray_mean": _to_grayscale(image).mean().round(3)}
    # Collect both RGB and HSV stds.
    for key, val in zip(
        ["red", "green", "blue", "hue", "saturation", "brightness"],
        (*__channel_mean(image), *__channel_mean(_rgb2hsv(image))),
    ):
        results["{}_mean".format(key)] = val
    return results


def channel_std(image: Union[numpy.ndarray, Image.Image]) -> Dict[str, float]:
    """Calculate standard deviation in each image channel.

    If the image is grayscale, only the total standard deviation is returned.
    Else both RGB and HSV channel standard deviations are evaluated.

    Args:
        image: Input image.

    Returns:
        Channel standard deviations.

    Example:
        ```python
        import histoprep.functional as F
        from histoprep.helpers import read_image

        image = read_image("path/to/image.jpeg")
        channel_std = F.channel_std(image)
        ```
    """
    image = _to_array(image)
    if image.ndim == 2:
        return {"gray_std": image.std().round(3)}
    # Start with gray.
    results = {"gray_std": _to_grayscale(image).std().round(3)}
    # Collect both RGB and HSV stds.
    for key, val in zip(
        ["red", "green", "blue", "hue", "saturation", "brightness"],
        (*__channel_std(image), *__channel_std(_rgb2hsv(image))),
    ):
        results["{}_std".format(key)] = val
    return results


def __channel_mean(image: numpy.ndarray) -> Tuple[float, float, float]:
    """Calculate channel mean."""
    return [image[..., c].mean().round(3) for c in range(3)]


def __channel_std(image: numpy.ndarray) -> Tuple[float, float, float]:
    """Calculate channel std."""
    return [image[..., c].std().round(3) for c in range(3)]
