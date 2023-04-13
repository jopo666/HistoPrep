"""Helper class for preprocessing tile images."""

__all__ = ["OutlierDetector"]

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from histoprep.functional._images import create_image_collage, read_images_from_paths

ERROR_NO_METRICS = (
    "Metadata does not contain any metrics, make sure tiles are saved "
    "with `save_metrics=True`."
)
ERROR_NO_DIFFERENCE = (
    "Difference between min={} and max={} values for column '{}' is zero."
)
ERROR_AX_WITH_IMAGES = "Passing `ax` when `num_images > 0` is not supported."
FIGURE_WIDTH, FIGURE_HEIGHT = plt.rcParams["figure.figsize"]
RGB_MEAN_COLUMNS = ["red_mean", "green_mean", "blue_mean"]
RGB_STD_COLUMNS = ["red_std", "green_std", "blue_std"]
XYWH_COLUMNS = ["x", "y", "w", "h"]


class OutlierDetector:
    """Class for exploring tile metadata and detecting outliers."""

    def __init__(self, dataframe: pl.DataFrame) -> None:
        self.__dataframe = dataframe
        self.__outliers = np.repeat([False], repeats=len(dataframe))
        self.__outlier_selections = []
        self.__metric_columns = [
            x for x in dataframe.columns if "_q" in x or (x.endswith(("_mean", "_std")))
        ]
        if len(self.__metric_columns) == 0:
            raise ValueError(ERROR_NO_METRICS)

    @classmethod
    def from_parquet(cls, *args, **kwargs) -> "TileMetadata":
        """Wrapper around `polars.read_parquet` function."""
        return cls(pl.read_parquet(*args, **kwargs))

    @classmethod
    def from_csv(cls, *args, **kwargs) -> "TileMetadata":
        """Wrapper around `polars.read_csv` function."""
        return cls(pl.read_csv(*args, **kwargs))

    @property
    def dataframe(self) -> pl.DataFrame:
        """Polars dataframe with metadata."""
        return self.__dataframe

    @property
    def dataframe_without_metrics(self) -> pl.DataFrame:
        """Polars dataframe without metadata."""
        forbidden = ["background", "black_pixels", "white_pixels", *self.metric_columns]
        return self.dataframe[[x for x in self.dataframe.columns if x not in forbidden]]

    @property
    def coordinates(self) -> np.ndarray:
        """Array of tile coordinates."""
        return self.dataframe[XYWH_COLUMNS].to_numpy()

    @property
    def outliers(self) -> np.ndarray:
        """Array of outlier indices."""
        return self.__outliers

    @property
    def outlier_selections(self) -> list[dict[str, Union[np.ndarray, str]]]:
        """List of dicts with outlier selections and descriptions."""
        return self.__outlier_selections

    @property
    def metric_columns(self) -> np.ndarray:
        """Image metric columns."""
        return self.__metric_columns

    @property
    def metrics(self) -> np.ndarray:
        """Array of normalized image metrics (divided by 255)."""
        return self.dataframe[self.metric_columns].to_numpy() / 255

    @property
    def mean_and_std(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Means and standard deviations for RGB channels."""
        mean = tuple((self.dataframe[RGB_MEAN_COLUMNS].mean() / 255).to_numpy()[0])
        std = tuple((self.dataframe[RGB_STD_COLUMNS].mean() / 255).to_numpy()[0])
        return mean, std

    def add_outliers(self, selection: np.ndarray, *, desc: str) -> None:
        """Set outliers to `True` with selection, and append the selection to
        `outlier_selections` property.

        Args:
            selection: Selection for indexing `outliers` property.
            desc: Description for selection.
        """
        self.__outliers[selection] = True
        self.__outlier_selections.append({"selection": selection, "desc": desc})

    def random_image_collage(
        self,
        selection: np.ndarray,
        *,
        num_rows: int = 4,
        num_cols: int = 16,
        shape: tuple[int, int] = (64, 64),
        num_workers: int = 1,
    ) -> Image.Image:
        """Generate a random collage from `paths[selection]`.

        Args:
            selection: Selection for paths.
            num_rows: Number of rows in the collage image. Defaults to 4.
            num_cols: Number of columns in the collage image. Defaults to 16.
            shape: Size of each image in the collage. Defaults to (64, 64).
            num_workers: Number of image loading workers. Defaults to 1.

        Returns:
            Collage image of randomly samples images based on selection.
        """
        if selection.sum() == 0:
            raise ValueError("Empty selection.")
        rng = np.random.default_rng()
        sampled_paths = rng.choice(
            self["path"][selection],
            size=min(selection.sum(), num_cols * num_rows),
            replace=False,
        )
        return create_image_collage(
            images=read_images_from_paths(sampled_paths, num_workers),
            num_cols=num_cols,
            shape=shape,
        )

    def cluster_kmeans(self, num_clusters: int, **kwargs) -> np.ndarray:
        """Perform kmeans clustering on the metrics and order the clusters based on the
        distance from the mean cluster center.

        Args:
            num_clusters: Number of clusters.
            **kwargs: Passed on to `sklearn.cluster.MiniBatchKMeans`.

        Returns:
            Cluster assignments.
        """
        if "n_init" not in kwargs:
            kwargs["n_init"] = "auto"
        clust = MiniBatchKMeans(n_clusters=num_clusters, **kwargs)
        clusters = clust.fit_predict(self.metrics / 255)
        # Reorder based on distances from the mean cluster center.
        mean_center = clust.cluster_centers_.mean(0)
        distances = np.array(
            [np.linalg.norm(x - mean_center) for x in clust.cluster_centers_]
        )
        ordered_clusters = np.zeros_like(clusters)
        for new_idx, old_idx in enumerate(distances.argsort()[::-1]):
            ordered_clusters[clusters == old_idx] = new_idx
        return ordered_clusters

    def plot_histogram(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        *,
        num_bins: int = 20,
        num_images: int = 12,
        num_workers: int = 1,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> Union[plt.Axes, np.ndarray[plt.Axes]]:
        """Plot column values in a histogram with example images.

        Args:
            column: Column name.
            min_value: Minimum value for filtering. Defaults to None.
            max_value: Maximum value for filtering. Defaults to None.
            num_bins: Number of bins. Defaults to 20.
            num_images: Number of images per bin. Defaults to 12.
            num_workers: Number of image loading workers. Defaults to 1.
            ax: Axis for histogram. Cannot be passed when `num_images>0`. Defaults to
                None.
            **kwargs: Passed to `plt.hist`.

        Raises:
            ValueError: No difference between min and max values.
            ValueError: Passing an axis when `num_images>0`.

        Returns:
            Matplotlib axis or axes when `num_images>0`.
        """
        if ax is not None and num_images > 0:
            raise ValueError(ERROR_AX_WITH_IMAGES)
        values = self[column]
        if min_value is None:
            min_value = values.min()
        if max_value is None:
            max_value = values.max()
        if min_value == max_value:
            raise ValueError(ERROR_NO_DIFFERENCE.format(min_value, max_value, column))
        selection = (values >= min_value) & (values <= max_value)
        if "ec" not in kwargs:
            kwargs["ec"] = "black"
        if num_images == 0:
            # Plot only histogram.
            return _plot_histogram(values[selection], num_bins, ax=ax, **kwargs)
        plt.gca().remove()  # Auto removal depracated since 3.6
        # Initialize figure.
        ax_hist = plt.subplot2grid((4, num_bins), (0, 0), colspan=num_bins)
        ax_images = []
        for i in range(num_bins):
            ax_images.append(
                plt.subplot2grid((4, num_bins), (1, i), colspan=1, rowspan=3)
            )
        # Plot histogram.
        _plot_histogram(values[selection], num_bins, ax=ax_hist, **kwargs)
        # Plot images.
        bin_images = _get_bin_collages(
            paths=self["path"][selection],
            values=values[selection],
            num_bins=num_bins,
            num_images=num_images,
            rng=np.random.default_rng(),
            num_workers=num_workers,
        )
        for idx, bin_image in enumerate(bin_images):
            ax_images[idx].imshow(bin_image)
            ax_images[idx].axis("off")
        return np.array([ax_hist, *ax_images])

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.dataframe.get_column(key).to_numpy()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_images={len(self.dataframe)}, "
            f"num_outliers={self.outliers.sum()})"
        )


def _plot_histogram(
    values: np.ndarray, n_bins: int, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    """Plot histogram."""
    if ax is None:
        plt.hist(values, bins=n_bins, **kwargs)
        ax = plt.gca()
    else:
        ax.hist(values, bins=n_bins, **kwargs)
    ax.set_xlim(values.min(), values.max())
    return ax


def _get_bin_collages(
    *,
    paths: np.ndarray,
    values: np.ndarray,
    num_bins: int,
    num_images: int,
    num_workers: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Collect bin image collages."""
    __, cutoffs = np.histogram(values, bins=num_bins)
    bin_indices = np.digitize(values, cutoffs[:-1]) - 1
    bin_paths = _collect_bin_paths(
        paths, bin_indices=bin_indices, n_images_per_bin=num_images, rng=rng
    )
    return _read_bin_paths(
        bin_paths,
        n_bins=num_bins,
        n_images_per_bin=num_images,
        num_workers=num_workers,
    )


def _collect_bin_paths(
    paths: np.ndarray,
    bin_indices: np.ndarray,
    n_images_per_bin: int,
    rng: np.random.Generator,
) -> list[str]:
    """Collect image paths based on binned values."""
    output = []
    for bin_idx in range(bin_indices.max() + 1):
        bin_paths = paths[bin_indices == bin_idx]
        if len(bin_paths) == 0:
            output += [None] * n_images_per_bin
        elif len(bin_paths) < n_images_per_bin:
            output += bin_paths.tolist()
            output += [None] * (n_images_per_bin - len(bin_paths))
        else:
            output += rng.choice(bin_paths, n_images_per_bin).tolist()
    return output


def _read_bin_paths(
    bin_paths: list[Optional[str]], n_bins: int, n_images_per_bin: int, num_workers: int
) -> list[np.ndarray]:
    """Read paths for each bin and generate vertically stacked array."""
    all_images = []
    resize_shape = None
    for image in read_images_from_paths(bin_paths, num_workers=num_workers):
        if resize_shape is None and image is not None:
            resize_shape = image.shape
        all_images.append(image)
    if resize_shape is None:
        # No images.
        empty = np.zeros((2 * n_images_per_bin, 2, 3), dtype=np.uint8) + 255
        return [empty for __ in range(n_bins)]
    # Generate collages.
    output, column = [], []
    for img in all_images:
        if img is None:
            img = np.zeros(resize_shape, dtype=np.uint8) + 255  # noqa
        column.append(img)
        if len(column) == n_images_per_bin:
            output.append(np.vstack(column))
            column = []
    return output
