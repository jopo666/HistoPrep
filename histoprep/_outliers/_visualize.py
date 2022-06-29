import math
import warnings
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy
import pandas
from PIL import Image

from ..helpers._io import read_and_resize

FIGURE_WIDTH, FIGURE_HEIGHT = plt.rcParams["figure.figsize"]

__all__ = ["OutlierVisualizer"]


class OutlierVisualizer:
    def __init__(self, metadata: pandas.DataFrame):
        """Visualise preprocessing metrics to identify outliers. Plotting might take a
        while if metadata contains several million samples.

        Args:
            metadata: Metadata with preprocessing metrics.

        Example:
            ```python
            import histoprep
            # Initialize class.
            visualise = histoprep.OutlierVisualizer(metadata)
            # Lets visualise tiles with data loss...
            visualise.plot_histogram_with_examples(column="black_pixels")
            # ... or with too much background ...
            visualise.plot_histogram_with_examples(column="background")
            # ... or if the colour channel variances picked up any outlies.
            visualise.plot_rgb_std(log_scale=True)
            visualise.plot_hsv_std(log_scale=True)
            ```
        """
        self.__metadata = metadata
        self.__all_paths = metadata["path"].to_numpy()

    def plot_rgb_mean(
        self,
        bins: int = 50,
        log_scale: bool = False,
        figsize: Tuple[float, float] = (FIGURE_WIDTH * 2, FIGURE_HEIGHT),
    ):
        """Plot the standard deviation of RGB (and gray) channels for each tile.

        Args:
            bins: Number of bins. Defaults to 50.
            log_scale: Set y-axis to logarithmic scale. Defaults to False.
            figsize: Figure size. Defaults to (default_w * 2, default_h).
        """
        self.__histogram(
            columns=["gray_mean", "red_mean", "green_mean", "blue_mean"],
            titles=["Gray", "Red", "Green", "Blue"],
            log_scale=log_scale,
            subplot_kwargs=dict(nrows=1, ncols=4, figsize=figsize, sharey=True),
            hist_kwargs=dict(bins=bins),
        )
        plt.suptitle("Standard deviation color channels for each tile.")
        plt.tight_layout()

    def plot_rgb_std(
        self,
        bins: int = 50,
        log_scale: bool = False,
        figsize: Tuple[float, float] = (FIGURE_WIDTH * 2, FIGURE_HEIGHT),
    ):
        """Plot the standard deviation of RGB (and gray) channels for each tile.

        Args:
            bins: Number of bins. Defaults to 50.
            log_scale: Set y-axis to logarithmic scale. Defaults to False.
            figsize: Figure size. Defaults to (default_w * 2, default_h).
        """
        self.__histogram(
            columns=["gray_std", "red_std", "green_std", "blue_std"],
            titles=["Gray", "Red", "Green", "Blue"],
            log_scale=log_scale,
            subplot_kwargs=dict(nrows=1, ncols=4, figsize=figsize, sharey=True),
            hist_kwargs=dict(bins=bins),
        )
        plt.suptitle("Standard deviation color channels for each tile.")
        plt.tight_layout()

    def plot_hsv_mean(
        self,
        bins: int = 50,
        log_scale: bool = False,
        figsize: Tuple[float, float] = (FIGURE_WIDTH * 1.5, FIGURE_HEIGHT),
    ):
        """Plot the standard deviation of HSV channels for each tile.

        Args:
            bins: Number of bins. Defaults to 50.
            log_scale: Set y-axis to logarithmic scale. Defaults to False.
            figsize: Figure size. Defaults to (default_w * 1.5, default_h).
        """
        self.__histogram(
            columns=["hue_mean", "saturation_mean", "brightness_mean"],
            titles=["Hue", "Saturation", "Brightness"],
            log_scale=log_scale,
            subplot_kwargs=dict(nrows=1, ncols=3, figsize=figsize, sharey=True),
            hist_kwargs=dict(bins=bins),
        )
        plt.suptitle("Standard deviation of HSV channels for each tiles.")
        plt.tight_layout()

    def plot_hsv_std(
        self,
        bins: int = 50,
        log_scale: bool = False,
        figsize: Tuple[float, float] = (FIGURE_WIDTH * 1.5, FIGURE_HEIGHT),
    ):
        """Plot the standard deviation of HSV channels for each tile.

        Args:
            bins: Number of bins. Defaults to 50.
            log_scale: Set y-axis to logarithmic scale. Defaults to False.
            figsize: Figure size. Defaults to (default_w * 1.5, default_h).
        """
        self.__histogram(
            columns=["hue_std", "saturation_std", "brightness_std"],
            titles=["Hue", "Saturation", "Brightness"],
            log_scale=log_scale,
            subplot_kwargs=dict(nrows=1, ncols=3, figsize=figsize, sharey=True),
            hist_kwargs=dict(bins=bins),
        )
        plt.suptitle("Standard deviation of HSV channels for each tiles.")
        plt.tight_layout()

    @property
    def __quantile_columns(self):
        columns = []
        for col in self.__metadata.columns:
            if "_q=" in col:
                columns.append(col)
        return columns

    def plot_rgb_quantiles(
        self,
        bins: int = 50,
        log_scale: bool = False,
        figsize: Tuple[float, float] = (FIGURE_WIDTH * 2, FIGURE_HEIGHT * 2),
    ):
        """Plot the quantile values of RGB channels for each tile.

        Args:
            bins: Number of bins. Defaults to 50.
            log_scale: Set y-axis to logarithmic scale. Defaults to False.
            figsize: Figure size. Defaults to (default_w * 2, default_h * 2).
        """
        channels = ["red", "green", "blue"]
        columns = []
        titles = []
        for col in self.__quantile_columns:
            if any(x in col for x in channels):
                columns.append(col)
                titles.append("Quantile={}".format(col.split("_")[-1][2:]))
        ncols = sum("red" in x for x in columns)
        nrows = math.ceil(len(columns) / ncols)
        axes = self.__histogram(
            columns=columns,
            titles=titles,
            log_scale=log_scale,
            subplot_kwargs=dict(
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                sharey=True,
                sharex=True,
            ),
            hist_kwargs=dict(bins=bins),
        )
        for idx in range(ncols * nrows):
            if idx % ncols == 0:
                axes[idx].set_ylabel(channels.pop().capitalize())
            axes[idx].set_xlim(0, 255)
        plt.suptitle("Quantile values of RGB channels for each tile.")
        plt.tight_layout()

    def plot_hsv_quantiles(
        self,
        bins: int = 50,
        log_scale: bool = False,
        figsize: Tuple[float, float] = (FIGURE_WIDTH * 2, FIGURE_HEIGHT * 2),
    ):
        """Plot the quantile values of RGB channels for each tile.

        Args:
            bins: Number of bins. Defaults to 50.
            log_scale: Set y-axis to logarithmic scale. Defaults to False.
            figsize: Figure size. Defaults to (default_w * 2, default_h * 2).
        """
        channels = ["hue", "saturation", "brightness"]
        columns = []
        titles = []
        for col in self.__quantile_columns:
            if any(x in col for x in channels):
                columns.append(col)
                titles.append("Quantile={}".format(col.split("_")[-1][2:]))
        ncols = sum("hue" in x for x in columns)
        nrows = math.ceil(len(columns) / ncols)
        axes = self.__histogram(
            columns=columns,
            titles=titles,
            log_scale=log_scale,
            subplot_kwargs=dict(
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                sharey=True,
                sharex=True,
            ),
            hist_kwargs=dict(bins=bins),
        )
        for idx in range(ncols * nrows):
            if idx % ncols == 0:
                axes[idx].set_ylabel(channels.pop().capitalize())
            axes[idx].set_xlim(0, 255)
        plt.suptitle("Quantile values of HSV channels for each tile.")
        plt.tight_layout()

    def plot_histogram_with_examples(
        self,
        column: str,
        bins: int = 30,
        num_examples: int = 14,
        log_scale: bool = True,
        figsize: Tuple[float, float] = (FIGURE_WIDTH * 2, FIGURE_HEIGHT * 2),
    ):
        """Plot a histogram of column values, with example tiles from each bin.

        Args:
            column: Name of the column to plot.
            bins: Number of bins. Defaults to 30.
            num_examples: Number of example images per bin. Defaults to 16.
            log_scale: Set y-axis to logarithmic scale. Defaults to True.
            figsize: Figure size. Defaults to (default_w * 2, default_h * 2).
        """
        if column not in self.__metadata.columns:
            raise ValueError("Column {} not in metadata columns.".format(column))
        # Define figure.
        shape = (4 if num_examples <= 8 else 5, bins)
        fig = plt.figure(figsize=figsize)
        ax_hist = plt.subplot2grid(shape, (0, 0), colspan=bins)
        axes = []
        for i in range(bins):
            axes.append(
                plt.subplot2grid(shape, (1, i), colspan=1, rowspan=shape[0] - 1)
            )
        # Plot histogram.
        self.__metadata[column].hist(bins=bins, ec="black", ax=ax_hist)
        ax_hist.set_xlim(self.__metadata[column].min(), self.__metadata[column].max())
        if log_scale:
            ax_hist.set_yscale("log")
        # Load example images from each bin.
        bin_numbers = pandas.cut(
            self.__metadata[column], bins, labels=range(bins)
        ).to_numpy()
        paths = []
        for i in range(bins):
            bin_paths = self.__all_paths[bin_numbers == i]
            if len(bin_paths) == 0:
                paths += [None] * num_examples
            elif len(bin_paths) < num_examples:
                paths += bin_paths.tolist()
                paths += [None] * (num_examples - len(bin_paths))
            else:
                paths += numpy.random.choice(bin_paths, num_examples).tolist()
        # Read and plot examples.
        idx = 0
        tmp = []
        for tile in read_and_resize(paths, px=128):
            tmp.append(tile)
            if len(tmp) == num_examples:
                collage = Image.fromarray(numpy.vstack(tmp))
                axes[idx].imshow(collage)
                axes[idx].axis("off")
                idx += 1
                tmp = []
        plt.suptitle("Histogram of values from column: {}".format(column))
        plt.tight_layout(w_pad=0, h_pad=0.2)

    def __histogram(
        self,
        columns: List[str],
        titles: List[str],
        log_scale: bool,
        subplot_kwargs: dict,
        hist_kwargs: dict,
    ):
        """Helper function to plot histograms."""
        fig, axes = plt.subplots(**subplot_kwargs)
        axes = axes.ravel()
        for i, (column, title) in enumerate(zip(columns, titles)):
            self.__metadata[column].hist(ax=axes[i], **hist_kwargs)
            axes[i].set_title(title)
            if log_scale:
                axes[i].set_yscale("log")
        return axes
