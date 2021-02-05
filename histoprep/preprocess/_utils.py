import os
from typing import Union, List, Tuple
import warnings

from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


__all__ = [
    'combine_metadata',
    'plot_on_thumbnail',
    'plot_tiles',
    'plot_histograms',
    'plot_ranges'
]


def combine_metadata(
        parent_dir: str,
        csv_path: str = None,
        overwrite: bool = False,
) -> pd.DataFrame:
    """Combine all metadata into a single csv-file.

    Arguments:
        parent_dir: Directory with all the slides.
        csv_path: Path for the combined metadata.csv. Doesn't have to be defined
            if you just want to return the pandas dataframe and for example 
            save it in another format.
        overwrite: Whether to overwrite if csv_path exists.

    Return:
        pandas.DataFrame: The combined metadata.
    """
    if not os.path.exists(parent_dir):
        raise IOError(f'{parent_dir} does not exist.')
    if csv_path is not None and os.path.exists(csv_path) and not overwrite:
        raise IOError(f'{csv_path} exists and overwrite=False.')
    dataframes = []
    directories = [x.path for x in os.scandir(parent_dir)]
    for directory in tqdm(
            directories,
            total=len(directories),
            desc='Combining metadata',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    ):
        metadata_path = os.path.join(directory, 'metadata.csv')
        # There might be slides that haven't been finished.
        if not os.path.exists(metadata_path):
            warnings.warn(
                f'{metadata_path} path not found! This warning might arise '
                'if you are still cutting slides, some of the slides were '
                'broken or no tissue was found on the slide.'
            )
        # There might be empty files.
        elif os.path.getsize(metadata_path) > 5:
            dataframes.append(pd.read_csv(metadata_path))
    if len(dataframes) == 0:
        print('No metadata.csv files found!')
        return
    metadata = pd.concat(dataframes)
    if csv_path is not None:
        metadata.to_csv(csv_path, index=False)
    return metadata


def plot_on_thumbnail(
        dataframe: pd.DataFrame,
        thumbnail: Image.Image,
        downsample: int
) -> Image.Image:
    """Plot all tiles in a dataframe to a thumbnail.

    Arguments:
        dataframe: Dataframe that contains the metadata of ONE SLIDE. The
            function obviously doesn't work with combined metadata.
        thumbnail: Thumbnail where the tiles should be drawn.
        downsample: Downsample of the thumbnail.

    Return:
        PIL.Image.Image: Thumbnail with the tiles of the dataframe drawn on it.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError('Expected {} not {}'.format(
            pd.DataFrame, type(dataframe)
        ))
    if not isinstance(thumbnail, Image.Image):
        raise ValueError('Expected {} not {}'.format(
            Image.Image, type(thumbnail)
        ))
    try:
        downsample = int(downsample)
    except:
        raise ValueError('Expected {} not {}'.format(
            int, type(downsample)
        ))
    # Draw tiles to the thumbnail.
    annotated_thumbnail = thumbnail.copy()
    annotated = ImageDraw.Draw(annotated_thumbnail)
    x = dataframe['x'].tolist()
    y = dataframe['y'].tolist()
    width = np.unique(dataframe['width'].tolist())
    if width.size > 1:
        raise ValueError('All widths must be same in the dataframe!')
    else:
        width = int(width)
    w = h = int(width/downsample)
    for (x, y) in list(zip(x, y)):
        x_d = round(x/downsample)
        y_d = round(y/downsample)
        annotated.rectangle([x_d, y_d, x_d+w, y_d+h], outline='blue', width=4)
    return annotated_thumbnail


def resize(image: Union[np.ndarray, Image.Image], max_pixels: int = 1_000_000):
    """Donwsaple image until it has less than max_pixels pixels."""
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
    dimensions = image.shape[:2]
    width, height = dimensions
    factor = 0
    while width*height > max_pixels:
        factor += 1
        width = int(dimensions[0]/2**factor)
        height = int(dimensions[1]/2**factor)
    image = Image.fromarray(image)
    return image.resize((height, width))


def plot_tiles(
        dataframe: pd.DataFrame,
        rows: int = 3,
        cols: int = 3,
        max_pixels: int = 1_000_000,
        title_column: str = None,
) -> Image.Image:
    """Return a random collection of tiles from given DataFrame.

    Arguments:
        dataframe: Dataframe that contains the metadata (from one or more 
            slides).
        rows: Maximum number of rows.
        cols: Maximum number of columns.
        max_pixels: Maximum number of pixels of the returned image.
        title_column: Dataframe column where to draw a values for the title.

    Return:
        PIL.Image.Image: Collection of random tiles from the dataframe.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError('Expected {} not {}'.format(
            pd.DataFrame, type(dataframe)
        ))
    if len(dataframe) == 0:
        print('No images.')
        return
    dataframe = dataframe.sample(n=min(len(dataframe),rows*cols))
    paths = list(dataframe.get('path', None))[:rows*cols]
    if title_column is not None:
        titles = list(dataframe.get(title_column, None))[:rows*cols]
    if any(x is None for x in paths):
        raise AttributeError(f"{row} doens't have attribute 'path'.")
    images = []
    for path in paths:
        try:
            images.append(Image.open(path))
        except FileNotFoundError:
            raise FileNotFoundError(
                f'For some reason the file {row.path} was not found. Try '
                'cutting the slide again.'
            )
    if len(images) == 0:
        print('No images.')
        return
    else:
        shape = np.array(images[0]).shape
    # Then combine images to grid.
    images = [images[i:i + cols] for i in range(0, len(images), cols)]
    rows = []
    for row in images:
        while len(row) != cols:
            row.append(np.ones(shape, dtype=np.uint8)*255)
        rows.append(np.hstack(row))
    summary = Image.fromarray(np.vstack(rows))
    if title_column is not None:
        print(f'{title_column.upper()}:')
        for row in [titles[i:i + cols] for i in range(0, len(titles), cols)]:
            row = [np.round(x, 3) for x in row]
            [print(str(x).center(8), end='') for x in row]
            print()
    return resize(summary, max_pixels)


def plot_histograms(
    dataframe: pd.DataFrame,
    prefix: str,
    cols: int = 3,
    bins: int = 20,
    figsize: tuple = (20, 20),
    share_x: bool = False,
    log_y: bool = False,
    fontsize=20
) -> None:
    """Plot histograms for different columns named prefix_*.

    Arguments:
        dataframe: Your metadata.
        prefix: Name of the value you want to plot (ie. hue).
        cols: Number of columns in the figure, rows are set automatically.
        bins: Bins for histogram.
        figsize: Figure size in inches.
        sharex: Whether to share x axis between subplots.
        log_y: Log scale for y-axis.
        fontsize: Font size for labels etc.

    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError('Expected {} not {}'.format(
            pd.DataFrame, type(dataframe)
        ))
    # Collect columns
    qs = [x.split('_')[-1] for x in dataframe.columns if prefix in x]
    if len(qs) <= cols and cols > 1:
        fig, axes = plt.subplots(1, cols, figsize=figsize, sharex=share_x)
        # Plot histograms.
        for i, q in enumerate(qs):
            col_name = prefix+'_'+str(q)
            ax = dataframe.plot.hist(ax=axes[i], y=col_name, bins=bins)
            apply_fontsize(ax, size=fontsize, name=prefix)
            if log_y:
                ax.set_yscale("log")
        plt.tight_layout()
    elif cols == 1:
        fig, axes = plt.subplots(len(qs), cols, figsize=figsize, sharex=share_x)
        # Plot histograms.
        for i, q in enumerate(qs):
            col_name = prefix+'_'+str(q)
            ax = dataframe.plot.hist(ax=axes[i], y=col_name, bins=bins)
            apply_fontsize(ax, size=fontsize, name=prefix)
            if log_y:
                ax.set_yscale("log")
        plt.tight_layout()
    else:
        qs = [qs[i:i + cols] for i in range(0, len(qs), cols)]
        rows = len(qs)
        # Init subplots.
        fig, axes = plt.subplots(len(qs), cols, figsize=figsize, sharex=share_x)
        # Plot histograms.
        for x, row in enumerate(qs):
            for y, q in enumerate(row):
                col_name = prefix+'_'+str(q)
                ax = dataframe.plot.hist(ax=axes[x, y], y=col_name, bins=bins)
                apply_fontsize(ax, size=fontsize, name=prefix)
                if log_y:
                    ax.set_yscale("log")
        plt.tight_layout()


def apply_fontsize(ax: plt.Axes, size: int, name: str):
    """Set font sizes for labels."""
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(size-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(size-2)
    ax.set_ylabel("Frequency", fontsize=size)
    ax.set_xlabel(name, fontsize=size)


def plot_ranges(
    dataframe: pd.DataFrame, 
    column: str, 
    ranges: List[Tuple[float]], 
    cols: int = 8, 
    rows: int = 4,
    max_pixels: int = 1_000_000 
) -> None :
    """Use tiles from different range of values using the plot_tiles()
    
    Arguments:
        dataframe: Your metadata.
        colum: Column name in dataframe.
        ranges: List of ranges to plot in format [(low,high), (low,high), ...]
        cols: For plot_tiles() function.
        rows: For plot_tiles() function.
        max_pixels: For plot_tiles() function.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError('Expected {} not {}'.format(
            pd.DataFrame, type(dataframe)
        ))
    if column not in dataframe.columns:
        raise ValueError(f'Column {column} not in given dataframe.')
    for low,high in ranges:
        if low > high:
            raise ValueError(
                f'{low} > {high}. Please give list of ranges in format '
                '[(low,high), (low,high), ...]'
            )
        print(f'{column}: {low} to {high}')
        display(
            plot_tiles(
                dataframe[
                    (dataframe[column] > low) & 
                    (dataframe[column] < high)
                ],
                cols=cols, rows=rows, max_pixels=max_pixels
            )
        )