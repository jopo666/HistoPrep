import os
from os.path import dirname, join
import multiprocessing as mp
from typing import Union, List, Dict, Tuple

import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import Layout, VBox, HBox
from IPython.display import display

__all__ = [
    'Explore',
    'plot_on_thumbnail',
    'plot_tiles',
    'plot_ranges',
]


def split_list(l: list, n: int = 1) -> list:
    """Split list into n parts"""
    length = len(l)
    split = []
    for i in range(n):
        split.append(l[i*length // n: (i+1)*length // n])
    return split


def get_load_image(nrows: int, ncols: int, px: int):
    """Generates a white image."""
    load_img = np.ones((nrows*px, ncols*px, 3))*255
    load_img = cv2.imencode('.png', load_img)[1].tobytes()
    return load_img


def get_tiles(
    df: pd.DataFrame,
    column: str = None,
    x_range: tuple = None,
    nrows: int = 2,
    ncols: int = 16,
    px: int = 64,
    to_bytes = True,
) -> bytes:
    """Return image grid of the tiles from selection."""
    # Collect random sample and tile paths.
    if column is None and x_range is None:
         paths = df.path
    else:
        paths = df.path[
            (df[column] > x_range[0]) &
            (df[column] < x_range[1])
        ]
    number_of_samples = min(ncols*nrows, len(paths.tolist()))
    paths = paths.sample(number_of_samples)
    # Load images
    tiles = []
    processes = min(ncols*nrows, os.cpu_count() - 1)
    with mp.Pool(processes) as p:
        for result in p.imap(cv2.imread, paths):
            tiles.append(result)
    # Resize.
    tiles = [cv2.resize(x, (px, px)) for x in tiles]
    # Add white images if they run out.
    while nrows*ncols > len(tiles):
        tiles.append(np.ones((px, px, 3)) * 255)
    # Combine.
    tiles = [np.hstack(r) for r in split_list(tiles, nrows)]
    tiles = np.vstack(tiles)
    if to_bytes:
        # To bytes.
        __, encoded = cv2.imencode('.png', tiles)
        return encoded.tobytes()
    else:
        tiles = cv2.cvtColor(tiles.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return Image.fromarray(tiles)


def get_widgets(
    df: pd.DataFrame,
    group: str,
    nrows: int = 2,
    ncols: int = 16,
    px: int = 64,
):
    # Collect all columns.
    columns = {
        'sharpness': [x for x in df.columns if 'sharpness' in x],
        'data_loss': [x for x in df.columns if '_pixels' in x],
        'hue': [x for x in df.columns if 'hue_' in x],
        'sat': [x for x in df.columns if 'sat_' in x],
        'val': [x for x in df.columns if 'val_' in x],
        'red': [x for x in df.columns if 'red_' in x],
        'green': [x for x in df.columns if 'green_' in x],
        'blue': [x for x in df.columns if 'blue_' in x],
        'gray': [x for x in df.columns if 'gray_' in x],
    }
    # Collect all options.
    col_options = {
        'sharpness': [(x.split('_')[-1], x) for x in columns['sharpness']],
        'data_loss': [(x.split('_')[0], x) for x in columns['data_loss']],
        'hue': [(x.split('_')[-1], x) for x in columns['hue']],
        'sat': [(x.split('_')[-1], x) for x in columns['sat']],
        'val': [(x.split('_')[-1], x) for x in columns['val']],
        'red': [(x.split('_')[-1], x) for x in columns['red']],
        'green': [(x.split('_')[-1], x) for x in columns['green']],
        'blue': [(x.split('_')[-1], x) for x in columns['blue']],
        'gray': [(x.split('_')[-1], x) for x in columns['gray']],
    }
    drop_options = [
        ('Hue', 'hue'),
        ('Saturation', 'sat'),
        ('Value/brightness', 'val'),
        ('Red', 'red'),
        ('Green', 'green'),
        ('Blue', 'blue'),
        ('Gray', 'gray'),
    ]
    # Prepare arguments for each plot group.
    args = {
        'sharpness': {
            'col_desc': 'Reduction:',
            'x_range': (
                df[columns['sharpness'][0]].min(),
                df[columns['sharpness'][0]].max()
            ),
            'slider': widgets.FloatRangeSlider,
            'format': '.1f',
            'xmin': df[columns['sharpness'][0]].min(),
            'xmax': df[columns['sharpness'][0]].max(),
            'step': 0.1,
        },
        'data_loss': {
            'col_desc': 'Data loss:',
            'x_range': (
                df[columns['data_loss'][0]].min(),
                df[columns['data_loss'][0]].max()
            ),
            'slider': widgets.FloatRangeSlider,
            'format': '.3f',
            'xmin': 0,
            'xmax': 1,
            'step': 0.001,
        },
        'channels': {
            'col_desc': 'Quantile:',
            'x_range': (0, 255),
            'slider': widgets.IntRangeSlider,
            'format': 'd',
            'xmin': 0,
            'xmax': 255,
            'step': 1,
        }
    }

    # Build plot widgets.
    if group == 'channels':
        col = columns['hue'][0]
        opt = col_options['hue']
    else:
        col = columns[group][0]
        opt = col_options[group]
    plot_wdgts = {
        'column': widgets.SelectionSlider(
            options=opt,
            description=args[group]['col_desc'],
            continuous_update=False,
        ),
        'bins': widgets.IntSlider(
            30, 10, 100,
            step=5, description='Bins:',
            continuous_update=False,
        ),
        'x_range': args[group]['slider'](
            value=args[group]['x_range'],
            min=args[group]['xmin'],
            max=args[group]['xmax'],
            step=args[group]['step'],
            description='Range:',
            continuous_update=False,
            readout_format=args[group]['format'],
        ),
        'log_y': widgets.ToggleButton(
            value=False,
            description='log y',
            button_style='',
            icon=''
        )
        #'log_y': widgets.Checkbox(
        #    value=False, description='Log scale',
        #)
    }
    if group == 'channels':
        plot_wdgts['dropdown'] = widgets.Dropdown(
            options=drop_options,
            value=drop_options[0][1],
            description='Plot:',
        )
    else:
        plot_wdgts['dropdown'] = None

    # Build tiles_widgets
    tile_wdgts = {
        'nrows': widgets.Dropdown(
            options=[2, 4, 8],
            value=2,
            description='Rows:',
        ),
        'ncols': widgets.Dropdown(
            options=[4, 8, 16, 32, 64],
            value=16,
            description='Columns:',
        ),
        'px': widgets.Dropdown(
            options=[32, 64, 128, 256],
            value=64,
            description='Pixel width:',
        )
    }
    # Load first tiles.
    x_range = (args[group]['xmin'], args[group]['xmax'])
    first_tiles = get_tiles(df, col, x_range, nrows, ncols, px)
    # Create image widget
    tiles = widgets.Image(
        value=first_tiles,
        layout=widgets.Layout(width='100%')
    )
    tiles.__dict__['column'] = col
    tiles.__dict__['x_range'] = args[group]['x_range']
    tiles.__dict__['nrows'] = nrows
    tiles.__dict__['ncols'] = ncols
    tiles.__dict__['px'] = px
    # Link widgets to tile images.

    def update_tiles(change):
        for key, val in tiles.__dict__.items():
            if val == change['old']:
                tiles.__dict__[key] = change['new']

    plot_wdgts['column'].observe(update_tiles, names='value')
    plot_wdgts['x_range'].observe(update_tiles, names='value')
    tile_wdgts['nrows'].observe(update_tiles, names='value')
    tile_wdgts['ncols'].observe(update_tiles, names='value')
    tile_wdgts['px'].observe(update_tiles, names='value')

    # Button to plot new tiles.
    button_tiles = widgets.Button(
        description="Refresh tiles!",
    )

    def on_button_tiles_clicked(b):
        tiles.value = get_load_image(
            nrows=tiles.__dict__['nrows'],
            ncols=tiles.__dict__['ncols'],
            px=tiles.__dict__['px']
        )
        tiles.value = get_tiles(
            df,
            tiles.__dict__['column'],
            tiles.__dict__['x_range'],
            nrows=tiles.__dict__['nrows'],
            ncols=tiles.__dict__['ncols'],
            px=tiles.__dict__['px']
        )
    button_tiles.on_click(on_button_tiles_clicked)

    # Selection text.
    selection = widgets.Textarea(
        value='',
        placeholder="Copypastable selection will appear here!",
        description='Selection:',
        layout=widgets.Layout(width='100%', height='100px')
    )
    selection.__dict__['selections'] = []

    # Button to add stuff to selection.
    button_selection = widgets.Button(
        description="Add to selection!",
    )

    def on_button_selection_clicked(b):
        # Get widget values.
        column = plot_wdgts['column'].value
        x_range = plot_wdgts['x_range'].value
        # Add selection text to selections.
        if len(selection.__dict__['selections']) > 0:
            selection.__dict__['selections'][-1] += ' |'
        selection.__dict__['selections'].append(
            f"    "
            f"((df['{column}'] >= {x_range[0]}) & "
            f"(df['{column}'] <= {x_range[1]}))"
        )
        # Prepare suffix, prefix and indents.
        prefix = ['bad_tiles = df[(']
        suffix = [')]']
        # Build text.
        string = '\n'.join(prefix + selection.__dict__['selections'] + suffix)
        selection.value = string

    button_selection.on_click(on_button_selection_clicked)

    button_clear = widgets.Button(
        description="Clear selection!",
    )

    def on_button_clear_clicked(b):
        # Clear selections.
        selection.__dict__['selections'] = []
        selection.value = ''
    button_clear.on_click(on_button_clear_clicked)

    # Link drop menu.
    def update_column_options(change):
        plot_wdgts['column'].options = col_options[change['new']]
        plot_wdgts['column'].value = columns[change['new']][0]

    if plot_wdgts['dropdown'] is not None:
        plot_wdgts['dropdown'].observe(update_column_options, names='value')

    return (
        plot_wdgts,
        tiles, button_tiles, tile_wdgts,
        selection, button_selection, button_clear
    )


def get_ui(
    plot_wdgts: dict,
    tile_wdgts: dict,
    button_tiles: widgets.Widget,
    button_selection: widgets.Widget,
    button_clear: widgets.Widget,
):
    # Change layout:
    plot_wdgts['column'].layout = widgets.Layout(width='100%')
    plot_wdgts['bins'].layout = widgets.Layout(width='100%')
    plot_wdgts['x_range'].layout = widgets.Layout(width='100%')
    plot_wdgts['log_y'].layout = widgets.Layout(width='10%')
    tile_wdgts['nrows'].layout = widgets.Layout(width='100%')
    tile_wdgts['ncols'].layout = widgets.Layout(width='100%')
    tile_wdgts['px'].layout = widgets.Layout(width='100%')
    button_tiles.layout = widgets.Layout(width='100%')
    button_selection.layout = widgets.Layout(width='20%')
    button_clear.layout = widgets.Layout(width='20%')

    if plot_wdgts['dropdown'] is None:
        top = HBox([
            plot_wdgts['column'],
            plot_wdgts['bins']
        ])
        del plot_wdgts['dropdown']
    else:
        plot_wdgts['dropdown'].layout = widgets.Layout(width='50%')
        top = HBox([
            plot_wdgts['dropdown'],
            plot_wdgts['column'],
            plot_wdgts['bins']
        ])
    middle = HBox([
        plot_wdgts['x_range'],
        plot_wdgts['log_y'],
        button_selection,
        button_clear,
    ])
    bottom_ui = HBox([
        button_tiles,
        tile_wdgts['nrows'],
        tile_wdgts['ncols'],
        tile_wdgts['px'],
    ])
    top_ui = VBox([top, middle])
    return top_ui, bottom_ui


def get_plot_func(df, dropdown=False):
    if dropdown:
        def plot_func(column, bins, x_range, log_y, dropdown):
            x = df[column][
                (df[column] >= x_range[0]) &
                (df[column] <= x_range[1])
            ]
            plt.figure(figsize=(15, 5))
            plt.hist(x, bins=bins)
            plt.xlim(x_range)
            if log_y and len(x) > 0:
                plt.yscale('log')
            plt.show()
    else:
        def plot_func(column, bins, x_range, log_y):
            x = df[column][
                (df[column] >= x_range[0]) &
                (df[column] <= x_range[1])
            ]
            plt.figure(figsize=(15, 5))
            plt.hist(x, bins=bins)
            plt.xlim(x_range)
            if log_y and len(x) > 0:
                plt.yscale('log')
            plt.show()
    return plot_func


def Explore(
    metadata: pd.DataFrame,
    sharpness: bool = False,
    data_loss: bool = False,
    channels: bool = False,
):
    """Explore metadata to discover outliers.

    This function can be used for easy exploration of the metadata 
    generated during cutting of the tiles. By exploring the histograms
    it is easy to detect outlying values in certain preprocessing 
    metrics.

    Args:
        metadata (pd.DataFrame): Metadata generated during cutting.
        sharpness (bool, optional): Explore sharpness. 
            Defaults to False.
        data_loss (bool, optional): Explore data loss. 
            Defaults to False.
        channels (bool, optional): Explore different channel quantiles.
            Defaults to False.

    Raises:
        ValueError: ``sharpness``, ``data_loss`` and ``channels`` are 
            all set to False.
        ValueError: More than one from ``sharpness``, ``data_loss`` and
        ``channels`` is set to ``True``.
    """
    # Check options.
    options = ['sharpness', 'data_loss', 'channels']
    if sum([sharpness, data_loss, channels]) == 0:
        raise ValueError(
            f'Please set one of the options {options} to True.')
    if sum([sharpness, data_loss, channels]) > 1:
        raise ValueError(
            f'Please set only one of the options {options} to True.')
    if sharpness:
        what_to_plot = 'sharpness'
        dropdown = False
    if data_loss:
        what_to_plot = 'data_loss'
        dropdown = False
    if channels:
        what_to_plot = 'channels'
        dropdown = True

    plot_w, t, b_t, tile_w, selection_w, b_s, b_c = get_widgets(
        metadata, what_to_plot)
    top_ui, bottom_ui = get_ui(plot_w, tile_w, b_t, b_s, b_c)
    plot_func = get_plot_func(metadata, dropdown)
    out = widgets.interactive_output(plot_func, plot_w)
    display(top_ui, out, bottom_ui, t, selection_w)

def plot_on_thumbnail(
    df: pd.DataFrame, 
    max_thumbnails: int = 5,
    min_tiles: int = None,
) -> dict:
    """Plot tiles in the metadata dataframe onto thumbnails.

    Args:
        df (pd.DataFrame): Pandas dataframe with metadata.
        max_thumbnails (int, optional): Maximum number of thumbnails to
            find. Defaults to 5.
        min_tiles (int, optional): Minimum number of tiles 
            necessary to plot on thumbnail. Defaults to None.

    Raises:
        TypeError: Dataframe is the wrong type.

    Returns:
        dict: Dictionary with annotated slide thumbnails.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Expected {} not {}'.format(
            pd.DataFrame, type(df)
        ))
    # See if there are multiple slides.
    slides = df.slide_name.unique()
    if min_tiles is not None:
        # Discard slides with too few occurrences.
        new_slides = []
        for slide in slides:
            occurrences = len(df[df.slide_name == slide])
            if occurrences >= min_tiles:
                new_slides.append(slide)
        if len(new_slides) == 0:
            print(f'No slides found with minimum of {min_tiles} tiles!')
        slides = [x for x in slides if len(df[df.slide_name == x]) >= min_tiles]
    if len(slides) > max_thumbnails:
        slides = np.random.choice(slides, max_thumbnails)
    # Load thumbnails.
    thumbnails = {}
    for slide in slides:
        tile_path = df[df.slide_name==slide].path.tolist()[0]
        data_path = os.path.dirname(tile_path.split('tiles')[0])
        # Find thumbnail and downsample.
        for x in os.scandir(data_path):
            if 'thumbnail' in x.name:
                downsample = x.name.split('_')[-1].split('.')[0]
                if downsample.isnumeric():
                    thumbnails[slide] = (Image.open(x.path), int(downsample))

    # Generate annotations.
    annotated_thumbnails = {}
    for slide in slides:
        # Prepare thumbnail.
        thumbnail, downsample = thumbnails[slide]
        annotated_thumbnail = thumbnail.copy()
        annotated = ImageDraw.Draw(annotated_thumbnail)
        # Get tile coords.
        tmp = df[df.slide_name == slide]
        coords = list(zip(tmp['x'],tmp['y'],tmp['width']))
        # Draw.
        for x, y, width in coords:
            w_d = width/downsample
            x_d = x/downsample
            y_d = y/downsample
            annotated.rectangle(
                [x_d, y_d, x_d+w_d, y_d+w_d], 
                outline='blue', width=2
            )
        annotated_thumbnails[slide] = annotated_thumbnail
    return annotated_thumbnails


def plot_tiles(df: pd.DataFrame, nrows: int = 4, ncols: int = 16, px: int = 64) -> Image.Image:
    """Return a random collection of tiles from given metadata dataframe.

    Args:
        df (pd.DataFrame): Dataframe with metadata
        nrows (int, optional): Number of rows. Defaults to 4.
        ncols (int, optional): Number of columns. Defaults to 16.
        px (int, optional): Width of individual tiles. Defaults to 64.

    Raises:
        TypeError: Dataframe is the wrong type.

    Returns:
        Image.Image: Collection of random tiles.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError('Expected {} not {}'.format(
            pd.DataFrame, type(df)
        ))
    if len(df) == 0:
        print('No images.')
        return
    df = df.sample(n=min(len(df), nrows*ncols))
    tiles = get_tiles(df, nrows=nrows, ncols=ncols, px=px, to_bytes=False)
    return tiles


def plot_ranges(
    df: pd.DataFrame,
    column: str,
    ranges: List[Tuple[float]],
    nrows: int = 4,
    ncols: int = 8,
    px: int = 64,
) -> dict:
    """Plot random images from ranges from the given column,

    Args:
        df (pd.DataFrame): Dataframe with metadata.
        column (str): Desired column from the dataframe.
        ranges (List[Tuple[float]]): Ranges to be plotted  in format
            [(low,high), ...]
        nrows (int, optional): Number of rows. Defaults to 4.
        ncols (int, optional): Number of columns. Defaults to 16.
        px (int, optional): Width of individual tiles. Defaults to 64.

    Raises:
        TypeError: Dataframe is the wrong type.
        ValueError: Column not in dataframe.
        ValueError: Ranges in the wrong format.

    Returns:
        dict: [description]
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Expected {} not {}'.format(
            pd.DataFrame, type(df)
        ))
    if column not in df.columns:
        raise ValueError(f'Column {column} not in the given dataframe.')
    range_tiles = {}
    for low, high in ranges:
        if low > high:
            raise ValueError(
                f'{low} > {high}. Please give list of ranges in format '
                '[(low,high), (low,high), ...]'
            )
        tiles = get_tiles(df, column='sat_0.1', x_range=(low,high), 
                          nrows=nrows, ncols=ncols, px=px, to_bytes=False)
        range_tiles[f'{low}-{high}'] = tiles
    return range_tiles