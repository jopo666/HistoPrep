from typing import Union
import warnings

from PIL import Image,ImageDraw
import pandas as pd
import numpy as np

__all__ = [
    'plot_tiles',
    'plot_on_thumbnail'
]

def plot_on_thumbnail(
        dataframe: pd.DataFrame,
        thumbnail: Image.Image,
        downsample: int
        ) -> Image.Image:
    """Plot all tiles in a dataframe to a thumbnail."""
    if not isinstance(dataframe,pd.DataFrame):
        raise ValueError('Expected {} not {}'.format(
            pd.DataFrame, type(dataframe)
            ))
    if not isinstance(thumbnail,Image.Image):
        raise ValueError('Expected {} not {}'.format(
            Image.Image, type(thumbnail)
            ))
    try:
        downsample = int(downsample)
    except:
        raise ValueError('Expected {} not {}'.format(
            int,type(downsample)
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
    for (x,y) in list(zip(x,y)):
        x_d = round(x/downsample)
        y_d = round(y/downsample)
        annotated.rectangle([x_d,y_d,x_d+w,y_d+h],outline='blue',width=4)
    return annotated_thumbnail

def plot_tiles(
        dataframe: pd.DataFrame,
        rows: int = 3, 
        cols: int = 3,
        max_pixels = 1_000_000,
        title_column = None,
        ) -> Image.Image:
    """Return a RANDOM collection of tiles from given DataFrame."""
    if not isinstance(dataframe,pd.DataFrame):
        raise ValueError('Expected {} not {}'.format(
            pd.DataFrame, type(dataframe)
            ))
    dataframe.sample(frac = 1)
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
    if len(images)==0:
        print('No images.')
        return
    else:
        shape = np.array(images[0]).shape
    # Then combine images to grid.
    images = [images[i:i + cols] for i in range(0, len(images), cols)]
    rows = []
    for row in images:
        while len(row)!=cols:
            row.append(np.ones(shape,dtype=np.uint8)*255)
        rows.append(np.hstack(row))
    summary = Image.fromarray(np.vstack(rows))
    if title_column is not None:
        print(f'{title_column.upper()}:')
        for row in [titles[i:i + cols] for i in range(0, len(titles), cols)]:
            row = [np.round(x,3) for x in row]
            [print(str(x).center(8), end='') for x in row]
            print()
    return resize(summary,max_pixels)

def resize(image,MAX_PIXELS=5_000_000):
    dimensions = np.array(image).shape[:2]
    width,height = dimensions
    factor = 0
    while width*height > MAX_PIXELS:
        factor += 1
        width = int(dimensions[0]/2**factor)
        height = int(dimensions[1]/2**factor)
    image = Image.fromarray(np.array(image).astype('uint8'))
    return image.resize((height,width))