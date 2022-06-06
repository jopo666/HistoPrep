import logging
import os
import shutil
import time
from typing import Callable, Dict, List, Tuple, Union

import cv2
import numpy
from PIL import Image, ImageDraw

from ..helpers import remove_directory


def check_region(
    XYWH: tuple, dimensions: Tuple[int, int]
) -> Tuple[bool, Tuple[int, int]]:
    """Checks if region is out of bounds and calculates padding."""
    x, y, w, h = XYWH
    # Check dimensions.
    out_of_bounds = False
    padding = None
    if y > dimensions[0] or x > dimensions[1]:
        # Starting point overbound at either axis --> No way to fix.
        out_of_bounds = True
    elif y + h > dimensions[0] or x + w > dimensions[1]:
        # Height or width causes overbound --> read what we can and pad.
        out_of_bounds = True
        # Get allowed h and w --> padding.
        padding = (
            min(dimensions[0] - y, h),
            min(dimensions[1] - x, w),
        )
    return out_of_bounds, padding


def copy_to_scratch(src: str, dst: str, verbose: bool = True):
    """Copy slide to scratch directory."""
    if not os.path.exists(dst):
        # Copy only if the path doesn't exist.
        shutil.copyfile(src, dst, follow_symlinks=True)
    # See if there is a separate data folder.
    if src.endswith(".mrxs"):
        logging.debug("Copying the mrxs data files.")
        src_dir = src.replace(".mrxs", "")
        dst_dir = dst.replace(".mrxs", "")
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.scandir(src_dir):
            shutil.copyfile(f.path, os.path.join(dst_dir, f.name))
    # Return new path
    return dst


def find_level(preferred: int, maximum: int, level_dimensions: tuple) -> int:
    """Helper function to find a thumbnail level."""
    level = None
    for _level, dimensions in level_dimensions.items():
        if max(dimensions) <= maximum:
            level = _level
            if max(dimensions) <= preferred:
                break
    if level is None:
        raise ValueError(
            "No level with all dimensions less than {}. Dimensions for each"
            " level are:\n{}".format(maximum, level_dimensions)
        )
    return level


def check_level(level: int, level_dimensions: tuple) -> None:
    # Check level.
    if level < 0:
        # Indexing from the end.
        level += len(level_dimensions)
    if level not in level_dimensions:
        raise ValueError(
            "Level {} not available. Please choose from {}.".format(
                level, list(level_dimensions.keys())
            )
        )


def prepare_paths_and_directories(
    output_dir: str,
    slide_name: str,
    overwrite: bool,
) -> Dict[str, str]:
    """Prepare paths and directories."""
    # Define slide output directory.
    slide_dir = os.path.join(output_dir, slide_name)
    # Check both directory paths.
    for dir_path in [output_dir, slide_dir]:
        if os.path.exists(dir_path) and not os.path.isdir(dir_path):
            raise NotADirectoryError(
                "Slide directory {} exists and but isn't a directory.".format(dir_path)
            )
    # Prepare rest of the paths.
    tile_directory = os.path.join(slide_dir, "tiles")
    spot_directory = os.path.join(slide_dir, "spots")
    tile_metadata_path = os.path.join(slide_dir, "tile_metadata.csv")
    spot_metadata_path = os.path.join(slide_dir, "spot_metadata.csv")
    thumbnail_path = os.path.join(slide_dir, "thumbnail.jpeg")
    annotated_tiles_path = os.path.join(slide_dir, "annotated_thumbnail_tiles.jpeg")
    annotated_spots_path = os.path.join(slide_dir, "annotated_thumbnail_spots.jpeg")
    tissue_mask_path = os.path.join(slide_dir, "tissue_mask.jpeg")
    spot_mask_path = os.path.join(slide_dir, "spot_mask.jpeg")
    unreadable_tiles_path = os.path.join(slide_dir, "unreadable_tiles.csv")
    unreadable_spots_path = os.path.join(slide_dir, "unreadable_spots.csv")
    # Check output directory.
    if os.path.exists(slide_dir):
        if not overwrite:
            return
        for path in [
            tile_metadata_path,
            spot_metadata_path,
            thumbnail_path,
            annotated_spots_path,
            annotated_tiles_path,
            tissue_mask_path,
            spot_mask_path,
            unreadable_tiles_path,
            unreadable_spots_path,
        ]:
            if os.path.exists(path):
                os.remove(path)
        # Then remove tiles and spots.
        if os.path.exists(tile_directory):
            remove_directory(tile_directory)
        if os.path.exists(spot_directory):
            remove_directory(spot_directory)
    # Return all paths.
    return {
        "slide_directory": output_dir,
        "tile_directory": tile_directory,
        "spot_directory": spot_directory,
        "tile_metadata": tile_metadata_path,
        "spot_metadata": spot_metadata_path,
        "thumbnail": thumbnail_path,
        "annotated_tiles": annotated_tiles_path,
        "annotated_spots": annotated_spots_path,
        "tissue_mask": tissue_mask_path,
        "spot_mask": spot_mask_path,
        "unreadable_tiles": unreadable_tiles_path,
        "unreadable_spots": unreadable_spots_path,
    }


def annotate_thumbnail(
    thumbnail: Union[Image.Image, numpy.ndarray],
    downsample: Tuple[float, float],
    coordinates: List[Tuple[int, int, int, int]],
    numbers: numpy.ndarray = None,
) -> Image.Image:
    """Annotate thumbnail with tile coordinates."""
    # Draw tiles to the thumbnail.
    if isinstance(thumbnail, numpy.ndarray):
        thumbnail = Image.fromarray(thumbnail)
    # Make a copy.
    thumbnail = thumbnail.copy()
    annotated = ImageDraw.Draw(thumbnail)
    # Start drawing.
    for x, y, w, h in coordinates:
        y_d = round(y / downsample[0])
        h_d = round(h / downsample[0])
        x_d = round(x / downsample[1])
        w_d = round(w / downsample[1])
        # xy expects [(x0, y0), (x1, y1)].
        annotated.rectangle(
            xy=[(x_d, y_d), (x_d + w_d, y_d + h_d)],
            outline="red",
            width=2,
        )
    if numbers is not None:
        arr = numpy.array(thumbnail)
        for (x, y, w, h), number in zip(coordinates, numbers):
            y_d = round(y / downsample[0])
            h_d = round(h / downsample[0])
            x_d = round(x / downsample[1])
            w_d = round(w / downsample[1])
            # Write text.
            cv2.putText(
                img=arr,
                text=number,
                org=(x_d, y_d + h_d),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
            )
        thumbnail = Image.fromarray(arr)
    return thumbnail


def worker_initializer_fn(slide_reader):
    """Save slide reader as a global variable."""
    global __SLIDE_READER__
    __SLIDE_READER__ = slide_reader


def save_tile_worker(
    xywh: Tuple[int, int, int, int],
    level: int,
    tile_directory: str,
    tissue_threshold: int,
    preprocess_metrics: Callable,
    image_format: str,
    quality: int,
) -> Tuple[tuple, dict, float]:
    """Save a tile image and calculate preprocessing metrics. If a region cannot
    be read, returns the Exception instead.

    NOTE: Requires that each worker is initialized with `worker_initializer_fn`.
    """
    tic = time.perf_counter()
    # Unpack.
    x, y, w, h = xywh
    # Save basic metadata.
    tile_metadata = {
        "slide_name": __SLIDE_READER__.slide_name,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "level": level,
    }
    # Read region.
    try:
        tile = __SLIDE_READER__.read_region((x, y, w, h), level=level)
    except Exception as e:
        # Could not read region -> return the Exception and times as None.
        return xywh, e, None
    # Define tile path.
    filepath = os.path.join(
        tile_directory, "x{}_y{}_w{}_h{}.{}".format(x, y, w, h, image_format)
    )
    tile_metadata["path"] = os.path.realpath(filepath)
    # Update metadata with preprocessing metrics.
    if preprocess_metrics is not None:
        try:
            tile_metadata.update(preprocess_metrics(tile, tissue_threshold))
        except Exception as e:
            # Could not calculate preprocessing metrics.
            return xywh, e, None
    # Save image.
    if not os.path.exists(filepath):
        try:
            tile.save(filepath, quality=quality)
        except Exception as e:
            # Could not save region.
            return xywh, e, None
    # Return metadata and times.
    return xywh, tile_metadata, time.perf_counter() - tic


def save_spot_worker(
    spot_info: Tuple[int, int, int, int, str],
    spot_directory: str,
    image_format: str,
    quality: int,
):
    """Saves a single spot."""
    tic = time.perf_counter()
    # Unpack.
    x, y, w, h, number = spot_info
    # Create name.
    spot_name = "{}_spot_{}".format(__SLIDE_READER__.slide_name, number)
    # Save basic metadata.
    spot_metadata = {
        "slide_name": __SLIDE_READER__.slide_name,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "spot_number": number,
        "spot_name": spot_name,
    }
    # Read region.
    try:
        spot = __SLIDE_READER__.read_region((x, y, w, h))
    except Exception as e:
        # Could not read region.
        return spot_info, e, None
    # Define spot path.
    filepath = os.path.join(spot_directory, "{}.{}".format(spot_name, image_format))
    spot_metadata["path"] = filepath
    # Save image.
    if not os.path.exists(filepath):
        try:
            spot.save(filepath, quality=quality)
        except Exception as e:
            # Could not save spot.
            return spot_info, e, None
    # Return metadata and times.
    return spot_info, spot_metadata, time.perf_counter() - tic
