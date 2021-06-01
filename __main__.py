import time
import logging
import sys
import os
from os.path import dirname, join, exists, isdir

import openslide
import numpy as np

from args import get_arguments
import histoprep as hp
from histoprep.helpers._utils import remove_extension, format_seconds
from histoprep._czi_reader import OpenSlideCzi

allowed = [
    'mrxs',
    'tiff',
    'svs',
    'tif',
    'ndpi',
    'vms',
    'vmu',
    'scn',
    'svslide',
    'bif',
    'czi',
]

# Define logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_etc(times: list, tic: float, num_left: int):
    """Print ETC and return times with new time added.

    Args:
        times (list): List of individual times.
        tic (float): Start time.
        num_left (int): Number of iterations left.

    Returns:
        [type]: [description]
    """
    times.append(time.time()-tic)
    etc = np.mean(times)*num_left
    print(f'ETC: {format_seconds(etc)}')
    return times


def collect_paths(args):
    paths = []
    for f in os.scandir(args.input_dir):
        suffix = f.name.split('.')[-1]
        if suffix in allowed:
            paths.append(f)
    if len(paths) == 0:
        logger.warning(
            'No slides found! Please check that input_dir '
            'you defined is correct!'
        )
        sys.exit()
    if not args.overwrite:
        # See which slides have been processed before.
        not_processed = []
        for f in paths:
            name = remove_extension(f.name)
            if not exists(join(args.output_dir, name, 'metadata.csv')):
                not_processed.append(f)
        if len(not_processed) == 0:
            print('All slides have been cut!')
            sys.exit()
        num_cut = len(paths)-len(not_processed)
        if num_cut != 0:
            print(f'{num_cut} slide(s) had been cut before.')
        paths = not_processed
    return paths


def check_file(file):
    try:
        if file.name.endswith('czi'):
            OpenSlideCzi(file.path)
        else:
            openslide.OpenSlide(file.path)
    except:
        logger.warning(f'Slide broken! Skipping {file.name}')
        return False
    return True


def check_downsamples(path, downsample):
    """Check if any close downsamples can be found."""
    r = openslide.OpenSlide(path)
    downsamples = [int(x) for x in r.level_downsamples]
    if downsample in downsamples:
        return downsample
    elif int(downsample*2) in downsamples:
        logger.warning(
            f'Downsample {downsample} not available, '
            f'using {int(downsample*2)}.'
        )
        return int(downsample*2)
    elif int(downsample/2) in downsamples:
        logger.warning(
            f'Downsample {downsample} not available, '
            f'using {int(downsample/2)}.'
        )
        return int(downsample/2)
    else:
        return None


def cut_tiles(args):
    # Collect all slide paths
    slides = collect_paths(args)
    # Loop through each slide and cut.
    total = str(len(slides))
    print(f'HistoPrep will process {total} slides.')
    # Initialise list of times for ETC
    times = []
    tic = None
    for i, f in enumerate(slides):
        print(f'[{str(i).rjust(len(total))}/{total}] - {f.name}', end=' - ')
        if not check_file(f):
            continue
        # Calculate ETC.
        if tic is None:
            print('ETC: ...')
        else:
            times = get_etc(times=times, tic=tic, num_left=len(slides)-i-1)
        # Start time.
        tic = time.time()
        # Check downsample.
        downsample = check_downsamples(f.path, args.downsample)
        if downsample is None:
            logger.warning(
                f'No downsample close to {args.downsample} available, '
                f'trying to generate a thumbnail image.'
            )
            downsample = args.downsample
        # Prepare Cutter.
        try:
            cutter = hp.Cutter(
                slide_path=f.path,
                width=args.width,
                overlap=args.overlap,
                threshold=args.threshold,
                downsample=downsample,
                max_background=args.max_bg,
                create_thumbnail=True,
            )
        except KeyboardInterrupt:
            logger.warning('KeyboardInterrupt detected. Shutting down.')
            sys.exit()
        except Exception as e:
            logger.warning(
                f'Something went wrong with error: "{e}"'
                f'\nSkipping slide {f.name}.'
            )
            continue
        
        # Cut cut cut away!
        cutter.save(
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            image_format=args.image_format,
            quality=args.quality,
        )
    print(f'All {total} slides processed.')


def dearray(args):
    # Collect all slide paths.
    tma_arrays = collect_paths(args)
    # Loop through each array and cut.
    total = str(len(tma_arrays))
    print(f'HistoPrep will process {total} TMA arrays.')
    # Initialise list of times for ETC
    times = []
    for i, f in enumerate(tma_arrays):
        print(f'[{str(i).rjust(len(total))}/{total}] - {f.name}', end=' - ')
        # Prepare Dearrayer.
        if not check_file(f):
            continue
        # Calculate ETC
        if i == 0:
            print('ETC: ...')
        else:
            times = get_etc(times=times, tic=tic, num_left=len(tma_arrays)-i-1)
        # Start time.
        tic = time.time()
        # Dearray!
        dearrayer = hp.Dearrayer(
            slide_path=f.path,
            threshold=args.threshold,
            downsample=args.downsample,
            min_area_multiplier=args.min_area,
            max_area_multiplier=args.max_area,
            kernel_size=(args.kernel_size, args.kernel_size),
            create_thumbnail=True,
        )
        # Dearray away!
        dearrayer.save_spots(
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            image_format=args.image_format,
            quality=args.quality,
        )
        if args.cut:
            dearrayer.save_tiles(
                width=args.width,
                overlap=args.overlap,
                max_background=args.max_bg,
                overwrite=args.overwrite,
                image_format=args.image_format,
                quality=args.quality,
            )
    print(f'All {total} TMA arrays processed.')


if __name__ == '__main__':
    args = get_arguments()

    if args.step == 'cut':
        cut_tiles(args)

    elif args.step == 'dearray':
        dearray(args)

    else:
        raise NotImplemented(
            "I don't know how you did that, but that's not allowed."
        )
