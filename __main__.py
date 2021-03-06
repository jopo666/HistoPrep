import os
from os.path import dirname, join, exists, isdir

import openslide

from args import get_arguments
import histoprep as hp
from histoprep.helpers._utils import remove_extension
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
    'czi'
]


def collect_paths(args):
    paths = []
    for f in os.scandir(args.input_dir):
        suffix = f.name.split('.')[-1]
        if suffix in allowed:
            paths.append(f)
    if not args.overwrite:
        # See which slides have been processed before.
        not_processed = []
        for f in paths:
            name = remove_extension(f.name)
            if not exists(join(args.output_dir, name, 'metadata.csv')):
                not_processed.append(f)
        if len(not_processed) == 0:
            print('All slides have been cut!')
            exit()
        num_cut = len(paths)-len(not_processed)
        if num_cut != 0:
            print(f'{num_cut} slide(s) had been cut before.')
        paths = not_processed
    return paths

def check_file(file):
    try:
        if f.name.endswith('czi'):
            OpenSlideCzi(f.path)
        else:
            openslide.OpenSlide(f.path)
    except:
        print(f'Slide broken! Skipping {f.name}')
        return False
    return True

def cut_tiles(args):
    # Collect all slide paths
    slides = collect_paths(args)
    # Loop through each slide and cut.
    total = str(len(slides))
    print(f'HistoPrep will process {total} slides.')
    for i, f in enumerate(slides):
        print(f'[{str(i).rjust(len(total))}/{total}] - {f.name}', end=' - ')
        if not check_file:
            continue
        # Prepare Cutter.
        cutter = hp.Cutter(
            slide_path=f.path,
            width=args.width,
            overlap=args.overlap,
            threshold=args.threshold,
            downsample=args.downsample,
            max_background=args.max_bg,
            create_thumbnail=True,
        )
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
    for i, f in enumerate(tma_arrays):
        print(f'[{str(i).rjust(len(total))}/{total}] - {f.name}', end=' - ')
        # Prepare Dearrayer.
        if not check_file:
            continue
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
