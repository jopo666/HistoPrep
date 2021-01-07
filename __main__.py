import os
from os.path import dirname, join, exists, isdir

import openslide

from args import get_arguments
import histoprep as hp
from histoprep._helpers import remove_extension


def collect_paths(args):
    paths = []
    allowed = []
    not_allowed = []
    for f in os.scandir(args.input_dir):
        try:
            suffix = f.name.split('.')[-1]
            if suffix in allowed:
                paths.append(f)
            elif suffix in not_allowed or isdir(f.path):
                continue
            else:
                openslide.OpenSlide(f.path)
                paths.append(f)
                allowed.append(suffix)
        except:
            not_allowed.append(suffix)
    if not args.overwrite:
        # See which slides have been processed before.
        not_processed = []
        for f in paths:
            name = remove_extension(f.name)
            if not exists(join(args.output_dir, name, 'thumbnail.jpeg')):
                not_processed.append(f)
        if len(not_processed) == 0:
            print('All slides have been cut!')
            exit()
        num_cut = len(paths)-len(not_processed)
        if num_cut != 0:
            print(f'{num_cut} slide(s) had been cut before.')
        paths = not_processed
    return paths


def cut_tiles(args):
    # Collect all slide paths
    slides = collect_paths(args)
    # Loop through each slide and cut.
    total = str(len(slides))
    print(f'HistoPrep will process {total} slides.')
    for i, f in enumerate(slides):
        print(f'[{str(i).rjust(len(total))}/{total}] - {f.name}', end=' - ')
        # Test if file is okay.
        try:
            openslide.OpenSlide(f.path)
        except:
            print(f'Slide broken! Skipping {f.name}')
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
        cutter.cut(
            parent_dir=args.output_dir,
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
        # Test if file is okay.
        try:
            openslide.OpenSlide(f.path)
        except:
            print(f'TMA array file broken! Skipping {f.name}')
            continue
        # Prepare Dearrayer.
        dearrayer = hp.Dearrayer(
            slide_path=f.path,
            threshold=args.threshold,
            downsample=args.downsample,
            min_area=args.min_area,
            max_area=args.max_area,
            kernel_size=(args.kernel_size, args.kernel_size),
            fontsize=args.fontsize,
            create_thumbnail=True,
        )
        # Dearray away!
        dearrayer.save(
            parent_dir=args.output_dir,
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
            "I don't know how you did that, but that's not allowed.")
