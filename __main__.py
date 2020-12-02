import os
from os.path import dirname,join,exists

import openslide

from args import get_arguments
import histoprep as hp
from histoprep._helpers import remove_extension
from histoprep.metadata import combine_metadata


if __name__ == '__main__':
    args = get_arguments()

    if args.step == 'cut':
        # Collect all slide paths
        SLIDES = []
        for f in os.scandir(args.input_dir):
            if any(f.path.endswith(x) for x in ['tiff','tif','mrxs']):
                SLIDES.append(f)
        if not args.overwrite:
            # See which slides have been cut before.
            NOT_CUT = []
            for f in SLIDES:
                name = remove_extension(f.name)
                if not exists(join(args.output_dir,name,'metadata.csv')):
                    NOT_CUT.append(f)
            if len(NOT_CUT) == 0:
                print('All slides have been cut!')
                exit()
            num_cut = len(SLIDES)-len(NOT_CUT)
            if num_cut != 0:
                print(f'{num_cut} slide(s) had been cut before.')
            SLIDES = NOT_CUT
        # Loop through each slide and cut.
        TOTAL = str(len(SLIDES))
        print(f'HistoPrep will process {TOTAL} slides.')
        for i,f in enumerate(SLIDES):
            print(f'[{str(i).rjust(len(TOTAL))}/{TOTAL}] - {f.name}',end=' - ')
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
                sat_thresh=args.sat_thresh,
                downsample=args.downsample,
                max_background=args.max_bg,
            )
            # Cut cut cut away!
            cutter.cut(
                parent_dir=args.output_dir,
                overwrite=args.overwrite,
                image_format=args.image_format,
                quality=args.quality,
                )
        print(f'All {TOTAL} slides processed.')
        combine_metadata(args.parent_dir, save_csv=True)

    elif args.step == 'tma':
        raise NotImplementedError()

    elif args.step == 'label':
        raise NotImplementedError()