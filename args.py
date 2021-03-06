import os
import argparse

TITLE = """ python3 HistoPrep {step} {arguments}

█  █  █  ██  ███  ███  ███  ███  ███ ███
█  █  █ █     █  █   █ █  █ █  █ █   █  █
████  █  ██   █  █   █ ███  ███  ██  ███
█  █  █    █  █  █   █ █    █  █ █   █
█  █  █  ██   █   ███  █    █  █ ███ █

             by Jopo666 (2021)
"""


def get_arguments():
    parser = argparse.ArgumentParser(
        usage=TITLE,
    )
    subparsers = parser.add_subparsers(
        title='Select one of the below',
        dest='step',
        metavar='')
    cut = subparsers.add_parser('cut',
                                help='Cut tiles from large histological slides.',
                                usage='python3 HistoPrep cut input_dir output_dir width {optional arguments}')
    dearray = subparsers.add_parser('dearray',
                                    help='Dearray an tissue microarray (TMA) slide.',
                                    usage='python3 HistoPrep dearray input_dir output_dir {optional arguments}')

    ### CUT ###
    cut.add_argument('input_dir',
                     help='Path to the slide directory.')
    cut.add_argument('output_dir',
                     help="Will be created if doesn't exist.")
    cut.add_argument('width', type=int,
                     help='Tile width.')
    cut.add_argument('--overlap', type=float, default=0.0, metavar='',
                     help='Tile overlap. [Default: %(default)s]')
    cut.add_argument('--max_bg', type=float, default=0.7, metavar='',
                     help='Maximum background percentage for a tile. [Default: %(default)s]')
    cut.add_argument('--downsample', type=int, default=32, metavar='',
                     help='Thumbnail downsample. [Default: %(default)s]')
    cut.add_argument('--threshold', type=int, default=None, metavar='',
                     help='Threshold for background detection. [Default: %(default)s]')
    cut.add_argument('--overwrite', action='store_true',
                     help='[Default: %(default)s]')
    cut.add_argument('--image_format', default='jpeg', metavar='',
                     choices=['jpeg', 'png'], help='Image format. [Default: %(default)s]')
    cut.add_argument('--quality', type=int, default=95, metavar='',
                     help='Quality for jpeg compression. [Default: %(default)s]')

    ### DEARRAY ###
    dearray.add_argument('input_dir',
                         help='Path to the slide directory.')
    dearray.add_argument('output_dir',
                         help="Will be created if doesn't exist.")
    dearray.add_argument('--downsample', type=int, default=64, metavar='',
                         help='Thumbnail downsample. [Default: %(default)s]')
    dearray.add_argument('--threshold', type=int, default=None, metavar='',
                         help='Threshold for background detection. [Default: %(default)s]')
    dearray.add_argument('--min_area', type=float, default=0.2, metavar='',
                         help='Minimum area for a spot. [Default: median_area*%(default)s]')
    dearray.add_argument('--max_area', type=float, default=None, metavar='',
                         help='Maximum area for a spot. [Default: median_area*%(default)s]')
    dearray.add_argument('--kernel_size', type=int, default=8, metavar='',
                         help='Kernel size for spot detection (give as a single integer). [Default: (%(default)s,%(default)s)]')
    dearray.add_argument('--cut', action='store_true',
                         help='Cut spots after dearraying. [Default: %(default)s]')
    dearray.add_argument('--width', type=int, default=512,
                         help='Tile width. [Default: %(default)s]')
    dearray.add_argument('--overlap', type=float, default=0.0, metavar='',
                         help='Tile overlap. [Default: %(default)s]')
    dearray.add_argument('--max_bg', type=float, default=0.7, metavar='',
                         help='Maximum background percentage for a tile. [Default: %(default)s]')
    dearray.add_argument('--overwrite', action='store_true',
                         help='Remove everything before dearraying and cutting. [Default: %(default)s]')
    dearray.add_argument('--image_format', default='jpeg', metavar='', choices=['jpeg', 'png'],
                         help='Image format (jpeg or png). [Default: %(default)s]')
    dearray.add_argument('--quality', type=int, default=95, metavar='',
                         help='Quality for jpeg compression. [Default: %(default)s]')

    args = parser.parse_args()

    # Check paths.
    if not os.path.exists(args.input_dir):
        raise IOError(f'Path {args.input_dir} not found.')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return args
