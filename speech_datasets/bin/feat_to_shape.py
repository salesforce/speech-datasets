#!/usr/bin/env python3
import argparse
import logging
import sys

from speech_datasets.transform import Transformation
from speech_datasets.utils.readers import file_reader_helper
from speech_datasets.utils.io_utils import get_commandline_args, strtobool

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert feature to its shape",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument("--filetype", type=str, default="hdf5", choices=["mat", "hdf5", "sound"],
                        help="Specify the file format for the rspecifier.")
    parser.add_argument("--preprocess-conf", type=str, default=None,
                        help="The configuration file for the pre-processing")
    parser.add_argument("--mem-mapped", type=strtobool, default=False,
                        help="Whether to use memory-mapped data loaders (where available)")
    parser.add_argument("rspecifier", type=str,
                        help="Read specifier for feats. e.g. ark:some.ark")
    parser.add_argument("out", nargs="?", type=argparse.FileType("w"), default=sys.stdout,
                        help="The output filename. " "If omitted, then output to sys.stdout")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logger.info(get_commandline_args())

    if args.preprocess_conf is not None:
        preprocessing = Transformation(args.preprocess_conf)
        logger.info("Apply preprocessing: {}".format(preprocessing))
    else:
        preprocessing = None

    for utt, shape in file_reader_helper(
            args.rspecifier, args.filetype, return_shape=True, transform=preprocessing):
        shape_str = ",".join(map(str, shape))  # shape is a tuple of ints
        args.out.write("{} {}\n".format(utt, shape_str))


if __name__ == "__main__":
    main()
