#!/usr/bin/env python3
import argparse
from distutils.util import strtobool
import logging

from speech_datasets.transform import Transformation
from speech_datasets.utils.readers import file_reader_helper
from speech_datasets.utils.io_utils import get_commandline_args
from speech_datasets.utils.writers import file_writer_helper

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="apply mean-variance normalization to files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument("--in-filetype", type=str, default="hdf5", choices=["mat", "hdf5"],
                        help="Specify the file format for the rspecifier. "
                             '"mat" is the matrix format in kaldi')
    parser.add_argument("--out-filetype", type=str, default="hdf5", choices=["mat", "hdf5"],
                        help="Specify the file format for the wspecifier. "
                             '"mat" is the matrix format in kaldi')

    parser.add_argument("--norm-means", type=strtobool, default=True,
                        help="Do mean normalization or not.")
    parser.add_argument("--norm-vars", type=strtobool, default=False,
                        help="Do variance normalization or not.")
    parser.add_argument("--reverse", type=strtobool, default=False,
                        help="Do reverse mode or not")
    parser.add_argument("--utt2spk", type=str, default=None,
                        help="A text file of utterance to speaker map.")
    parser.add_argument("--compress", type=strtobool, default=False,
                        help="Save in compressed format")
    parser.add_argument("--compression-method", type=int, default=2,
                        help="Specify the method (if mat) or gzip-level (if hdf5)")
    parser.add_argument("--cmvn-type", type=str, choices=["global", "speaker", "utterance"],
                        help="Type of CMVN to apply (global, per-speaker, or per-utterance)")
    parser.add_argument("stats_file", help="File containing CMVN stats.")
    parser.add_argument("rspecifier", type=str, help="Read specifier id, e.g. ark:some.ark")
    parser.add_argument("wspecifier", type=str, help="Write specifier id, e.g. ark:some.ark")

    args = parser.parse_args()
    if args.cmvn_type == "speaker" and args.utt2spk is None:
        raise argparse.ArgumentError(
            args.cmvn_type, "If cmvn-type is 'speaker', utt2spk must be provided.")

    return args


def main():
    args = parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logger.info(get_commandline_args())

    cmvn = Transformation([{"type": "cmvn",
                            "stats": args.stats_file,
                            "cmvn_type": args.cmvn_type,
                            "norm_means": args.norm_means,
                            "norm_vars": args.norm_vars,
                            "utt2spk": args.utt2spk,
                            "reverse": args.reverse}])

    with file_writer_helper(
        args.wspecifier,
        filetype=args.out_filetype,
        compress=args.compress,
        compression_method=args.compression_method,
    ) as writer:
        for utt, data in file_reader_helper(args.rspecifier, args.in_filetype,
                                            transform=cmvn, return_dict=True):
            writer[utt] = data


if __name__ == "__main__":
    main()
