#!/usr/bin/env python3
import argparse
import logging

import tqdm

from speech_datasets.transform import Transformation
from speech_datasets.utils.readers import file_reader_helper
from speech_datasets.utils.io_utils import get_commandline_args
from speech_datasets.utils.types import CMVNStats
from speech_datasets.utils.writers import write_cmvn_stats

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cepstral mean and variance normalization (CMVN) "
                    "statistics. Can compute CMVN statistics either globally, "
                    "per-speaker, or per-utterance. Writes output in Kaldi "
                    "mat format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filetype", type=str, default="hdf5", choices=["mat", "hdf5"],
                        help="Specify the file format for the rspecifier. "
                             '"mat" is the matrix format in kaldi',)
    parser.add_argument("--preprocess-conf", type=str, default=None,
                        help="The configuration file for the pre-processing")
    parser.add_argument("--cmvn-type", type=str, default="global",
                        choices=["global", "speaker", "utterance"])
    parser.add_argument("--spk2utt", type=str, default=None,
                        help="A text file of speaker to utterance-list map.")
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument("rspecifier", type=str,
                        help="Read specifier for feats, e.g. ark:some.ark or "
                             "scp,ark:some.scp,some.ark")
    parser.add_argument("wfilename", type=str, help="File to write CMVN stats to")
    args = parser.parse_args()

    if args.cmvn_type == "speaker" and args.spk2utt is None:
        raise argparse.ArgumentError(
            args.spk2utt, f"--spk2utt argument must be given when cmvn-type is 'speaker'")

    return args


def main():
    args = parse_args()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARN,
                        format=logfmt)
    logger.info(get_commandline_args())

    if args.cmvn_type == "speaker":
        logger.info("Performing as speaker CMVN mode")
        utt2spk_dict = {}
        with open(args.spk2utt) as f:
            for line in f:
                spk, utts = line.rstrip().split(None, maxsplit=1)
                for utt in utts.split():
                    utt2spk_dict[utt] = spk

        def utt2spk(x):
            return utt2spk_dict[x]

    else:
        logger.info(f"Performing as {args.cmvn_type} CMVN mode")
        if args.spk2utt is not None:
            logger.warning(f"spk2utt is not used for {args.cmvn_type} CMVN mode")

        if args.cmvn_type == "utterance":
            def utt2spk(x):
                return x

        else:  # args.cmvn_type == "global"
            def utt2spk(x):
                return None

    if args.preprocess_conf is not None:
        preprocessing = Transformation(args.preprocess_conf)
        logger.info("Apply preprocessing: {}".format(preprocessing))
    else:
        preprocessing = None

    # Calculate stats for each "speaker"
    cmvn_stats, n = {}, 0
    for utt, matrix in tqdm.tqdm(file_reader_helper(
            args.rspecifier, args.filetype, transform=preprocessing)):
        # Init at the first seen of the spk
        spk = utt2spk(utt)
        spk_stats = CMVNStats(
            count=matrix.shape[0], sum=matrix.sum(axis=0),
            sum_squares=(matrix ** 2).sum(axis=0))
        if spk not in cmvn_stats:
            cmvn_stats[spk] = spk_stats
        else:
            cmvn_stats[spk] += spk_stats
        n += 1
    logger.info(f"Processed {n} utterances")
    assert n > 0, n

    write_cmvn_stats(args.wfilename, args.cmvn_type, cmvn_stats)


if __name__ == "__main__":
    main()
