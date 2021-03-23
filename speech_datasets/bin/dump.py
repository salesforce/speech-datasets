#!/usr/bin/env python3

# Copyright 2020 Salesforce Research (Aadyot Bhatnagar)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from distutils.util import strtobool
import logging

import kaldiio
import tqdm

from speech_datasets.transform import Transformation
from speech_datasets.utils.io_utils import get_commandline_args, consolidate_utt_info
from speech_datasets.utils.types import str_or_none, humanfriendly_or_none
from speech_datasets.utils.writers import file_writer_helper

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="read .wav files & ")
    parser.add_argument("--feature-config", default=None, type=str_or_none,
                        help="YAML file for feature extraction (if extracting any features)")
    parser.add_argument("--text-file", default=None,
                        help="file mapping utterance ID to transcript")
    parser.add_argument("--utt2spk-file", default=None,
                        help="file mapping utterance ID to speaker ID")

    parser.add_argument("--archive-format", type=str, default="hdf5", choices=["mat", "hdf5"],
                        help="Specify the file format for output. \"mat\" is the matrix format in kaldi")
    parser.add_argument("--sample-frequency", type=humanfriendly_or_none, default=None,
                        help="If the sampling rate is specified, resample the input.")
    parser.add_argument("--compress", type=strtobool, default=False, help="Save in compressed format")
    parser.add_argument("--compression-method", type=int, default=2,
                        help="Specify the method(if mat) or " "gzip-level(if hdf5)")
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument("--segments", type=str,
                        help="segments-file format: each line is either"
                             "<segment-id> <recording-id> <start-time> <end-time>"
                             "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5")
    parser.add_argument("rspecifier", type=str, help="WAV scp file")
    parser.add_argument("wspecifier", type=str, help="Write specifier")

    return parser.parse_args()


def main():
    args = parse_args()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logger.info(get_commandline_args())

    utt_text_speaker = consolidate_utt_info(
        scp=None, text=args.text_file, utt2spk=args.utt2spk_file)

    with kaldiio.ReadHelper(
        args.rspecifier, segments=args.segments
    ) as reader, file_writer_helper(
        args.wspecifier,
        filetype=args.archive_format,
        compress=args.compress,
        compression_method=args.compression_method,
        sample_frequency=args.sample_frequency,
        transform=Transformation(args.feature_config)
    ) as writer:
        for utt_id, (rate, wave) in tqdm.tqdm(reader, miniters=100, maxinterval=30):
            utt_dict = {"x": wave, "rate": rate}
            utt_dict.update(utt_text_speaker.get(utt_id, {}))
            try:
                writer[utt_id] = utt_dict
            except Exception as e:
                logger.warning(
                    f"Failed to process utterance {utt_id} with exception:\n{str(e)}")
                continue


if __name__ == "__main__":
    main()
