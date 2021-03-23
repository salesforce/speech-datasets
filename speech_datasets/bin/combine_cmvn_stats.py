import argparse

from speech_datasets.utils.readers import read_cmvn_stats
from speech_datasets.utils.writers import write_cmvn_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmvn_type", choices=["global", "speaker", "utterance"])
    parser.add_argument("--output_file", type=str)
    parser.add_argument("cmvn_stats_files", nargs="+")
    return parser.parse_args()


def combine_cmvn_dicts(stats_dicts):
    out_dict = {}
    for d in stats_dicts:
        for spk, val in d.items():
            if spk not in out_dict:
                out_dict[spk] = val
            else:
                out_dict[spk] += val
    return out_dict


def main():
    args = parse_args()
    out_dict = combine_cmvn_dicts(read_cmvn_stats(path, args.cmvn_type)
                                  for path in args.cmvn_stats_files)
    write_cmvn_stats(args.output_file, args.cmvn_type, out_dict)


if __name__ == "__main__":
    main()
