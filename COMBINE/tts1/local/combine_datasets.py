import argparse
import os
import shutil

from speech_datasets.utils import get_root
from speech_datasets.utils.io_utils import get_combo_idx
from speech_datasets.utils.types import str2bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["asr", "tts"])
    parser.add_argument("--write_dir", type=str2bool, default=True)
    parser.add_argument("datasets", nargs="+", type=str)
    args = parser.parse_args()

    # Ensure that all datasets are specified as <dataset>/<split>
    datasets = sorted(set(args.datasets))
    dataset_splits = [d.split("/", maxsplit=1) for d in datasets]
    assert all(len(d) == 2 for d in dataset_splits), \
        f"All datasets must be specified as <dataset>/<split>, but got " \
        f"{datasets} instead"

    # Verify that all datasets have been prepared
    dataset_dirs = [os.path.join(get_root(), ds[0], f"{args.task}1", "data", ds[1])
                    for ds in dataset_splits]
    assert all(os.path.isdir(d) for d in dataset_dirs), \
        f"Please make sure that all dataset splits are valid, and that all " \
        f"datasets you wish to combine have already been prepared by stage 1 " \
        f"of {args.task}.sh"

    # Get the index of this dataset combination (add to the registry if needed)
    idx = get_combo_idx(datasets, args.task)
    data_dir = os.path.join(get_root(), "COMBINE", f"{args.task}1", "data")
    if idx < 0:
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "registry.txt"), "a") as f:
            f.write(" ".join(datasets) + "\n")
        idx = get_combo_idx(datasets, args.task)

    if not args.write_dir:
        return idx

    # Create a directory for this dataset combo & prepare it
    dirname = os.path.join(data_dir, str(idx))
    os.makedirs(dirname, exist_ok=True)
    write_segments = any(os.path.isfile(os.path.join(d, "segments"))
                         for d in dataset_dirs)
    with open(os.path.join(dirname, "wav.scp"), "wb") as wav, \
            open(os.path.join(dirname, "text"), "wb") as text, \
            open(os.path.join(dirname, "utt2spk"), "wb") as utt2spk, \
            open(os.path.join(dirname, "segments"), "w") as segments:
        for d in dataset_dirs:

            # wav.scp, text, and utt2spk can just be concatenated on
            with open(os.path.join(d, "wav.scp"), "rb") as src_wav:
                shutil.copyfileobj(src_wav, wav)
            with open(os.path.join(d, "text"), "rb") as src_text:
                shutil.copyfileobj(src_text, text)
            with open(os.path.join(d, "utt2spk"), "rb") as src_utt2spk:
                shutil.copyfileobj(src_utt2spk, utt2spk)

            if write_segments:
                # If a segments file exists, we can just concatenate it on
                if os.path.isfile(os.path.join(d, "segments")):
                    with open(os.path.join(d, "segments"), "r") as src_segments:
                        shutil.copyfileobj(src_segments, segments)

                # Otherwise, we need to use wav.scp to create a dummy segments
                # line format is <segment_id> <record_id> <start_time> <end_time>
                # <start_time> = 0, <end_time> = -1 means use the whole recording
                else:
                    with open(os.path.join(d, "wav.scp"), "r") as src_wav:
                        for line in src_wav:
                            utt_id, _ = line.rstrip().split(None, maxsplit=1)
                            segments.write(f"{utt_id} {utt_id} 0.0 -1.0\n")

    return idx


if __name__ == "__main__":
    combo_idx = main()
    print(combo_idx)
