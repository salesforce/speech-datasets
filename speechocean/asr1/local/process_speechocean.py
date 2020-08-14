#!/usr/bin/env python3

import argparse
import collections
import glob
import os
import re
from typing import Dict, Set


def script_file_to_metadata(fname: str) -> Dict[str, str]:
    base = re.sub(r"\..*", "", os.path.basename(fname))
    assert len(base) == 6
    return {"channel": base[0], "speaker": base[1:5], "session": base[5]}


def metadata_to_wav_dir(metadata: Dict[str, str], base_dir: str) -> str:
    ch, spk, sess = metadata["channel"], metadata["speaker"], metadata["session"]
    return os.path.join(
        base_dir, "DATA", f"CHANNEL{ch}", "WAVE", f"SPEAKER{spk}", f"SESSION{sess}")


def process_transcript(transcript: str) -> str:
    # noise = Speaker noise, unintelligible utterances, and intermittent noises
    transcript = re.sub(r"<SPK/>|\*\*|<NON/>", "<NOISE>", transcript)
    # Ignore stationary noise, filler words, and non-primary-speaker noise
    transcript = re.sub(r"<STA/>|<FIL/>|<NPS/>", "", transcript)
    # Ignore punctuations, except apostrophes & <NOISE>
    transcript = re.sub(r"[^\w\s'<>]", "", transcript)
    transcript = re.sub(r"(\s)'", r"\1", re.sub(r"'(\s)", r"\1", transcript))
    # Remove any extra spaces & make everything lowercase
    return " ".join(transcript.split()).lower()


def get_utts_info(speaker: str, script_file: str, wav_dir: str) -> Dict[str, Dict[str, str]]:
    """Extracts the filename and transcript for all """
    utt_id = None
    utts_info = {}
    with open(script_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # The even lines (starting at 0) are formatted
            # <utt_id>\t<formatted transcript>
            if i % 2 == 0:
                utt_id = line.split("\t")[0]

            # The odd lines (starting at 0) are unformatted
            # \t<unformatted transcript w/ non-speech events>
            else:
                assert utt_id is not None
                fname = os.path.join(wav_dir, f"{utt_id}.WAV")
                transcript = process_transcript(line.split("\t")[1])
                if os.path.exists(fname):
                    utts_info[utt_id] = {"file": fname,
                                         "speaker": speaker,
                                         "transcript": transcript}
    return utts_info


def get_all_utts_info(base_dir: str, speaker_set: Set[str] = None) -> Dict[str, Dict[str, str]]:
    """Returns a dictionary whose key is an utterance ID, and whose values are a dictionary
    containing that utterance's file, speaker, and transcript."""
    spk2files = collections.defaultdict(list)
    scripts = sorted(glob.glob(os.path.join(base_dir, "DATA", "CHANNEL*", "SCRIPT", "*.TXT")))
    for script in scripts:
        metadata = script_file_to_metadata(script)
        speaker = metadata["speaker"]
        if speaker_set is None or speaker in speaker_set:
            spk2files[speaker].append((script, metadata_to_wav_dir(metadata, base_dir)))

    utts_info = collections.OrderedDict()
    for speaker, files in sorted(spk2files.items()):
        for (script, wav_dir) in files:
            utts_info.update(get_utts_info(speaker, script, wav_dir))

    return utts_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcdir", type=str, required=True,
                        help="directory with extracted SpeechOcean data")
    parser.add_argument("--dstdir", type=str, required=True,
                        help="directory to save wav.scp, text, utt2spk, and spk2gender")
    parser.add_argument("--data_prefix", type=str, required=True,
                        help="name of this dataset, e.g. {speechocean-066} for King-ASR-066")
    parser.add_argument("--speaker_list", type=str, default=None,
                        help="file containing the list of speaker IDs to use for making this "
                             "data split, one speaker ID per line")
    return parser.parse_args()


def process_spk_id(spk_id: str, data_prefix: str):
    return data_prefix + spk_id


def process_utt_id(utt_id: str, data_prefix: str):
    assert len(utt_id) == 9
    channel = utt_id[0]
    speaker = utt_id[1:5]
    session = utt_id[5]
    idx = utt_id[6:]
    return f"{process_spk_id(speaker, data_prefix)}-{channel}{session}{idx}"


def main():
    args = parse_args()

    # Get the list of speakers we care about
    speaker_set = None
    if args.speaker_list is not None:
        with open(args.speaker_list, "r", encoding="utf-8") as f:
            speaker_set = set(line.rstrip("\n") for line in f.readlines())

    # Get the genders of all the speakers we care about, and write spk2gender
    spk2gender = {}
    with open(os.path.join(args.srcdir, "TABLE", "SPEAKER.TXT"), "r", encoding="utf-8") as f:
        f.readline()  # first line is just column names
        for line in f:
            speaker, gender = line.split("\t")[:2]
            if speaker_set is None or speaker in speaker_set:
                spk2gender[speaker] = gender.lower()

    with open(os.path.join(args.dstdir, "spk2gender"), "w") as f:
        for spk, gender in sorted(spk2gender.items()):
            f.write(f"{process_spk_id(spk, args.data_prefix)} {gender}" + "\n")

    # Get the metadata about all the utterances from the speakers we care about,
    # and use it to write wav.scp, utt2spk, and text
    utts_info = get_all_utts_info(args.srcdir, speaker_set)
    with open(os.path.join(args.dstdir, "wav.scp"), "w") as wav:
        with open(os.path.join(args.dstdir, "utt2spk"), "w") as utt2spk:
            with open(os.path.join(args.dstdir, "text"), "w") as text:
                for utt_id, info in utts_info.items():
                    utt_id = process_utt_id(utt_id, args.data_prefix)
                    spk_id = process_spk_id(info["speaker"], args.data_prefix)

                    wav.write(f"{utt_id} {info['file']}" + "\n")
                    utt2spk.write(f"{utt_id} {spk_id}" + "\n")
                    text.write(f"{utt_id} {info['transcript']}" + "\n")


if __name__ == "__main__":
    main()
