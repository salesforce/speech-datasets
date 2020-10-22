#!/usr/bin/env python

# Copyright 2020 Salesforce (Aadyot Bhatnagar)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""pytorch dataset and dataloader implementation for chainer training."""
from collections import defaultdict, OrderedDict
import itertools
import logging
import os
import shutil
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

import torch
import torch.utils.data
import torch.distributed as dist

from speech_datasets.text import SentencepieceTokenizer
from speech_datasets.transform import Transformation
from speech_datasets.utils import get_root
from speech_datasets.utils.io_utils import consolidate_utt_info, get_combo_idx
from speech_datasets.utils.readers import HDF5Reader, KaldiReader


logger = logging.getLogger(__name__)


def get_valid_datasets(task: str) -> Set[str]:
    """Returns the set of valid datasets that can be loaded."""
    assert task.lower() in ["asr", "tts"], task.lower()
    root = get_root()
    non_datasets = {"TEMPLATE", "COMBINE"}
    return {d.lower() for d in os.listdir(root) if d not in non_datasets
            and os.path.isdir(os.path.join(root, d, f"{task}1"))}


def validate_datasets(datasets: List[str], task: str, feats_type: str) \
        -> Dict[str, List[str]]:
    """Makes sure that the dataset names given (in form '<dataset>/<sub_dataset>')
    are valid, and that they have been prepared appropriately. Returns a
    Dict[str, Set[str]] dataset2subs, where dataset2subs[dataset] contains all
    sub-speech_datasets of dataset that were given."""
    dataset2subs = defaultdict(set)
    for d in datasets:
        try:
            dataset, sub = d.split("/", maxsplit=1)
        except ValueError:
            raise ValueError(
                f"Datasets must be specified as <dataset>/<sub_dataset> ",
                f"(e.g. wsj/train_si284, librispeech/dev-other, etc.), "
                f"but got {d} instead")
        dataset2subs[dataset.lower()].add(sub)

    # Make sure the task, feature type, and speech_datasets are all valid
    assert task.lower() in ["asr", "tts"], task.lower()
    assert feats_type.lower() in ["raw", "fbank", "fbank_pitch"], feats_type.lower()
    valid = get_valid_datasets(task)
    invalid = set(dataset2subs.keys()).difference(valid)
    assert len(invalid) == 0, \
        f"Invalid datasets: {invalid}. Valid {task} datasets are: {valid}."

    for dataset, subs in dataset2subs.items():
        root = os.path.join(get_root(), dataset, f"{task}1", "dump", feats_type)
        if not os.path.exists(root):
            valid_subs = {}
        else:
            valid_subs = {d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d))}
            valid_subs = valid_subs.difference({"orig", "no_short"})

        invalid_subs = {f"{dataset}/{s}" for s in subs.difference(valid_subs)}
        valid_subs = {f"{dataset}/{s}" for s in valid_subs}
        if len(valid_subs) == 0:
            raise FileNotFoundError(
                f"{feats_type} data has not been dumped/prepared for dataset "
                f"{dataset}.")
        if len(invalid_subs) > 0:
            raise FileNotFoundError(
                f"The following are invalid splits of dataset {dataset}: "
                f"{invalid_subs}\nValid splits are: {valid_subs}")

    # Convert dataset2subs to a SORTED OrderedDict[str, List[str]]
    # This is to ensure reproducibility across different processes
    dataset2subs = OrderedDict((k, sorted(dataset2subs[k]))
                               for k in sorted(dataset2subs.keys()))
    return dataset2subs


def get_combo_scp(datasets: List[str], task: str, feats_type: str) -> \
        Tuple[Optional[str], Optional[str]]:
    """Checks the dataset combination directory to see if this particular
    combination of datasets has been prepared already. Returns the SCP file
    and data format (hdf5 or mat) if it has, (None, None) otherwise."""
    combo_idx = get_combo_idx(datasets, task)
    dirname = os.path.join(get_root(), "COMBINE", f"{task}1", "dump",
                           feats_type, str(combo_idx))
    if combo_idx < 0 or not os.path.isdir(dirname):
        return None, None

    try:
        with open(os.path.join(dirname, "archive_format")) as f:
            fmt = f.readline().rstrip()
            assert fmt in ["hdf5", "mat"], \
                f"Combined dataset must be dumped to 'hdf5' or 'mat' " \
                f"archives, but got {fmt} instead."

        with open(os.path.join(dirname, "data_format")) as f:
            scp = f"{dirname}/{f.readline().rstrip()}.scp"
            assert os.path.isfile(scp), f"{scp} is not a file."

        return scp, fmt

    except Exception as e:
        logger.warning(
            f"{feats_type} data has not been properly dumped/prepared for "
            f"combined dataset. Failed loading with exception:\n{str(e)}")
        logger.warning("Loading combined dataset from separate SCP's instead")
        return None, None


def get_dataset_scps(datasets: List[str], task: str, feats_type: str,
                     check_combine=True) -> \
        Tuple[List[str], type]:
    """Given a list of <dataset>/<split> specifiers, obtain the SCP file(s) that
    index all the datasets (and splits) that we care about."""
    scp_files = []
    fmts = []
    dsets = []
    fmt2reader = {"hdf5": HDF5Reader, "mat": KaldiReader}

    if check_combine:
        scp, fmt = get_combo_scp(datasets, task, feats_type)
        if scp is not None:
            return [scp], fmt2reader[fmt]

    dataset2subs = validate_datasets(datasets, task, feats_type)
    for dataset, subs in dataset2subs.items():
        root = os.path.join(get_root(), dataset, f"{task}1", "dump", feats_type)
        for sub in subs:
            name = f"{dataset}/{sub}"
            dsets.append(name)
            dirname = os.path.join(root, sub)
            try:
                with open(os.path.join(dirname, "archive_format")) as f:
                    fmt = f.readline().rstrip()
                assert fmt in fmt2reader.keys(), \
                    f"Dataset {name} must be dumped to one of " \
                    f"{set(fmt2reader.keys())} archives, but got {fmt} instead."
                fmts.append(fmt)

                with open(os.path.join(dirname, "data_format")) as f:
                    scp = f"{f.readline().rstrip()}.scp"
                scp_files.append(os.path.join(dirname, scp))

            except Exception:
                logger.error(f"{feats_type} data has not been properly dumped/"
                             f"prepared for dataset {name}.")
                raise

    if not all(f == fmts[0] for f in fmts):
        err = "\n".join(f"{dset}: {fmt}" for dset, fmt in zip(dsets, fmts))
        raise RuntimeError(
            "Expected all datasets to be dumped to the same archive "
            "format, but got:\n" + err)

    return scp_files, fmt2reader[fmts[0]]


class SpeechDataLoader(torch.utils.data.DataLoader):
    def __init__(self, datasets: Union[str, List[str]],
                 task="asr", precomputed_feats_type="raw",
                 text_filename="text",
                 transform_conf: Union[str, List[Dict[str, Any]]] = None,
                 batch_size=1, max_len=None, train=False, shuffle=False,
                 num_replicas=None, rank=None, ensure_equal_parts=True,
                 num_workers=1, data_cache_mb=4096,
                 spmodel=None, token_list=None):
        """
        :param datasets: a list of strings specifying which datasets to load.
            Each dataset should be formatted as `<dataset_name>/<split_name>`,
            e.g. `wsj/train_si284`, `swbd/rt03`, `librispeech/dev-other`, etc.
        :param task: either "asr" or "tts"
        :param precomputed_feats_type: "raw", "fbank", or "fbank_pitch". Tells
            the data loader where to look for archive files, and what sort of
            pre-computed features have been dumped to those archives.
        :param text_filename: name of the file associating each utterance to its
            transcription. Can be useful to override if you want to transcribe
            (for example) phones instead of characters.
        :param transform_conf: either the filename of a `yaml` file specifying a
            transformation, or a `List[Dict[str, Any]]` specifying that
            transformation. `None` means no data transformation.
        :param batch_size: batch size
        :param train: whether the dataset's transform should be applied in
            training mode
        :param shuffle: whether to shuffle the dataset
        :param max_len: the maximum average utterance length of a batch (after
            any data transformation is applied). The sum of utterance lengths
            in the batch is restricted to be less than `batch_size * max_len`.
        :param num_replicas: the number of distributed copies of the dataset
            being used, e.g. for training a `DistributedDataParallel` model.
            If you are running multiple jobs in parallel, and want each job to
            work on a different subset of the dataset, set this parameter to the
            total number of jobs. You probably don't need to specify this
            manually if you are using torch.distributed.
        :param rank: the index (amongst all distributed workers) of the current
            worker, e.g. for training a `DistributedDataParallel` model. If you
            are running multiple jobs in parallel, and want each job to work on
            a different subset of the dataset, set this parameter to the index
            of the current job. You probably don't need to specify this manually
            if you are using torch.distributed.
        :param ensure_equal_parts: Whether to ensure that all parallel processes
            receive the same number of batches to process. This should always be
            enabled if you are training with `DistributedDataParallel`, but you
            may wish to disable it if you are evaluating your model and wish to
            have each utterance processed exactly once.
        :param num_workers: the number of parallel processes to use to
            apply the data transformation (specified by `transform_conf`) to
            the data being loaded. Default is `1`. If `None`, the value used is
            `math.ceil(os.n_cpu() / num_replicas) - 1`.
        :param data_cache_mb: the number of megabytes the cache (for
            pre-fetching archive files into memory) can contain.
        :param spmodel: the path to a `sentencepiece` model to use to tokenize
            the text. Can be trained with BPE, word, or char.
        :param token_list: the path to a list of `sentencepiece` tokens to use
            use to tokenize the text. The indices in `token_list` override those
            in `spmodel`. By default, we will search the directory containing
            `spmodel` for a `tokens.txt` and use it if found.
        """
        # For using the next() syntax
        self.iter = None
        self.current_position = 0

        # Initialize parameters for distributed data loading
        is_dist = dist.is_available() and dist.is_initialized()
        if num_replicas is None:
            num_replicas = dist.get_world_size() if is_dist else 1
        self.num_replicas = num_replicas
        if rank is None:
            rank = dist.get_rank() if is_dist else 0
        self.rank = rank
        if rank >= num_replicas:
            raise ValueError("You must specify rank < n_replicas. "
                             f"(Got rank={rank}, n_replicas={num_replicas}).")

        # Build a sentencepiece tokenizer if desired
        self.tokenizer = None
        if spmodel is not None:
            if token_list is None:
                token_list = os.path.join(os.path.dirname(spmodel), "tokens.txt")
            elif isinstance(token_list, list):
                if not all(isinstance(item, str) for item in token_list):
                    token_list = None
            if (isinstance(token_list, str) or isinstance(token_list, Path)) and not os.path.isfile(token_list):
                token_list = None
            self.tokenizer = SentencepieceTokenizer(spmodel, token_list)

        # Get all the datasets & their sub-datasets, and validate them
        datasets = [datasets] if isinstance(datasets, str) else datasets
        assert len(datasets) > 0, "Cannot load 0 datasets"
        scp_files, reader_class = get_dataset_scps(datasets, task, precomputed_feats_type)

        try:
            self.aux_utt_info = {}
            for scp in scp_files:
                scp_dir = os.path.dirname(scp)
                self.aux_utt_info.update(consolidate_utt_info(
                    text=os.path.join(scp_dir, text_filename),
                    utt2spk=os.path.join(scp_dir, "utt2spk"),
                    utt2num_frames=os.path.join(scp_dir, "utt2num_frames")))
        except FileNotFoundError:
            self.aux_utt_info = {}
            scps, _ = get_dataset_scps(datasets, task, precomputed_feats_type)
            for scp in scps:
                scp_dir = os.path.dirname(scp)
                self.aux_utt_info.update(consolidate_utt_info(
                    text=os.path.join(scp_dir, text_filename),
                    utt2spk=os.path.join(scp_dir, "utt2spk"),
                    utt2num_frames=os.path.join(scp_dir, "utt2num_frames")))

        # Initialize the transform & determine how it re-samples inputs
        transform = Transformation(transform_conf, precomputed_feats_type)
        input_hz, output_hz = None, None
        for fn in transform.functions:
            if hasattr(fn, "sample_frequency"):
                if input_hz is None:
                    input_hz = fn.sample_frequency
                output_hz = fn.sample_frequency
            if hasattr(fn, "frame_shift_ms"):
                if input_hz is None:
                    input_hz = 1000 / fn.frame_shift_ms
                output_hz = 1000 / fn.frame_shift_ms

        # Get the approximate length of each utterance, post-transformation
        ratio = output_hz / input_hz if input_hz and output_hz else 1.0
        utt2len = {k: v["length"] * ratio for k, v in self.aux_utt_info.items()}

        # Combine all the relevant SCP files into a single temporary file, and
        # use this temporary file to initialize the actual dataset
        with NamedTemporaryFile(delete=True) as tmpfile:
            with open(tmpfile.name, "ab") as dst:
                for scp in scp_files:
                    with open(scp, "rb") as src:
                        shutil.copyfileobj(src, dst)

            dataset = reader_class(
                f"scp:{tmpfile.name}", return_dict=True, train=train,
                shuffle=shuffle, n_parts=num_replicas, i_part=rank,
                transform=transform, pre_fetch_next_epoch=True,
                num_workers=num_workers, data_cache_mb=data_cache_mb,
                batch_size=batch_size, max_len=max_len,
                utt2len=utt2len, ensure_equal_parts=ensure_equal_parts)

            super().__init__(dataset, batch_size=1, shuffle=False,
                             collate_fn=self.collate_fn)

    @property
    def shuffle(self):
        return self.dataset.shuffle

    @property
    def train(self):
        return self.dataset.train

    @property
    def epoch(self):
        return self.dataset.seed

    def set_epoch(self, epoch):
        self.iter = None
        self.dataset.seed = epoch
        self.current_position = 0

    def close(self):
        self.dataset.close()

    def __len__(self):
        return len(self.dataset)

    @property
    def num_utts(self):
        return self.dataset.num_utts

    def collate_fn(self, items):
        batch = []
        items = itertools.chain.from_iterable(items)
        for uttid, data in items:
            aux_info = self.aux_utt_info.get(uttid, {})
            aux_info.pop("length", None)
            data.update(aux_info)
            data["x"] = torch.from_numpy(data["x"]).float()
            if self.tokenizer is not None:
                data["labels"] = torch.tensor(
                    self.tokenizer.text2ids(data["text"]))
            data.pop("rate", None)
            data["uttid"] = uttid
            batch.append(data)
        return batch

    def next(self):
        """Return the next data batch."""
        if self.iter is None:
            self.iter = iter(self)

        # try to get the next data point from the iterator, but loop back to
        # the start if the reported length is too long. note that if length
        # is correct reported, we should not encounter any StopIteration
        try:
            ret = next(self.iter)
            self.current_position += 1
        except StopIteration:
            logger.warning(f"Reported len(self.dataset)={len(self)} for epoch "
                           f"{self.epoch}, but actual length is "
                           f"{self.current_position}. This is a bug.")
            self.iter = None
            return self.next()

        # Increment the epoch if we just hit the end of the current one
        if self.current_position == len(self):
            self.set_epoch(self.epoch + 1)
            self.iter = iter(self)

        return ret

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
