#!/usr/bin/env python

# Copyright 2020 Salesforce (Aadyot Bhatnagar)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""pytorch dataset and dataloader implementation for chainer training."""
from collections import defaultdict
import logging
import math
import os
import shutil
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist

from speech_datasets.text import build_tokenizer
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
        -> Dict[str, Set[str]]:
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
                f"(e.g. wsj/train_si284, librispeech/dev_other, etc.), "
                f"but got {d} instead")
        dataset2subs[dataset.lower()].add(sub)

    # Make sure the task, feature type, and speech_datasets are all valid
    assert task.lower() in ["asr", "tts"], task.lower()
    assert feats_type.lower() in ["raw", "fbank", "fbank_pitch"], feats_type.lower()
    valid = get_valid_datasets(task)
    invalid = set(dataset2subs.keys()).difference(valid)
    assert len(invalid) == 0, \
        f"Invalid datasets: {invalid}. Valid {task} datasets are: {valid}."

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


def get_dataset_scps(datasets: List[str], task: str, feats_type: str) -> \
        Tuple[List[str], type]:
    """Given a list of <dataset>/<split> specifiers, obtain the SCP file(s) that
    index all the datasets (and splits) that we care about."""
    scp_files = []
    fmts = []
    dsets = []
    fmt2reader = {"hdf5": HDF5Reader, "mat": KaldiReader}

    scp, fmt = get_combo_scp(datasets, task, feats_type)
    if scp is not None:
        return [scp], fmt2reader[fmt]

    dataset2subs = validate_datasets(datasets, task, feats_type)
    for dataset, subs in dataset2subs.items():
        for sub in subs:
            name = f"{dataset}/{sub}"
            dsets.append(name)
            dirname = f"{get_root()}/{dataset}/{task}1/dump/{feats_type}/{sub}"
            try:
                with open(os.path.join(dirname, "archive_format")) as f:
                    fmt = f.readline().rstrip()
                    assert fmt in ["hdf5", "mat"], \
                        f"Dataset {name} must be dumped to 'hdf5' or 'mat' " \
                        f"archives, but got {fmt} instead."
                    fmts.append(fmt)

                with open(os.path.join(dirname, "data_format")) as f:
                    scp_files.append(f"{dirname}/{f.readline().rstrip()}.scp")

            except Exception:
                logger.error(f"{feats_type} data has not been properly dumped/"
                             f"prepared for dataset {name}.")
                raise

    if not all(f == fmt for f in fmts):
        err = "\n".join(f"{dset}: {fmt}" for dset, fmt in zip(dsets, fmts))
        raise RuntimeError(
            "Expected all datasets to be dumped to the same archive "
            "format, but got:\n" + err)

    return scp_files, fmt2reader[fmt]


class SpeechDataLoader(torch.utils.data.DataLoader):
    def __init__(self, datasets: Union[str, List[str]],
                 task="asr", precomputed_feats_type="raw",
                 transform_conf: Union[str, List[Dict[str, Any]]] = None,
                 batch_size=1, shuffle=False, drop_last=False,
                 num_replicas=None, rank=None, spmodel=None,
                 n_transform_proc=None, data_cache_mb=4096):

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
        self.tokenizer = build_tokenizer("bpe", spmodel) if spmodel else None

        # Get all the datasets & their sub-datasets, and validate them
        datasets = [datasets] if isinstance(datasets, str) else datasets
        assert len(datasets) > 0, "Cannot load 0 datasets"
        scp_files, reader_class = get_dataset_scps(datasets, task, precomputed_feats_type)

        # HDF5Reader gets speaker & text info, but we need to furnish these
        # manually for the other readers
        self.aux_utt_info = {}
        if reader_class is not HDF5Reader:
            for scp in scp_files:
                self.aux_utt_info.update(consolidate_utt_info(
                    text=os.path.join(os.path.dirname(scp), "text"),
                    utt2spk=os.path.join(os.path.dirname(scp), "utt2spk")))

        # Combine all the relevant SCP files into a single temporary file, and
        # use this temporary file to initialize the actual dataset
        with NamedTemporaryFile(delete=True) as tmpfile:
            with open(tmpfile.name, "ab") as dst:
                for scp in scp_files:
                    with open(scp, "rb") as src:
                        shutil.copyfileobj(src, dst)

            transform = Transformation(transform_conf, precomputed_feats_type)
            dataset = reader_class(
                f"scp:{tmpfile.name}", return_dict=True,
                shuffle=shuffle, n_parts=num_replicas, i_part=rank,
                transform=transform, pre_fetch_next_epoch=True,
                nproc=n_transform_proc, capacity_mb=data_cache_mb)

            super().__init__(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=self.collate_fn, drop_last=drop_last)

    @property
    def shuffle(self):
        return self.dataset.shuffle

    @property
    def epoch(self):
        return self.dataset.seed

    @epoch.setter
    def epoch(self, e):
        self.dataset.seed = e

    def set_epoch(self, e):
        self.epoch = e

    def close(self):
        self.dataset.close()

    def __len__(self):
        n_utts = len(self.dataset)
        if self.drop_last:
            return math.floor(n_utts / self.batch_size)
        else:
            return math.ceil(n_utts / self.batch_size)

    def collate_fn(self, items):
        batch = []
        for uttid, data in items:
            if isinstance(data, dict):
                data.update(self.aux_utt_info.get(uttid, {}))
                data["x"] = torch.from_numpy(data["x"]).float()
                if self.tokenizer is not None:
                    data["labels"] = torch.tensor(
                        self.tokenizer.text2ids(data["text"]))

            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()

            batch.append(data)
        return batch

    def __iter__(self):
        yield from super().__iter__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
