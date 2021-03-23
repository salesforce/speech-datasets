from abc import abstractmethod
import atexit
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import io
import logging
import math
import os
import sys
from typing import Any, Dict, List, Tuple
import weakref

import h5py
import kaldiio
import ray
import soundfile
from torch.utils.data import IterableDataset

from speech_datasets.transform import Transformation
from speech_datasets.utils.io_utils import parse_scp_file, split_scp_dict
from speech_datasets.utils.io_utils import FileQueue
from speech_datasets.utils.types import CMVNStats

logger = logging.getLogger(__name__)
_ray_refs = 0


def _weakref_close(reader_ref):
    """Closes a weakref to the reader (if not already dead). Each reader
    registers this function with atexit with a weakref to itself. We use a
    weakref atexit doesn't hold on to a reference to the reader, so the reader
    can actually be garbage collected before the program exits (if needed)."""
    reader = reader_ref()
    if reader is not None:
        reader.close()


def apply_transform(transform, key_datadict, train):
    key, datadict = key_datadict
    datadict['x'] = transform(**datadict, uttid_list=key, train=train)
    return datadict


@ray.remote(num_cpus=1, num_gpus=0)
class TransformActor:
    def __init__(self, transform, train):
        self.transform = transform
        self.train = train

    def apply(self, key_datadict):
        return apply_transform(transform=self.transform,
                               key_datadict=key_datadict,
                               train=self.train)


def file_reader_helper(rspecifier: str, filetype: str, train=True,
                       return_shape: bool = False, return_dict: bool = False,
                       transform: Transformation = None):
    """Read uttid and array in kaldi style

    This function might be a bit confusing as "ark" is used
    for HDF5 to imitate "kaldi-rspecifier".

    Args:
        rspecifier: Give as "ark:feats.ark" or "scp:feats.scp"
        filetype: "mat": kaldi-matrix, "hdf5": HDF5, "sound": raw soundfiles
        train: Whether to apply the transform in training mode or eval mode.
        return_shape: Return the shape of the matrix, instead of the matrix.
        return_dict: Return a dictionary containing all the relevant info about
                     the matrix & other associated data. For HDF5 only.
        transform: transformation to apply to signal (e.g. feature extraction,
                   speed perturbation, spectral augmentation, etc.)
    Returns:
        Generator[Tuple[str, np.ndarray], None, None]:

    Examples:
        Read from kaldi-matrix ark file:

        >>> for u, array in file_reader_helper('ark:feats.ark', 'mat'):
        ...     array

        Read from HDF5 file:

        >>> for u, array in file_reader_helper('ark:feats.h5', 'hdf5'):
        ...     array

    """
    if filetype == "mat":
        return KaldiReader(rspecifier, return_shape=return_shape, return_dict=return_dict,
                           transform=transform, train=train, num_workers=0)
    elif filetype == "hdf5":
        return HDF5Reader(rspecifier, return_shape=return_shape, return_dict=return_dict,
                          transform=transform, train=train, num_workers=0)
    elif filetype == "sound":
        return SoundReader(rspecifier, return_shape=return_shape, return_dict=return_dict,
                           transform=transform, train=train, num_workers=0)
    else:
        raise NotImplementedError(f"filetype={filetype}")


def read_cmvn_stats(path, cmvn_type):
    if cmvn_type == "global":
        d = {None: kaldiio.load_mat(path)}
    else:
        d = dict(kaldiio.load_ark(path))
    return {spk: CMVNStats.from_numpy(stats) for spk, stats in d.items()}


class HDF5SignalRateFile(h5py.File):
    def __getitem__(self, key):
        data = super().__getitem__(key)
        return data[()], data.attrs["rate"]


class BaseReader(IterableDataset):
    """Uses a .scp file to read data from .h5 files. Pre-fetches .h5 files into
    dictionaries stored in memory for efficient access."""
    def __init__(self, rspecifier, return_shape=False, return_dict=False,
                 transform: Transformation = None, train=False,
                 batch_size: int = None, max_len: int = None, utt2len=None,
                 ensure_equal_parts=True, pre_fetch_next_epoch=False,
                 data_cache_mb=2048, num_workers: int = 1,
                 n_parts=1, i_part=0, shuffle=False, seed=0):
        if ":" not in rspecifier:
            raise ValueError(
                f'Give "rspecifier" such as "scp:some.scp: {rspecifier}"')
        self.ark_or_scp, self.filepath = rspecifier.split(":", maxsplit=1)
        if self.ark_or_scp not in ["scp", "ark"]:
            raise ValueError(f"Filetype must be scp or ark, but got {self.ark_or_scp}")
        elif self.ark_or_scp == "scp":
            self._full_scp_dict = parse_scp_file(self.filepath)
        else:
            self._full_scp_dict = None
            if pre_fetch_next_epoch:
                raise ValueError(
                    f"Cannot pre-fetch next epoch if reading file directly. "
                    f"rspecifier={rspecifier}, pre_fetch_next_epoch=True. "
                    f"Change rspecifier to scp, or pre_fetch_next_epoch=False.")
            if shuffle:
                raise ValueError(
                    f"Cannot shuffle data if reading file directly. "
                    f"rspecifier={rspecifier}, shuffle=True. "
                    f"Change rspecifier to scp, or shuffle=False.")
            if max_len is not None:
                raise ValueError(
                    f"Cannot enforce dynamic batching if rreading file directly. "
                    f"rspecifier={rspecifier}, max_len={max_len}. "
                    f"Change rspecifier to scp, or max_len=None.")

        # For loading data from SCP
        self._n_parts = n_parts
        self._i_part = i_part
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._max_len = max_len
        self._utt2len = utt2len
        self._ensure_equal_parts = ensure_equal_parts
        self._bszs = None

        # For determining the data format to return
        self.return_dict = return_dict
        self.return_shape = return_shape and not return_dict
        self.train = train
        self.transform = Transformation() if transform is None else transform

        # Set up an actor pool to apply the transform if needed
        if self.transform.is_null() or (num_workers is not None and num_workers < 1):
            self.num_workers, self.actor_pool = 0, None
        else:
            ncpu = os.cpu_count() or 1
            if num_workers is None:
                num_workers = max(1, math.ceil(ncpu / self._n_parts) - 1)
            self.num_workers = num_workers

            global _ray_refs
            _ray_refs += 1
            if not ray.is_initialized():
                ray.init(num_gpus=0, include_dashboard=False,
                         ignore_reinit_error=True)
            actors = [TransformActor.remote(transform, train)
                      for _ in range(num_workers)]
            self.actor_pool = ray.util.ActorPool(actors)

        # For pre-fetching and caching hdf5/ark files in memory.
        self.pre_fetch_next_epoch = pre_fetch_next_epoch
        self.queue = FileQueue(max_size=data_cache_mb * (2 ** 20),
                               read_file=self.get_file_dict,
                               get_file_size=self.file_dict_size)
        self.files_loaded = None
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.seed = seed

        # Make sure that the data loader is shut down in case of premature exits
        atexit.register(_weakref_close, weakref.ref(self))
        atexit.register(ray.shutdown)

    @abstractmethod
    def get_file_dict(self, path: str, uttid_locs: List[Tuple[str, str]] = None) \
            -> Dict[str, Dict[str, Any]]:
        """Gets a dict of items associated with the file at the given path.
        file_dict[uttid] contains all the data associated with uttid."""
        raise NotImplementedError

    def __del__(self):
        """Shut down the thread pool."""
        if self.actor_pool is not None:
            global _ray_refs
            _ray_refs -= 1
            if _ray_refs == 0:
                ray.shutdown()
        self.thread_pool.shutdown(wait=False)

    def __len__(self):
        if self.ark_or_scp == "ark":
            raise RuntimeError(
                "Cannot get length of Reader reading directly from ark/h5")
        return len(self._bszs)

    @property
    def num_utts(self):
        if self.ark_or_scp == "ark":
            raise RuntimeError(
                "Cannot get number of utterances in Reader reading directly from ark/h5")
        return sum(self._bszs)

    def close(self):
        """Empties the queue and suspends loading any files not yet loaded."""
        self.queue.clear()  # this will stop any pending queue.put()
        files_future = self.files_loaded
        if files_future is not None and not files_future.cancelled():
            self.files_loaded.cancel()
            while not (files_future.done() or files_future.cancelled()):
                pass
        assert self.queue.empty()
        self.queue.open()
        self._fetch_seed = self._seed

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Sets the random seed & cancels the current file-loading job if we are
        currently (or have already) pre-fetched data for the wrong seed."""
        self._seed = seed
        if not hasattr(self, "_fetch_seed") or (seed != self._fetch_seed and self.shuffle):
            self.close()
            self._fetch_seed = seed
            scp_dict, bszs = self.get_scp_dict_and_bszs()
            self._bszs = bszs
            if self.pre_fetch_next_epoch:
                self.load_files(scp_dict)
        else:
            self._fetch_seed = seed

    @property
    def shuffle(self):
        return self._shuffle

    def get_scp_dict_and_bszs(self):
        """Shuffles the SCP dict and gets the relevant split to load from."""
        batch_size = 1 if self._batch_size is None else self._batch_size
        scp_dict, bszs = split_scp_dict(
            self._full_scp_dict, n_parts=self._n_parts,
            randomize=self.shuffle, seed=self._fetch_seed,
            batch_size=batch_size, max_len=self._max_len, utt2len=self._utt2len,
            ensure_equal_parts=self._ensure_equal_parts)[self._i_part]
        if self._bszs is None:
            self._bszs = bszs
        return scp_dict, bszs

    def load_files(self, scp_dict=None):
        """Schedules a background thread to read the relevant file dicts from
        either the SCP dict given, or the one from self.get_scp_dict_and_bszs().
        The thread finishes as soon as it fails to load a file (which happens
        if queue.clear() is called) or once it's done loading all files."""
        def wrapper(d):
            for path, uttid_locs in d.items():
                if not self.queue.put(path, uttid_locs):
                    break

        if scp_dict is None:
            scp_dict, _ = self.get_scp_dict_and_bszs()
        self.files_loaded = self.thread_pool.submit(wrapper, scp_dict)

    @staticmethod
    def file_dict_size(file_dict):
        """file_dict[utt_id] is a dict containing a numpy array (under key
        "x") and other metadata. We primarily care about the size of the
        numpy arrays in the file_dict, so this is the size we return."""
        return sum(map(lambda utt: utt["x"].nbytes, file_dict.values()))

    def datadict_to_output(self, datadict):
        """Converts the dict retrieved from the archive into the desired form."""
        if self.return_dict:
            return datadict
        if self.return_shape:
            return datadict["x"].shape
        return datadict["x"]

    def output_iterator(self, file_dict):
        """Output iterator which applies self.transform to all the signals in
        file_dict, and returns them (in desired format) along w/ their keys."""
        if self.actor_pool is not None:
            it = self.actor_pool.map(
                lambda a, v: a.apply.remote(v), file_dict.items())
        else:
            it = map(
                lambda kd: apply_transform(self.transform, kd, self.train),
                file_dict.items())
        return zip(file_dict.keys(), map(self.datadict_to_output, it))

    def __iter__(self):
        """Loads the data one .h5 archive at a time. Uses a queue to pre-fetch
        the contents of each file into memory (while not exceeding our maximum
        allowed cache size). Yields data points one at a time."""
        if self.ark_or_scp == "scp":
            # Fetch data for this epoch in the background (if not pre-fetched)
            scp_dict, bszs = self.get_scp_dict_and_bszs()
            self._bszs = bszs
            if (self.files_loaded is None or self.files_loaded.done() or
                    self.files_loaded.cancelled()) and self.queue.empty():
                self.load_files(scp_dict)

            i_batch, batch = 0, []
            for j_file, expected_path in enumerate(scp_dict):
                n_from_file = 0
                path, file_dict = self.queue.get(expected_path)

                # Pre-fetch the next epoch as soon as we're done with this one
                if self.pre_fetch_next_epoch and self._fetch_seed == self._seed \
                        and self.files_loaded.done():
                    self._fetch_seed += 1
                    self.load_files()

                output_iterator = self.output_iterator(file_dict)
                if self._batch_size is None:
                    yield from output_iterator
                else:
                    try:
                        for i_batch in range(i_batch, len(bszs)):
                            while len(batch) < bszs[i_batch]:
                                batch.append(next(output_iterator))
                                n_from_file = n_from_file + 1
                            if i_batch == len(bszs) - 1 and j_file == len(scp_dict) - 1:
                                raise StopIteration
                            else:
                                assert len(batch) == bszs[i_batch], \
                                    f"Expected batch {i_batch} to have size {bszs[i_batch]}, " \
                                    f"but got a batch of size {len(batch)} instead"
                                yield batch
                                batch = []

                    except StopIteration:
                        assert n_from_file == len(file_dict), \
                            f"Expected to get {len(file_dict)} utts from " \
                            f"{os.path.basename(path)}, but got {n_from_file}."
                        logger.debug(f"FINISHED FILE {j_file+1}/{len(scp_dict)}: "
                                     f"{os.path.basename(path)}")
                        if i_batch == len(bszs) - 1 and j_file == len(scp_dict) - 1:
                            assert len(batch) == bszs[i_batch], \
                                f"Expected batch {i_batch} to have size {bszs[i_batch]}, " \
                                f"but got a batch of size {len(batch)} instead"
                            yield batch
                        continue

        else:  # self.ark_or_scp == "ark"
            if self.filepath == "-":  # Required h5py>=2.9 for hdf5
                filepath = io.BytesIO(sys.stdin.buffer.read())
            else:
                filepath = self.filepath
            file_dict = self.get_file_dict(filepath)
            output_iterator = self.output_iterator(file_dict)

            if self._batch_size is None:
                yield from output_iterator
            else:
                done = False
                while not done:
                    batch = []
                    try:
                        while len(batch) < self._batch_size:
                            batch.append(next(output_iterator))
                        yield batch
                    except StopIteration:
                        done = True
                        yield batch


class KaldiReader(BaseReader):
    def get_file_dict(self, path: str, uttid_locs: List[Tuple[str, str]] = None) -> \
            Dict[str, Dict[str, Any]]:
        def mat2dict(mat):
            return {"rate": mat[(0,) * mat.ndim] or None, "x": mat[1:]}

        file_dict = OrderedDict(
            (uttid, mat2dict(mat)) for uttid, mat in kaldiio.load_ark(path))
        if uttid_locs is None:
            return file_dict
        return OrderedDict((k, file_dict[k]) for k, loc in uttid_locs)


class HDF5Reader(BaseReader):
    def get_file_dict(self, path: str, uttid_locs: List[Tuple[str, str]] = None) \
            -> Dict[str, Dict[str, Any]]:
        def data2dict(data):
            signal, attrs = data[()], data.attrs
            return {"x": signal, "speaker": attrs.get("speaker"),
                    "text": attrs.get("text"), "rate": attrs.get("rate")}

        with h5py.File(path, "r") as file:
            file_dict = OrderedDict((uttid, data2dict(data))
                                    for uttid, data in file.items())
        if uttid_locs is None:
            return file_dict
        return OrderedDict((k, file_dict[loc]) for k, loc in uttid_locs)


class SoundReader(BaseReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ark_or_scp != "scp":
            raise ValueError(
                f'Only supporting "scp" for sound file: {self.ark_or_scp}')

    def get_file_dict(self, path: str, uttid_locs: List[Tuple[str, str]] = None) \
            -> Dict[str, Dict[str, Any]]:
        signal, rate = soundfile.read(path, dtype="int16")
        return {path: {"x": signal, "rate": rate}}
