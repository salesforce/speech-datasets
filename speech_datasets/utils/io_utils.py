from collections import defaultdict, deque, OrderedDict
from distutils.util import strtobool as dist_strtobool
import logging
import os
import sys
from threading import Condition, RLock
from typing import Dict, List, Tuple

import numpy as np

from speech_datasets.utils import get_root

logger = logging.getLogger(__name__)


def strtobool(x):
    # distutils.util.strtobool returns integer, but it's confusing,
    return bool(dist_strtobool(x))


def get_commandline_args():
    extra_chars = [" ", ";", "&", "(", ")", "|", "^", "<", ">", "?", "*", "[",
                   "]", "$",  "`", '"', "\\", "!", "{", "}"]

    # Escape the extra characters for shell
    argv = [
        arg.replace("'", "'\\''")
        if all(char not in arg for char in extra_chars)
        else "'" + arg.replace("'", "'\\''") + "'"
        for arg in sys.argv
    ]

    return sys.executable + " " + " ".join(argv)


class FileQueue(object):
    """Thread-safe queue which reads files into memory using `read_file` and
    doesn't read any more files when the total size of the cached files
    exceeds `max_size`. The size of each file (loaded by `read_file`) is given
    by `get_file_size`. Note that get() and put() are blocking operations,
    and they are fulfilled on a first-come first-served basis."""
    def __init__(self, max_size=0, read_file=None, get_file_size=None):
        self._max_size = max_size
        self.read_file = read_file if read_file else lambda x: x
        self.get_file_size = get_file_size if get_file_size else lambda x: 1

        # _cur_size and _queue need atomic access
        self._cur_size = 0
        self.n = 0
        self._closed = False
        self._queue = deque()
        self._mutex = RLock()

        # first process to try to add to/remove from the queue gets to do so
        self._not_full = Condition()
        self._not_empty = Condition()

        # track whether the respective condition variable is waiting on its
        # condition. this determines whether an acquisition that should notify
        # this condition variable, should block or not.
        self.__not_full_waiting = False
        self.__not_empty_waiting = False

    def full(self, *, debug_msg=False):
        with self._mutex:
            is_full = self._cur_size >= self._max_size > 0
            if is_full and debug_msg:
                logger.debug("QUEUE FULL")
            return is_full

    def empty(self):
        with self._mutex:
            is_empty = len(self._queue) == 0
            return is_empty

    def close(self):
        """Cancels all pending put operations."""
        with self._not_full:
            with self._not_empty:
                self._closed = True
                self._not_full.notify_all()
                self._not_empty.notify_all()

    def open(self):
        with self._not_full:
            self._closed = False

    def clear(self):
        """Close the queue and then clear it."""
        with self._not_full:
            with self._not_empty:
                with self._mutex:
                    self._queue.clear()
                    self._cur_size = 0
                    self.close()

    @property
    def _not_full_waiting(self):
        with self._mutex:
            return self.__not_full_waiting

    @_not_full_waiting.setter
    def _not_full_waiting(self, x):
        with self._mutex:
            self.__not_full_waiting = x

    @property
    def _not_empty_waiting(self):
        with self._mutex:
            return self.__not_empty_waiting

    @_not_empty_waiting.setter
    def _not_empty_waiting(self, x):
        with self._mutex:
            self.__not_empty_waiting = x

    def put(self, path, *args, **kwargs):
        """Returns whether we succeeded in adding the item to the queue."""
        with self._not_full:
            logger.debug(f"TRY PUT {os.path.basename(path)}")
            self._not_full_waiting = True
            self._not_full.wait_for(lambda: self._closed or not self.full(debug_msg=True))
            self._not_full_waiting = False

            if self._closed:
                return False
            file = self.read_file(path, *args, **kwargs)

            with self._mutex:
                self._queue.append((path, file))
                self._cur_size += self.get_file_size(file)

            if self._not_empty.acquire(blocking=self._not_empty_waiting):
                self._not_empty.notify()
                self._not_empty.release()

        logger.debug(f"PUT {os.path.basename(path)}")
        return True

    def get(self, expected_path=None):
        with self._not_empty:
            if expected_path is not None:
                logger.debug(f"TRY GET {os.path.basename(expected_path)}")
            self._not_empty_waiting = True
            self._not_empty.wait_for(lambda: self._closed or not self.empty())
            self._not_empty_waiting = False

            with self._mutex:
                if self.empty():
                    path, file = None, None
                else:
                    path, file = self._queue.popleft()
                    self._cur_size -= self.get_file_size(file)

            if self._not_full.acquire(blocking=self._not_full_waiting):
                self._not_full.notify()
                self._not_full.release()

        if path is not None:
            if expected_path is not None and path != expected_path:
                raise RuntimeError(
                    f"A race condition caused us to fetch the wrong file. "
                    f"Expected {expected_path}, but got {path} instead.")
            logger.debug(f"GET {os.path.basename(path)}")
        else:
            logger.debug(f"QUEUE EMPTY")
        return path, file


def sub_file_name(original_filename: str, file_idx: int,
                  double_ext: bool = False) -> str:
    base, ext = os.path.splitext(original_filename)
    if double_ext:
        base2, ext2 = os.path.splitext(base)
        if len(ext2) > 0 and not all(c.isdigit() for c in ext2[1:]):
            ext = ext2 + ext
            base = base2
    return base + "." + str(file_idx) + ext


def parse_scp_line(line: str) -> Tuple[str, str, str]:
    key, value = line.rstrip().split(None, maxsplit=1)
    if ":" not in value:
        path, loc = value, ""
    else:
        path, loc = value.split(":", maxsplit=1)
        if loc == "":
            raise ValueError(
                "scp file's lines should be either 'key value' or 'key value:loc',"
                "but got 'key value:' instead.")
    return key, path, loc


def consolidate_utt_info(scp: str = None, text: str = None, utt2spk: str = None,
                         utt2num_frames: str = None,
                         permissive: bool = True) -> Dict[str, Dict[str, str]]:
    """Consolidates all the info from a scp, text, and utt2spk file into
     a single dictionary."""
    def read(fname):
        return open(fname, "r", encoding="utf-8")

    # Read in all the information
    expected = set()
    utts = defaultdict(dict)
    if scp is not None:
        expected.update({"path", "loc"})
        with read(scp) as fscp:
            for line in fscp:
                k, path, loc = parse_scp_line(line)
                utts[k]["path"] = path
                utts[k]["loc"] = loc

    # Make sure to account for utterances with no transcriptions
    if text is not None:
        expected.add("text")
        with read(text) as f:
            for line in f:
                parsed = line.rstrip().split(None, maxsplit=1)
                utts[parsed[0]]["text"] = "" if len(parsed) == 1 else parsed[1]

    if utt2spk is not None:
        with read(utt2spk) as f:
            expected.add("speaker")
            for line in f:
                k, speaker = line.rstrip().split(None, maxsplit=1)
                utts[k]["speaker"] = speaker

    if utt2num_frames is not None:
        with read(utt2num_frames) as f:
            expected.add("length")
            for line in f:
                k, length = line.rstrip().split(None, maxsplit=1)
                utts[k]["length"] = int(length)

    # Validate that all the utterances have all the relevant info
    non_none_files = ", ".join(x for x in [scp, text, utt2spk] if x)
    filtered_utts = {k: u for k, u in utts.items() if u.keys() == expected}
    if len(filtered_utts) != len(utts):
        if permissive:
            logger.warning(
                f"Some utterances do not have entries in all of the files "
                f"{non_none_files}'. Only keeping utterances with entries"
                f"entries in all files.")
        else:
            raise RuntimeError(
                f"Expected all utterances to have entries in the files "
                f"{non_none_files}, but some are missing from some files.")

    return filtered_utts


def parse_scp_file(scp_fname) -> Dict[str, List[Tuple[str, str]]]:
    """Reads an input SCP file (`scp_fname`) and parses it. Returns `scp_dict`,
    which is an OrderedDict[str, OrderedDict[str, str]] that looks like
    ```
    {
        archive_0: [(uttid_0_0, loc_0_0), ..., (uttid_0_j, loc_0_j), ...],
        ...,
        archive_i: [(uttid_i_0, loc_i_0), ..., (uttid_i_j, loc_i_j), ...],
        ...
    }
    ```
    where archive_i is the i'th archive file referenced in the SCP (in the order
    it appears in the SCP file), uttid_i_j is the ID of the j'th utterance in
    archive_i (again in the order it appeared in the SCP file), and loc_i_j is
    the location of this utterance within archive_i (an integer offset for
    Kaldi ark files, and a string key for hdf5 archives).
    """
    scp_dict = OrderedDict()
    with open(scp_fname, "r", encoding="utf-8") as f:
        for line in f:
            uttid, path, loc = parse_scp_line(line)
            if path not in scp_dict:
                scp_dict[path] = []
            scp_dict[path].append((uttid, loc))

    return scp_dict


def split_scp_dict(scp_dict: Dict[str, List[Tuple[str, str]]],
                   n_parts=1, randomize=False, seed: int = None,
                   batch_size=1, max_len=None, utt2len=None,
                   ensure_equal_parts=True) \
        -> List[Tuple[Dict[str, List[Tuple[str, str]]], List[int]]]:
    """Splits a `scp_dict` returned by `parse_scp_file()` into `n_parts` sub-
    parts. These sub-parts proceed one archive file at a time, but the order of
    these archive files, as well as the order of utterances within them, can
    be randomized if desired. Furthermore a series of batch sizes corresponding
    to each part is returned."""
    assert batch_size >= 1
    archives = list(scp_dict.keys())
    n_archives = len(archives)

    # Determine the orders in which to iterate over archives & the utterances
    # within those archives
    if randomize:
        rs = np.random.RandomState(seed=seed)
        archive_order = rs.permutation(n_archives)
        archive2utt_order = {archive: rs.permutation(len(scp_dict[archive])) for
                             archive in archives}
    else:
        archive_order = np.arange(n_archives)
        archive2utt_order = {archive: np.arange(len(scp_dict[archive])) for
                             archive in archives}

    # Get all utterances in the right order (along with their archive)
    all_utts = [(archives[i], *scp_dict[archives[i]][j])  # (path, uttid, loc)
                for i in archive_order
                for j in archive2utt_order[archives[i]]]

    # Compute the appropriate sequence (possibly dynamic) of batch sizes
    start, batch_sizes = 0, []
    while start < len(all_utts):
        tmp_end = min(start + batch_size, len(all_utts))
        if max_len is None or utt2len is None:
            end = tmp_end
        else:
            candidate_utts = all_utts[start:tmp_end]
            lens = [utt2len[uttid] for archive, uttid, loc in candidate_utts]
            dynamic_batch_size = (np.cumsum(lens) <= max_len * batch_size).sum()
            end = start + max(1, dynamic_batch_size)
        batch_sizes.append(end - start)
        start = end

    # Some batches will get 1 extra batch from the first part
    n_utts, n_batches = len(all_utts), len(batch_sizes)
    n_batches_per_part, extra = divmod(n_batches, n_parts)
    if n_batches < n_parts:
        raise RuntimeError(
            f"Cannot partition a SCP with {n_utts} utterances into "
            f"{n_parts} parts with batch size {batch_size}. This causes "
            f"the number of batches ({n_batches}) to be less than the "
            f"number of parts ({n_parts}).")

    # Determine the sequence of batch sizes for each part
    part_n_batches = [n_batches_per_part + int(i < extra)
                      for i in range(n_parts)]
    part_j0_batch = np.cumsum([0, *part_n_batches])
    part_batch_sizes = [batch_sizes[part_j0_batch[i]:part_j0_batch[i+1]]
                        for i in range(n_parts)]

    # Determine the sequence of utterances for each part
    part_n_utts = [sum(x) for x in part_batch_sizes]
    part_j0_utt = np.cumsum([0, *part_n_utts])
    part_utts = [all_utts[part_j0_utt[i]:part_j0_utt[i+1]]
                 for i in range(n_parts)]

    # Ensure each part has the same number of batches (if desired)
    if ensure_equal_parts and extra > 0:
        extra_batch_sizes = batch_sizes[:n_parts - extra]
        extra_j0_utt = np.cumsum([0, *extra_batch_sizes])
        for i, j in enumerate(range(extra, n_parts)):
            part_batch_sizes[j].append(batch_sizes[i])
            part_utts[j].extend(all_utts[extra_j0_utt[i]:extra_j0_utt[i+1]])

    if ensure_equal_parts:
        assert all(len(part_batch_sizes[i]) == len(part_batch_sizes[0])
                   for i in range(n_parts)), \
            "All parts should have the same number of batches."

    assert all(sum(part_batch_sizes[i]) == len(part_utts[i])
               for i in range(n_parts)),\
        "Sum of batch sizes for each part should match the number of " \
        "utterances in the part."

    # Construct each part as an ordered dict (archive -> List[uttid, loc])
    parts = []
    for i in range(n_parts):
        current_part = OrderedDict()
        for archive, uttid, loc in part_utts[i]:
            if archive not in current_part:
                current_part[archive] = []
            current_part[archive].append((uttid, loc))
        parts.append(current_part)

    return list(zip(parts, part_batch_sizes))


def get_combo_idx(datasets: List[str], task: str) -> int:
    """Check if a particular combination of `datasets` (each formatted as
    f"{dataset}/{split}") has been registered yet for task `task`."""
    # Check if data combination registry has been created yet
    combine_dir = os.path.join(get_root(), "COMBINE", f"{task}1")
    registry = os.path.join(combine_dir, "data", "registry.txt")
    if not os.path.isfile(registry):
        return -1

    # Check if this particular combo is in the data combo registry
    datasets = sorted(set(datasets))
    with open(registry) as f:
        registry = [sorted(set(line.rstrip().split())) for line in f]
    combo_idxs = [i+1 for i, d in enumerate(registry) if d == datasets]

    return -1 if len(combo_idxs) == 0 else combo_idxs[0]
