from collections import defaultdict, deque, OrderedDict
from distutils.util import strtobool as dist_strtobool
import logging
import os
import sys
from threading import Condition, Lock
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
        self._lock = Lock()

        # first process to try to add/remove to/from the queue gets to do so
        self._put_lock = Condition(Lock())
        self._get_lock = Condition(Lock())

    def _full(self):
        return self._cur_size >= self._max_size > 0

    def _empty(self):
        return len(self._queue) == 0

    def full(self):
        with self._lock:
            return self._full()

    def empty(self):
        with self._lock:
            return self._empty()

    def close(self):
        """Cancels all pending put operations."""
        with self._put_lock:
            self._closed = True
            self._put_lock.notify_all()

    def open(self):
        with self._put_lock:
            self._closed = False

    def clear(self):
        # Close the queue and then clear it
        self.close()
        with self._put_lock:
            with self._lock:
                self._queue.clear()
                self._cur_size = 0

    def put(self, path, *args, **kwargs):
        """Returns whether we succeeded in adding the item to the queue."""
        with self._put_lock:
            logger.debug(f"TRY PUT {os.path.basename(path)}")
            while self.full() and not self._closed:
                logger.debug("QUEUE FULL")
                self._put_lock.wait()
            if self._closed:
                return False

            file = self.read_file(path, *args, **kwargs)
            with self._lock:
                was_empty = self._empty()
                self._queue.append((path, file))
                self._cur_size += self.get_file_size(file)

        if was_empty:
            with self._get_lock:
                self._get_lock.notify()

        logger.debug(f"PUT {os.path.basename(path)}")
        return True

    def get(self):
        with self._get_lock:
            while self.empty():
                self._get_lock.wait()
            with self._lock:
                was_full = self._full()
                path, file = self._queue.popleft()
                self._cur_size -= self.get_file_size(file)

        if was_full:
            with self._put_lock:
                self._put_lock.notify()

        logger.debug(f"GET {os.path.basename(path)}")
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
                   n_parts: int = None, randomize: bool = False,
                   seed: int = None) -> List[Dict[str, List[Tuple[str, str]]]]:
    """Splits a `scp_dict` returned by `parse_scp_file()` into `n_parts` sub-
    parts. These sub-parts proceed one archive file at a time, but the order of
    these archive files, as well as the order of utterances within them, can
    be randomized if desired."""
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

    # Some parts will get 1 extra utterance from the first archive
    n_utts = sum(len(uttid_locs) for uttid_locs in scp_dict.values())
    n_utts_per_part, extra = divmod(n_utts, n_parts)
    if n_utts < n_parts:
        raise RuntimeError(f"Cannot partitition a SCP with {n_utts} utterances "
                           f"into {n_parts} parts (n_utts < n_parts).")

    parts, j, k = [], 0, 0
    for i in range(n_parts):
        current_part = OrderedDict()
        n, max_n = 0, n_utts_per_part + int(i < extra)
        while n < max_n:
            # Get utterances from the current archive (up til max_n)
            archive = archives[archive_order[j]]
            utt_idxs = archive2utt_order[archive][k:k+(max_n-n)]

            # Add utterances from the current archive in the desired order
            all_archive_utts = scp_dict[archive]
            current_part[archive] = [all_archive_utts[m] for m in utt_idxs]

            # Update the total number of utterances; move on to the next archive
            # if we have exhausted this one
            n += len(utt_idxs)
            if len(utt_idxs) == len(all_archive_utts):
                k = 0
                j += 1

        assert n == max_n

        # If i >= n_utts % n_parts > 0, we need to pull one extra utterance
        # from the start of the dataset (if n % n_parts > 0, but
        # i < n_utts % n_parts, we just took one extra data point up front)
        if i >= extra > 0:
            # Find the archive from which to pull an extra utterance
            j0, skip = 0, 0
            archive = archives[archive_order[j0]]
            while skip + len(archive2utt_order[archive]) < (i - extra):
                j0 += 1
                skip += len(archive2utt_order[archive])
                archive = archives[archive_order[j0]]

            # Obtain the index of the extra utterance within the archive, and
            # add it to this part
            extra_utt_idx = archive2utt_order[archive][i - extra - skip]
            uttid, loc = scp_dict[archive][extra_utt_idx]
            if archive in current_part and uttid in current_part[archive]:
                raise RuntimeError(
                    f"{n_utts} utterances is too few to split int {n_parts} "
                    f"parts. Tried to add the same utterance twice to one part.")
            elif archive not in current_part:
                current_part[archive] = []
            current_part[archive].append((uttid, loc))

        parts.append(current_part)

    return parts


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
