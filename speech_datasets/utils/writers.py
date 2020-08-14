import logging
from pathlib import Path
from typing import Dict, Union

import h5py
import kaldiio
import numpy as np
import resampy
import soundfile

from speech_datasets.transform import Transformation
from speech_datasets.utils.io_utils import sub_file_name

logger = logging.getLogger(__name__)


def file_writer_helper(
    wspecifier: str,
    filetype: str = "hdf5",
    compress: bool = False,
    compression_method: int = 2,
    pcm_format: str = "wav",
    sample_frequency: float = None,
    transform: Transformation = None,
    max_hr_per_file: Union[float, None] = 5,
):
    """Write matrices in kaldi style

    Args:
        wspecifier: e.g. ark,scp:out.ark,out.scp
        filetype: "mat": kaldi-matrix, "hdf5": HDF5, "sound": raw soundfiles
        compress: Compress or not
        compression_method: Specify compression level
        pcm_format: format of output wave file (if writing sound files directly)
        sample_frequency: sampling rate (in Hz) of audio to be written
        transform: the transformation to be applied before writing the data
        max_hr_per_file: maximum hours of audio to write per individual file

    Write in kaldi-matrix-ark with "kaldi-scp" file:

    >>> with file_writer_helper('ark,scp:out.ark,out.scp') as f:
    >>>     f['uttid'] = array

    This "scp" has the following format:

        uttidA out.ark:1234
        uttidB out.ark:2222

    where, 1234 and 2222 points the starting byte address of the matrix.
    (For detail, see official documentation of Kaldi)

    Write in HDF5 with "scp" file:

    >>> with file_writer_helper('ark,scp:out.h5,out.scp', 'hdf5') as f:
    >>>     f['uttid'] = array

    This "scp" file is created as:

        uttidA out.h5:uttidA
        uttidB out.h5:uttidB

    HDF5 can be, unlike "kaldi-ark", accessed to any keys,
    so originally "scp" is not required for random-reading.
    Nevertheless we create "scp" for HDF5 because it is useful
    for some use-case. e.g. Concatenation, Splitting.

    """
    if filetype == "mat":
        return KaldiWriter(wspecifier, compress=compress, sample_frequency=sample_frequency,
                           compression_method=compression_method, transform=transform,
                           max_hr_per_file=max_hr_per_file)
    elif filetype == "hdf5":
        return HDF5Writer(wspecifier, compress=compress, sample_frequency=sample_frequency,
                          transform=transform, max_hr_per_file=max_hr_per_file)
    elif filetype == "sound":
        return SoundWriter(wspecifier, pcm_format=pcm_format,
                           sample_frequency=sample_frequency, transform=transform)
    else:
        raise NotImplementedError(f"filetype={filetype}")


def parse_wspecifier(wspecifier: str) -> Dict[str, str]:
    """Parse wspecifier to dict

    Examples:
        >>> parse_wspecifier('ark,scp:out.ark,out.scp')
        {'ark': 'out.ark', 'scp': 'out.scp'}

    """
    ark_scp, filepath = wspecifier.split(":", maxsplit=1)
    if ark_scp not in ["ark", "scp,ark", "ark,scp"]:
        raise ValueError("{} is not allowed: {}".format(ark_scp, wspecifier))
    ark_scps = ark_scp.split(",")
    filepaths = filepath.split(",")
    if len(ark_scps) != len(filepaths):
        raise ValueError("Mismatch: {} and {}".format(ark_scp, filepath))
    spec_dict = dict(zip(ark_scps, filepaths))
    return spec_dict


def write_cmvn_stats(path, cmvn_type, stats_dict):
    stats_dict = {spk: stats.to_numpy() for spk, stats in stats_dict.items()}
    if cmvn_type == "global":
        kaldiio.save_mat(path, stats_dict[None])
    else:
        kaldiio.save_ark(path, stats_dict)


class BaseWriter:
    def __init__(self, wspecifier: str, sample_frequency: Union[float, None],
                 transform: Union[Transformation, None],
                 max_hr_per_file: Union[float, None]):
        self.spec_dict = parse_wspecifier(wspecifier)
        self.ark = self.spec_dict["ark"]
        self.input_hz = sample_frequency
        self.transform = Transformation() if transform is None else transform
        self.output_hz = self.input_hz
        for fn in self.transform.functions:
            if hasattr(fn, "sample_frequency"):
                self.output_hz = fn.sample_frequency
            if hasattr(fn, "frame_shift_ms"):
                self.output_hz = 1000 / fn.frame_shift_ms

        if "scp" in self.spec_dict:
            self.writer_scp = open(self.spec_dict["scp"], "w", encoding="utf-8")
        else:
            self.writer_scp = None

        self.writer = None
        self.file_idx = 1
        self.cur_file_sec = 0
        self.max_file_sec = max_hr_per_file * 3600 if max_hr_per_file else None

    def get_writer(self, fname):
        """Returns an open writer to write to the given filename."""
        raise NotImplementedError

    def __call__(self, key, value_dict):
        """The actual implementation of writing `value` under key `key`."""
        raise NotImplementedError

    @property
    def filename(self):
        """Returns the name of the currently open file."""
        if self.ark != "-" and self.max_file_sec is not None and self.max_file_sec > 0:
            return sub_file_name(self.ark, self.file_idx)
        return self.ark

    def resample_and_transform(self, value_dict):
        """Resamples the signal contained in value_dict to self.input_hz if needed
        (e.g. for raw audio), and then applies the desired transform to it."""
        signal, rate = value_dict["x"], value_dict.get("rate", self.input_hz)
        if self.input_hz and rate != self.input_hz:
            # FIXME: use sox?
            logger.warning(f"Resampling input from {rate}Hz to {self.input_hz}Hz.")
            dtype = signal.dtype
            signal = resampy.resample(
                signal.astype(np.float64), rate, self.input_hz, axis=0)
            signal = signal.astype(dtype)
            rate = self.input_hz

        value_dict["x"] = self.transform(signal, rate=rate)
        value_dict["rate"] = rate if self.output_hz is None else self.output_hz
        return value_dict

    def get_nframe(self, value_dict):
        """Returns the number of frames in the signal in `value_dict`."""
        return len(value_dict["x"])

    def try_open_next_writer(self, n_frame):
        """If the next signal (with `n_frame` frames sampled at rate `rate` in Hz)
        will cause the current file to go over its maximum allowed length, close
        the current writer and open the next one."""
        sample_sec = n_frame / self.output_hz if self.output_hz else 0
        new_file_sec = self.cur_file_sec + sample_sec
        if self.writer is None:
            assert self.cur_file_sec == 0
            self.writer = self.get_writer(self.filename)

        if self.max_file_sec and new_file_sec > max(sample_sec, self.max_file_sec):
            self.writer.close()
            self.file_idx += 1
            self.cur_file_sec = 0
            self.writer = self.get_writer(self.filename)

        self.cur_file_sec += sample_sec

    def __setitem__(self, key, value_dict: dict):
        """Writes the value for the given key. Automatically writes to a new file if
        the current file will become too full with the addition of `value`."""
        value_dict = self.resample_and_transform(value_dict)
        n_frame = self.get_nframe(value_dict)
        self.try_open_next_writer(n_frame)
        self(key, value_dict)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as e:
                logger.warning(f"Exception {str(e)} when closing self.writer")

        if self.writer_scp is not None:
            try:
                self.writer_scp.close()
            except Exception as e:
                logger.warning(f"Exception {str(e)} when closing self.writer")


class KaldiWriter(BaseWriter):
    """KaldiWriter

    Examples:
        >>> with KaldiWriter('ark:out.mat', compress=True, compression_method=2) as f:
        ...     f['key'] = array
    """
    def __init__(self, wspecifier, compress=False, compression_method=2,
                 sample_frequency=100., transform=None, max_hr_per_file=5.):
        super().__init__(wspecifier, sample_frequency, transform, max_hr_per_file)
        self.compression_method = compression_method if compress else None

    def get_writer(self, fname):
        writer = kaldiio.utils.open_like_kaldi(fname, "wb")
        return writer

    def __call__(self, key, value_dict):
        # Concatenate the rate to the signal to save it
        rate = value_dict.get("rate", 0)
        array = value_dict["x"]
        array = np.concatenate((rate * np.ones((1, *array.shape[1:])), array), axis=0)
        kaldiio.save_ark(self.writer, {key: array}, scp=self.writer_scp,
                         compression_method=self.compression_method)


class HDF5Writer(BaseWriter):
    """HDF5Writer

    Examples:
        >>> with HDF5Writer('ark:out.h5', compress=True) as f:
        ...     f['key'] = array
    """

    def __init__(self, wspecifier, compress=False, sample_frequency=100.,
                 transform=None, max_hr_per_file=5.):
        super().__init__(wspecifier, sample_frequency, transform, max_hr_per_file)
        self.kwargs = {"compression": "gzip"} if compress else {}
        if self.filename == "-":
            raise ValueError(
                f"HDF5Writer (archive format 'hdf5') received wspecifier {wspecifier}, but "
                f"it cannot write to stdout. Use KaldiWriter (archive format 'mat') instead.")

    def get_writer(self, fname):
        return h5py.File(fname, "w")

    def __call__(self, key, value_dict):
        value_dict["rate"] = value_dict.get("rate", self.input_hz)
        data = self.writer.create_dataset(key, data=value_dict["x"], **self.kwargs)
        for k, v in value_dict.items():
            if k != "x" and v is not None:
                data.attrs[k] = v
        if self.writer_scp is not None:
            self.writer_scp.write(f"{key} {self.filename}:{key}\n")


class SoundWriter(BaseWriter):
    """SoundWriter

    Examples:
        >>> fs = 16000
        >>> with SoundWriter('ark,scp:outdir,out.scp') as f:
        ...     f['key'] = fs, array
    """

    def __init__(self, wspecifier, pcm_format="wav", sample_frequency=16000., transform=None):
        super().__init__(wspecifier, sample_frequency, transform, None)
        self.pcm_format = pcm_format
        # for wspecifier, we expect something like
        # ark,scp:dirname,wav.scp
        # -> The wave files are found in dirname/*.wav
        Path(self.filename).mkdir(parents=True, exist_ok=True)

    def get_writer(self, fname):
        """SoundWriter writes each signal to a new file, so there is no dedicated writer."""
        return None

    def try_open_next_writer(self, n_frame):
        """SoundWriter writes each signal to a new file, so there is no dedicated writer."""
        pass

    def __call__(self, key, value_dict):
        signal, rate = value_dict["x"], value_dict.get("rate", self.input_hz)
        wavfile = Path(self.filename) / (key + "." + self.pcm_format)
        soundfile.write(wavfile, signal.astype(np.int16), rate)

        if self.writer_scp is not None:
            self.writer_scp.write(f"{key} {wavfile}\n")
