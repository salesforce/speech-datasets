import numpy as np
from kaldi import feat, matrix

from speech_datasets.transform.interface import TransformInterface


class KaldiFeatures(TransformInterface):
    def __init__(self, sample_frequency, frame_length_ms=25,
                 frame_shift_ms=10, window_type="povey"):
        self.sample_frequency = sample_frequency
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.window_type = window_type
        self.frame_opts = feat.window.FrameExtractionOptions()
        self.frame_opts.samp_freq = sample_frequency
        self.frame_opts.frame_length_ms = frame_length_ms
        self.frame_opts.frame_shift_ms = frame_shift_ms
        self.frame_opts.window_type = window_type
        self.default_kwargs = {}

    def __setstate__(self, state):
        self.__init__(**state)

    def __call__(self, x, rate=None, **kwargs):
        if rate is not None and rate != self.sample_frequency:
            raise RuntimeError(
                f"Sampling rate {rate}Hz of input WAV must be the same as the "
                f"sampling rate {self.sample_frequency}Hz expected by feature "
                f"computer.")
        try:
            if len(kwargs) > 0:
                full_kwargs = self.default_kwargs.copy()
                full_kwargs.update(kwargs)
            else:
                full_kwargs = self.default_kwargs
            if hasattr(self.computer, "compute"):
                output = self.computer.compute(matrix.Vector(x), **full_kwargs)
            else:
                output = self.computer(matrix.Vector(x), **full_kwargs)
            if not isinstance(output, np.ndarray):
                output = output.numpy()

        except AttributeError as e:
            if "computer" in str(e):
                raise NotImplementedError(
                    "All subclasses of KaldiFeatures must initialize self.computer "
                    "as either a kaldi.feat.* feature computer, or a method returning "
                    "a Kaldi feature matrix.")
            else:
                raise e

        except TypeError as e:
            if "Required argument" in str(e):
                raise NotImplementedError(
                    "Either override self.default_kwargs to provide default values "
                    "for all additional args required by self.computer.compute, or "
                    "provide them explicitly to __call__.")
            else:
                raise e

        if output.shape[0] == 0:
            raise RuntimeError(f"For input wav with shape {x.shape} at sampling "
                               f"frequency {self.sample_frequency}, output had length 0.")
        return output

    @property
    def dim(self):
        if hasattr(self.computer, "dim"):
            return self.computer.dim()
        raise NotImplementedError


class Spectrogram(KaldiFeatures):
    def __init__(self, sample_frequency, frame_length_ms=25,
                 frame_shift_ms=10, window_type="povey"):
        super().__init__(sample_frequency, frame_length_ms, frame_shift_ms, window_type)

        self.opts = feat.spectrogram.SpectrogramOptions()
        self.opts.frame_opts = self.frame_opts
        self.computer = feat.spectrogram.Spectrogram(self.opts)

        self.default_kwargs["vtln_warp"] = 1.0

    def __getstate__(self):
        dict(sample_frequency=self.sample_frequency,
             frame_length_ms=self.frame_length_ms,
             frame_shift_ms=self.frame_shift_ms,
             window_type=self.window_type)

    def __repr__(self):
        return "{name}(samp_freq={f}, window_type={w})".format(
            name=self.__class__.__name__, f=self.sample_frequency, w=self.window_type)


class LogMelSpectrogram(KaldiFeatures):
    def __init__(self, sample_frequency, num_mel_bins, frame_length_ms=25,
                 frame_shift_ms=10, window_type="povey"):
        super().__init__(sample_frequency, frame_length_ms, frame_shift_ms, window_type)
        self.num_mel_bins = num_mel_bins
        self.mel_opts = feat.mel.MelBanksOptions()
        self.mel_opts.num_bins = num_mel_bins
        self.opts = feat.fbank.FbankOptions()
        self.opts.frame_opts = self.frame_opts
        self.opts.mel_opts = self.mel_opts
        self.computer = feat.fbank.Fbank(self.opts)

        self.default_kwargs["vtln_warp"] = 1.0

    def __getstate__(self):
        return dict(sample_frequency=self.sample_frequency,
                    num_mel_bins=self.num_mel_bins,
                    frame_length_ms=self.frame_length_ms,
                    frame_shift_ms=self.frame_shift_ms,
                    window_type=self.window_type)

    def __repr__(self):
        return "{name}(samp_freq={f}, n_mels={n}, window_type={w})".format(
            name=self.__class__.__name__, f=self.sample_frequency, n=self.num_mel_bins, w=self.window_type)


class PitchFeatures(KaldiFeatures):
    def __init__(self, sample_frequency, frame_length_ms=25, frame_shift_ms=10,
                 add_delta_pitch=True, add_normalized_log_pitch=True,
                 add_pov_feature=True, add_raw_log_pitch=False):
        super().__init__(sample_frequency, frame_length_ms, frame_shift_ms)
        self.pitch_opts = feat.pitch.PitchExtractionOptions()
        self.pitch_opts.samp_freq = sample_frequency
        self.pitch_opts.frame_length_ms = frame_length_ms
        self.pitch_opts.frame_shift_ms = frame_shift_ms

        self.process_opts = feat.pitch.ProcessPitchOptions()
        self.process_opts.add_delta_pitch = add_delta_pitch
        self.process_opts.add_normalized_log_pitch = add_normalized_log_pitch
        self.process_opts.add_pov_feature = add_pov_feature
        self.process_opts.add_raw_log_pitch = add_raw_log_pitch
        assert self.dim > 0

    def __repr__(self):
        return (
            "{name}(samp_freq={f}, add_delta_pitch={dp}, "
            "add_normalized_log_pitch={nlp}, add_pov_feature={pov},"
            "add_raw_log_pitch={rlp})".format(
                name=self.__class__.__name__,
                f=self.pitch_opts.samp_freq,
                dp=self.process_opts.add_delta_pitch,
                nlp=self.process_opts.add_normalized_log_pitch,
                pov=self.process_opts.add_pov_feature,
                rlp=self.process_opts.add_raw_log_pitch)
        )

    def __getstate__(self):
        return dict(sample_frequency=self.sample_frequency,
                    frame_length_ms=self.frame_length_ms,
                    frame_shift_ms=self.frame_shift_ms,
                    add_delta_pitch=self.process_opts.add_delta_pitch,
                    add_normalized_log_pitch=self.process_opts.add_normalized_log_pitch,
                    add_pov_feature=self.process_opts.add_pov_feature,
                    add_raw_log_pitch=self.process_opts.add_raw_log_pitch)

    def computer(self, x):
        return feat.pitch.compute_and_process_kaldi_pitch(
            self.pitch_opts, self.process_opts, matrix.Vector(x))

    @property
    def dim(self):
        return sum([self.process_opts.add_delta_pitch,
                    self.process_opts.add_normalized_log_pitch,
                    self.process_opts.add_pov_feature,
                    self.process_opts.add_raw_log_pitch])


class FbankPitch(KaldiFeatures):
    def __init__(self, sample_frequency, num_mel_bins,
                 frame_length_ms=25, frame_shift_ms=10, window_type="povey",
                 add_delta_pitch=True, add_normalized_log_pitch=True,
                 add_pov_feature=True, add_raw_log_pitch=False):
        super().__init__(sample_frequency, frame_length_ms, frame_shift_ms, window_type)

        self.length_tol = 2
        self.fbank = LogMelSpectrogram(sample_frequency, num_mel_bins,
                                       frame_length_ms, frame_shift_ms, window_type)
        self.pitch = PitchFeatures(sample_frequency, frame_length_ms, frame_shift_ms,
                                   add_delta_pitch, add_normalized_log_pitch,
                                   add_pov_feature, add_raw_log_pitch)

    def __getstate__(self):
        return dict(sample_frequency=self.sample_frequency,
                    num_mel_bins=self.fbank.num_mel_bins,
                    frame_length_ms=self.frame_length_ms,
                    frame_shift_ms=self.frame_shift_ms,
                    window_type=self.fbank.window_type,
                    add_delta_pitch=self.pitch.process_opts.add_delta_pitch,
                    add_normalized_log_pitch=self.pitch.process_opts.add_normalized_log_pitch,
                    add_pov_feature=self.pitch.process_opts.add_pov_feature,
                    add_raw_log_pitch=self.pitch.process_opts.add_raw_log_pitch)

    def computer(self, x, **kwargs):
        fbank, pitch = self.fbank(x, **kwargs), self.pitch(x)
        min_len = min(fbank.shape[0], pitch.shape[0])
        max_len = max(fbank.shape[0], pitch.shape[0])

        if max_len - min_len < self.length_tol:
            return np.concatenate((fbank[:min_len], pitch[:min_len]), axis=1)
        else:
            raise RuntimeError(f"fbank features have {fbank.shape[0]} frames, but "
                               f"pitch features have {pitch.shape[0]} frames. This "
                               f"differs by more than the tolerance of {self.length_tol}")

    @property
    def dim(self):
        return self.fbank.dim + self.pitch.dim
