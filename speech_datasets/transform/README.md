# Overview
The high-level interface for data transformations is adapted from ESPNet's code and is defined in 
[`speech_datasets/transform/transformation.py`](transformation.py). You can specify a series of data transformations
to be applied in a `yaml` file, as a list of dictionaries. Each list item should contain a `type` keyword specifying the
type of transformation, as well as the keyword arguments needed to fully specify that particular transformation. 
Alternatively, you can provide a Python `List[Dict[str, Any]]` which contains the same information.

A transformation config file can be provided either to determine the process used to pre-compute features (i.e. by
overriding `conf/fbank.sh` or `conf/fbank_pitch.sh`), or to apply the transformation to data online (i.e. by supplying
it as the `transform_conf` argument of `speech_datasets.SpeechDataLoader`). 

We support the following data transformations: 
# Feature Extraction
We support feature computation using [PyKaldi](https://github.com/pykaldi/pykaldi), which simply acts as a Python
wrapper for Kaldi's binaries. These transformations all take raw audio as input, and have some shared parameters:
- `sample_frequency`: (`float`, required) the frequency at which the input audio is sampled
- `frame_length_ms`: (default `25`): the length of each frame in milliseconds
- `frame_shift_ms`: (default `10`): the number of milliseconds between the start of one frame and the next
- `window_type`: (default `"povey"`): the window type, one of `"hamming"`, `"hanning"`, `"povey"`, `"rectangular"`,
or `"blackman"`


### Filterbank features (`type: fbank`)
Computes filterbank features at each frame. Additional parameters:
- `num_mel_bins`: (`int`, required): the number of triangular Mel-frequency bins (aka the number of filterbank features)


### Filterbank + pitch features (`type: fbank_pitch`)
Computes filterbank & pitch features at each frame. Additional parameters:
- `num_mel_bins`: (`int`, required): the number of triangular Mel-frequency bins (aka the number of filterbank features)
- `add_delta_pitch`: (default `True`):  whether to add the time derivative of log-pitch to the output features
- `add_normalized_log_pitch`: (default `True`): whether to add the normalized log-pitch to the output features
- `add_pov_feature`: (default `True`): whether to add the warped NCCF to the output features
- `add_raw_log_pitch`: (default `False`): whether to add the (unnormalized) log-pitch to the output features


### Spectrogram (`type: spectrogram`)
Computes a spectrogram at each frame. No additional parameters.


# Feature Manipulation
### CMVN (`type: cmvn`)
Apply cepstral mean & variance normalization to utterances. TThis can be either global, per-speaker, or per-uttterance.
This transformation requires the following parameters:
- `cmvn_type`: (required): `global`, `speaker`, or `utterance`
- `stats`: (default `None`): a pre-computed file of CMVN statistics. Used for `global` and `speaker` CMVN.
- `norm_means`: (default `True`): whether to zero-center the features
- `norm_vars`: (default `False`): whether to re-scale the features to have unit variance
- `utt2spk`: (default `None`): an `utt2spk` file mapping utterances to speakers. Used for `speaker` CMVN.
- `reverse`: (default `False`): whether to reverse the application of CMVN
- `std_floor`: (default `1e-20`): the smallest standard deviation value allowed (for numerical precision)


### Delta Features (`type: delta`)
Computes all time derivative(s) of the input feature, up to the specified order. The transformation takes the following
parameters:
- `window`: (default `2`): number of frames in each window used to approximate time derivative computation
- `order`: (default `2`): highest-order time derivative of the input features to compute


# Data augmentation
### Spectral augmentation (`type: spec_augment`)
Applies random time warping, time masking, and frequency masking to features. The default setting is based on LD
(LibriSpeech Double) in Table 2 of [https://arxiv.org/pdf/1904.08779.pdf](https://arxiv.org/pdf/1904.08779.pdf).
This transformation is initialized with the following parameters:
- `resize_mode`: (default `"PIL"`): "PIL" (fast, nondifferentiable) or "sparse_image_warp" (slow, differentiable)
- `max_time_warp`: (default `80`) maximum frames (from the center frame of the spectrogram) to warp (W)
- `max_freq_width`: (default `27`) maximum width of the random frequency mask (F)
- `n_freq_mask`: (default `2`) the number of the random freq mask (m_F)
- `max_time_width`: (default `100`): maximum width of the random time mask (T)
- `n_time_mask`: (default `2`): the number of the random time mask (m_T)
- `inplace`: (default: `True`): overwrite intermediate array
- `replace_with_zero`: (default `True`): replace masked portions with zero if `True`, mean values otherwise


### Bandpass perturbation (`type: bandpass_perturbation`)
Randomly drops data along the frequency axis of features. The original idea is from the following:
"randomly-selected frequency band was cut off under the constraint of
leaving at least 1,000 Hz band within the range of less than 4,000Hz."
(The Hitachi/JHU CHiME-5 system: Advances in speech recognition for
everyday home environments using multiple microphone arrays;
[http://spandh.dcs.shef.ac.uk/chime_workshop/chime2018/papers/CHiME_2018_paper_kanda.pdf](
http://spandh.dcs.shef.ac.uk/chime_workshop/chime2018/papers/CHiME_2018_paper_kanda.pdf))
This transformation is initialized with the following parameters:
- `lower`: (default `0.0`): a lower bound on the fraction of inputs we'd like to apply dropout to (the actual ratio is
  sampled uniformly at random between `lower` and `upper`)
- `upper`: (default `0.75`): an upper bound on the fraction of inputs we'd like to apply dropout to (the actual ratio is
  sampled uniformly at random between `lower` and `upper`)
- `seed`: (default `None`): random seed to initialize this module with
- `axes`: (default `(-1,)`): the axes of the input arrays that we would like to drop out as units


### Speed perturbation (`type: speed_perturbation`)
Applies speed perturbation to raw audio. This transformation is initialized with the following parameters:
- `lower`: (default `0.9`): lower bound on new speed (as a ratio of the old speed)
- `upper`: (default `1.1`): upper bound on new speed (as a ratio of the old speed)
- `keep_length`: (default `False`): whether to truncate or pad to ensure the utterance length remains the same
- `seed`: (default `None`): random seed to initialize this module with
- `res_type`: (default `"kaiser_fast"`): resampling algorithm to use for speed perturbation. `"kaiser_fast"` is faster
but lower quality, while `"kaiser_best"` is high quality but very slow.


### Volume perturbation (`type: volume_perturbation`)
Applies volume perturbation to raw audio. This transformation is initialized with the following parameters:
- `lower`: (default `-1.6`): lower bound on the new volume as a ratio (in sound pressure level); if `dbunit` is `True`
   (default), the ratio (in terms of magnitude) is `10 ** (ratio / 20)`; otherwise, it is just `ratio`
- `upper`: (default `1.6`): upper bound on the new volume as a ratio (in sound pressure level); if `dbunit` is `True`
   (default), the ratio (in terms of magnitude) is `10 ** (ratio / 20)`; otherwise, it is just `ratio`
- `dbunit`: (default `True`) whether `lower` and `upper` are in terms of decibels, or just magnitude
- `seed`: (default `None`): random seed to initialize this module with


### Noise injection (`type: noise_injection`)
Injects noise into raw audio samples. This noise can either be from raw audio saved to disk, or isotropic noise if no
noise samples are provided. This transformation is initialized using the following parameters:
- `lower`: (default `-20`): lower bound on noise-to-signal ratio (in sound pressure level); if `dbunit` is `True`
   (default), the ratio (in terms of magnitude) is `10 ** (ratio / 20)`; otherwise, it is just `ratio`
- `upper`: (default `-5`): upper bound on noise-to-signal ratio (in sound pressure level); if `dbunit` is `True`
   (default), the ratio (in terms of magnitude) is `10 ** (ratio / 20)`; otherwise, it is just `ratio`
- `utt2noise`: (default `None`): If `filetype` is `"list"`, this should be a file mapping utterance ID's to audio
   files (each line should be `"<utt_id> <noise_file>"`). If `filetype` is `"hdf5"`, this should be a HDF5 file
   where all the raw audio has been dumped (e.g. using `asr.sh`). If `utt2noise` is None, just sample isotropic noise.
- `filetype`: (default `"list"`): `"list"` or `"hdf5"`
- `dbunit`: (default `True`) whether `lower` and `upper` are in terms of decibels, or just magnitude
- `seed`: (default `None`): random seed to initialize this module with
