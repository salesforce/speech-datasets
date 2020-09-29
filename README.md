# Table of Contents
1. [Introduction](#introduction)
    1. [Motivation and Related Work](#motivation-and-related-work)
    2. [Key Features](#key-features)
    3. [Approach](#approach)
2. [Getting Started](#getting-started)
    1. [Environment Setup](#environment-setup)
    2. [Data Preparation Recipes](#data-preparation-recipes)
    3. [High-Level Python Interface](#high-level-python-interface)
    4. [Example Training Code](#example-training-code)
    5. [Debugging Notes](#debugging-notes)
3. [Combining Datasets](#combining-datasets)
4. [Adding Support for New Datasets](#adding-support-for-new-datasets)
5. [Specifying Data Transformations](#specifying-data-transformations)
6. [Performance](#performance)
7. [References](#references)

# Introduction
### Motivation and Related Work
The most common approaches to preparing audio data for automatic speech recognition (ASR) and text to speech (TTS)
are solutions that pre-compute perturbations (speed, volume, etc.) and features (MFCCs, spectrograms, etc.) for an
entire dataset, and dump the results to archive files. [Kaldi](https://kaldi-asr.org/) is likely the best-known example
of this strategy.

While this approach is straightforward, due in no small part to [the](https://pypi.org/project/kaldiio/)
[preponderance](https://pypi.org/project/tf_kaldi_io/) [of](https://pypi.org/project/kaldi-io/) 
[libraries](https://pytorch.org/audio/kaldi_io.html) which automate data loading from Kaldi-style archive files,
it suffers from a number of limitations:
1. The wide range of scripts and binaries can be overwhelming for a new user. 
2. There is a large amount of code duplication between recipes for processing different datasets. 
3. Changing the type of features computed requires a user to re-run an entire recipe end-to-end.
4. Random access within a dataset is challenging to perform without either loading the entire dataset into memory,
incurring expensive random disk accesses, or writing a custom caching policy to address this limitation.

More recently, solutions like [ESPNet](https://github.com/espnet/espnet) and [TorchAudio](https://pytorch.org/audio/)
have attempted to address issues 1-3. However, ESPNet does not support data loading that can easily be decoupled from
the end-to-end solution it provides, while TorchAudio currently lacks support for standard (but not open-source)
datasets like Switchboard, WSJ, and Fisher. Moreover, both solutions leave issue 4 unaddressed, and their support for
online feature computation introduces a new concern: online feature computation is computationally expensive, and using it can greatly increase training times.

### Key Features
- After running a Kaldi-style data preparation script, data can be loaded directly into Python. 
- No non-trivial dependencies! Everything can be set up in a single `conda` environment.
- Allows you to work with datasets too large to fit in memory.
- Allows you to combine multiple datasets transparently. 
- Hides the latency of file I/O, online feature extraction, and data augmentation.
- Mimics the functionality of PyTorch's `DistributedSampler` to allow for easy training with `DistributedDataParallel`.
- Straightforward to use in conjunction with other libraries (ESPNet, Huggingface Transformers, your own PyTorch code)
- Minimal code duplication makes it easy for you to add new functionality, as well as support for new datasets. 

### Approach
Our approach to issues 1-3 (large codebase, code duplication, and exclusively offline feature computation) is largely
inspired by [ESPNet2](https://espnet.github.io/espnet/espnet2_tutorial.html). In many cases, we build on their source
code. However, the scope of this project is limited to dataset preparation and loading, rather than attempting to
present an end-to-end solution.

More specifically, we provide a shared top-level script which handles all data preparation steps (sometimes using
utility scripts adapted from Kaldi). Like ESPNet2, this script allows the user to dump either raw audio or pre-computed
features to archive files (whose sizes are carefully controlled). We maintain a single workflow with minimal code
duplication to not only make the code easier to understand for new users, but also minimize the amount of new code
needed to add support for new datasets or new features. See [Data Preparation Recipes](#data-preparation-recipes) and 
[Adding Support for New Datasets](#adding-support-for-new-datasets) for more details.

Once you run the data preparation script and dump the archive files, you can immediately start using
`speech_datasets.SpeechDataLoader` to load datasets. This is as straightforward as
```python
from speech_datasets import SpeechDataLoader as SDL

# Names of training & development datasets (these are sampled at 16kHz)
# We automatically check that the dataset name is valid, and whether it has been 
# prepared properly. Helpful error messages are logged if needed.
tr_data = ["librispeech/train-clean-100", "wsj/train_si284"]
dev_data = ["librispeech/train-other", "wsj/test_dev93"]

# Config to extract 80-dim filterbank features from raw audio sampled at 16kHz
t = [{"type": "fbank", "num_mel_bins": 80, "sample_frequency": 16000}]

# Use data loaders as context managers
with SDL(tr_data, transform_conf=t, train=True, shuffle=True, batch_size=16) as train_loader, \
        SDL(dev_data, transform_conf=t, train=False, shuffle=False, batch_size=16) as dev_loader:

    # Iterator syntax
    for epoch in range(10):
        train_loader.set_epoch(epoch)
        for batch in train_loader:
            # do training...

        for batch in dev_loader:
            # do evaluation...

    # next() syntax
    train_loader.set_epoch(10)
    for i in range(50000):
        batch = train_loader.next()
        # do training...

        # next() increments epoch & step within epoch automatically
        epoch = train_loader.epoch
        step_in_epoch = train_loader.current_position

        # do evaluation every 1000 steps
        if (i + 1) % 1000 == 0:
            for batch in dev_loader:
                # do evaluation...
```
<a name="archive-cache"></a>
Our implementation balances memory constraints with the need for random access within a dataset (issue 4), by
asynchronously loading entire archive files into a cache with a user-specified maximum size (default `4GiB`).
We fetch the next archive file into memory as soon as the cache has free space.

<a name="random-policy"></a>
To approximate a random access pattern, we randomize (1) the order in which the archive files are loaded, and (2) the
order in which utterances within each archive file are loaded. Moreover, for data splits flagged for training 
(e.g. `train_si284` for WSJ; `train-clean-100`, `train-clean-360`, and `train-other-500` for LibriSpeech), the data
preparation script randomizes the way in which utterances are split between different archive files. Empirically, we
see no degradation in accuracy compared to fully randomized sampling 
(see [Performance](#performance) for more details).

Finally, we hide the latency of file I/O and online feature computation & data transformation by having a pool of
independent CPU processes apply the desired transformation on pre-fetched data asynchronously. This happens 
automatically in the background while a model trains on the GPU in the foreground. We thus address all the
issues mentioned above.

# Getting Started
### Environment Setup
To install `speech_datasets` as a Python package, we provide a `Makefile` that sets up a `conda` environment with
both `speech_datasets` and all dependencies.  To set up a Python `<python_ver>` `conda` environment named `<venv_name>`
(either new or pre-existing) with PyTorch version `<torch_ver>`, simply call 
```shell script
make clean all CONDA=<conda_path> VENV_NAME=<venv_name> PYTHON_VERSION=<python_ver> TORCH_VERSION=<torch_ver>
```
By default, `<conda_path>` is the system's installation of `conda`, `<venv_name>` is `datasets`, `<python_ver>` is
`3.7.3`, and `<torch_ver>` is `1.4.0`. Note that we require `torch>=1.2.0`. If you don't have `conda` installed
already, the `Makefile` will install it for you in `tools/venv`; otherwise, `tools/venv` will be a symbolic link to
your system's `conda` directory.

On systems which use the `apt` package manager, the script [`tools/install_pkgs.sh`](tools/install_pkgs.sh) will
automatically install all other binaries you need. These binaries are needed to run the data preparation scripts.
Run this separately from the `Makefile`.

If you wish to use this package in conjunction with ESPNet, run ESPNet's `Makefile` without specifying a Python binary
(i.e. DON'T specify the `PYTHON=<path>` option as done
[here](https://espnet.github.io/espnet/installation.html#create-virtualenv-from-an-existing-python)). Then, run
```shell script
make clean all CONDA=<espnet_root>/tools/venv/bin/conda VENV_NAME=base TORCH_VERSION=<torch_ver>
```
Make sure to accept the prompts which warn you that the environment `base` already exists, and that this environment
may contain a different version of Python from what the `Makefile` requests. Specifying `TORCH_VERSION=''` prevents the
`Makefile` from installing a new version of PyTorch, which may be useful. However, `torch>=1.2.0` is still required,
so we suggest just using the default setting of `1.4.0` (the latest release still compatible with older versions of
ESPNet).


##### Notes
1. The `Makefile` will automatically detect your CUDA version (if any) and install the appropriate distribution of
PyTorch accordingly. If the `conda` environment `<venv_name>` already exists, you will be asked for confirmation
before anything is installed. Once this is done, you can simply activate the `conda` environment and start using
the package.

2. We use `conda` rather than `pip` because
    1. We depend on [PyKaldi](https://github.com/pykaldi/pykaldi), which has a `conda` package but not a `pip` package
    2. `conda` makes it easier to set up an installation of PyTorch compatible with the installed CUDA version (if any)

3. We install the Python package in editable mode (i.e. using `pip install -e .`), rather than installing it to
`site-packages` directly (i.e. using `pip install .`). This is because the data loader assumes that the
`speech_datasets` package is in the same directory as all the actual datasets, i.e.  that the directory structure of
this repo is maintained.

### Data Preparation Recipes
Before you can use `speech_datasets.SpeechDataLoader` to load a dataset, you need to prepare that dataset in the desired
format. Our data preparation recipes are adapted from the various scripts in [Kaldi](https://kaldi-asr.org/) and
[ESPNet2](https://espnet.github.io/espnet/espnet2_tutorial.html). More specifically, for each dataset, we provide a
top-level `<dataset>/asr1/run.sh` or `<dataset>/tts1/run.sh` script which proceeds in 7 stages.  **Note that we avoid
code duplication by having each** `run.sh` **simply provide dataset-specific options to a shared** 
[`TEMPLATE/asr1/asr.sh`](TEMPLATE/asr1/asr.sh) **or** [`TEMPLATE/tts1/tts.sh`](TEMPLATE/tts1/tts.sh) **script!**

You have the option to prepare the dataset as raw audio (`--feats-type raw`, default setting), pre-computed filterbank
features (`--feats-type fbank`), or pre-computed filterbank + pitch features (`--feats-type fbank_pitch`). If you choose
to pre-compute `fbank` or `fbank_pitch` features, you can specify how they should be computed by modifying the
configuration files `<dataset>/<asr1|tts1>/conf/fbank.yaml` and `<dataset>/<asr1|tts1>/conf/fbank_pitch.yaml`,
respectively. See the relevant (linked) section of [`CONTRIBUTING.md`](CONTRIBUTING.md#pre-computing-different-features)
for more details.

Before you can run the data preparation script for any dataset, update the paths in
[`TEMPLATE/asr1/db.sh`](TEMPLATE/asr1/db.sh) or [`TEMPLATE/tts1/db.sh`](TEMPLATE/tts1/db.sh). These should be the
absolute paths (on your system) to the directories and/or archive files containing the downloaded datasets. For some
open source datasets, we also provide download scripts, but in general, you will need to download the data yourself
and update `db.sh` with its actual location.

Once you have done this, navigate to `<dataset>/asr1` or `<dataset>/tts1` and invoke:

```shell script
./run.sh --stage <i> --stop-stage <j> --feats-type <raw|fbank|fbank_pitch> [other options]
```

The `run.sh` script proceeds in 7 stages (starting at `<i>` and ending at `<j>`). These stages are as follows:

1. Dataset-specific preparation. This involves creating train/dev/eval splits and converting the dataset to a standard
Kaldi-style file format. Note that all datasets are currently standardized to have lowercase text, where the only
non-linguistic symbol is `<noise>`.

2. Pre-computed speed perturbation (if desired). Specify this by providing the option 
`--speed-perturb-factors "0.9 1.0 1.1"`, where you can replace `"0.9 1.0 1.1"` with whatever factors you want. This
stage is skipped if you omit this option. Note that it is also possible to apply speed perturbation online if you
use `--feats-type raw`. 

3. Extract pre-computed features if desired, and dump the dataset to archive files. Each archive contains at least 1000
utterances (where possible), and at most 5 hours of audio.

4. Remove any utterances that are shorter than 100ms. If `--remove-empty-transcripts true` is specified
(default `true`), all utterances with no transcript are also removed. If `--remove-short-from-test` is specified
(default `false`), short utterances are removed from the test/dev sets as well as the training sets.

5. Compute CMVN statistics for all splits if features were pre-computed in stage 3 (`--feats-type fbank` or
`--feats-type fbank_pitch`). Specify CMVN type as a command line option (`--cmvn-type global` (default),
`--cmvn-type speaker`,  or `--cmvn-type utterance`). This stage is skipped if `--feats-type raw` is given. CMVN
statistics will be dumped to `<dataset>/dump/<feats_type>/<train_split>/<cmvn_type>_cmvn.ark` for each split
`<train_split>` of `<dataset>`. For global CMVN, CMVN statistics are also aggregated for all *training* splits
`<dataset>/dump/<feats_type>/global_cmvn.ark`. For utterance and speaker CMVN, CMVN statistics are aggregated for
all splits in `<dataset>/dump/<feats_type>/<cmvn_type>_cmvn.ark`. **NOTE: STAGE 5 DOES NOT APPLY CMVN TO YOUR DATA! 
IT ONLY COMPUTES THE STATISTICS. In order to apply CMVN, please supply a transformation of type `cmvn` to the data
loader (in Python) and specify the appropriate CMVN statistics file to do this.
See [here](speech_datasets/transform/README.md) for more details.**

6. Concatenate all text files (for language model training) into a single file.

7. Use the output of stage 6 to obtain a token inventory, either BPE's, characters, or words (specify 
`--token-type bpe` (default), `--token-type char`, or `--token-type word`, respectively). The maximum number of tokens
is controlled by `--n_tokens <n_tokens>` (default `2000`). Finally, a user can specify a set of non-linguistic symbols
by specifying the option `--nlsyms <nlsyms>`, where `<nlsyms>` is a comma-separated list of all non-linguistic symbols
(default `<noise>`). The resultant token list (and `sentencepiece` model for `--token-type bpe`) can be found in
`<dataset>/<asr1|tts1>/data/token_list`. 


### High-Level Python Interface
As alluded to above, the core functionality of this library is in the class `speech_datasets.SpeechDataLoader`. Its
constructor takes the following arguments:
- `datasets`: a list of strings specifying which datasets to load. Each dataset should be formatted as 
`<dataset_name>/<split_name>`, e.g. `wsj/train_si284`, `swbd/rt03`, `librispeech/dev-other`, etc.
- `task`: (default `"asr"`): either `"asr"` or `"tts"`
- `precomputed_feats_type`: (default `"raw"`): `"raw"`, `"fbank"`, or `"fbank_pitch"`. Tells the data loader where to
look for the archive files, and what sort of pre-computed features have been dumped to those archives.
- `transform_conf`: (default `None`): either the filename of a `yaml` file specifying a transformation, or a
`List[Dict[str, Any]]` specifying that transformation. `None` means no data transformation.
- `batch_size`: (default `1`): batch size
- `train`: (default `False`): whether the dataset's transform should be applied in training mode
- `shuffle`: (default `False`): whether to shuffle the dataset, using the policy described [above](#random-policy)
- `max_len`: the maximum average utterance length of a batch (after any data transformation is applied). The sum of
  utterance lengths in the batch is restricted to be at most `batch_size * max_len`.
- `num_replicas`: (default `None`): the number of distributed copies of the dataset being used, e.g. for training
  a `DistributedDataParallel` model. If you are running multiple jobs in parallel, and want each job to work on a
  different subset of the dataset, set this parameter to the total number of jobs. You probably don't need to specify
  this manually if you are using `torch.distributed`.
- `rank`: (default `None`): the index (amongst all distributed workers) of the current worker, e.g. for training
  a `DistributedDataParallel` model. If you are running multiple jobs in parallel, and want each job to work on
  a different subset of the dataset, set this parameter to the index of the current job. You probably don't need
  to specify this manually if you are using `torch.distributed`.
- `ensure_equal_parts`: Whether to ensure that all parallel processes receive the same number of batches to process.
  This should always be enabled if you are training with `DistributedDataParallel`, but you may wish to disable it
  if you are evaluating your model and wish to have each utterance processed exactly once.
- `n_transform_proc`: (default `None`): the number of parallel processes to use to apply the data transformation
  (specified by `transform_conf`) to the data being loaded. Default is `math.ceil(os.n_cpu() / num_replicas) - 1`.
- `data_cache_mb`: (default `4096`): the number of megabytes the cache (for pre-fetching archive files into memory,
  as described [above](#archive-cache)) can contain. 
- `spmodel`: (default `None`): the path to a `sentencepiece` BPE model to use to tokenize the text
- `token_list`: (default `None`): the path to a list of `sentencepiece` BPE units to use to tokenize the text.
  the indices in `token_list` override those in `spmodel`

The `speech_datasets.SpeechDataLoader` class inherits from `torch.utils.data.DataLoader`, and it implements `__len__`
and `__iter__` methods. To change the random seed used to order the data fetched, one should call
`loader.set_epoch(epoch)`, since the epoch number _is_ the random seed. This matches the implementation of
`torch.utils.data.distributed.DistributedSampler`. You should either
1. Use the data loader as a context manager, i.e.
    ```python
    with SpeechDataLoader(...) as loader:
       for batch in loader:
          # do stuff...
    ```
2. Invoke `loader.close()` after you are done, i.e.
    ```python
   loader = SpeechDataLoader(...)
   for batch in loader:
       # do stuff...
   loader.close()
    ```
This is because the asynchronous elements of the data loader (for pre-fetching data and computing data transformations
in the background) need to be shut down manually to fully terminate.

Finally, each batch is a list of dictionaries, one dictionary per utterance. The dictionary has the following keys:
- `uttid`: the utterance ID (`str`)
- `x`: the audio of the utterance (`torch.FloatTensor`), after any relevant data transformations have been applied
- `speaker`: the speaker of this utterance (`str`)
- `text`: a text transcription of the utterance (`str`)
- `labels`: a sequence of token indexes (`torch.IntTensor`) corresponding to the text transcript
  (only present if `spmodel` is provided to the constructor)

### Example Training Code
To make our proposed workflow more concrete, we provide a minimal [example](train_example/main.py) which uses 
`speech_datasets.SpeechDataLoader` to train a seq2seq transformer encoder-decoder model in PyTorch (this example also
depends on Huggingface's `transformers` library, which can installed by invoking `pip install transformers`). A
detailed walkthrough of this code can be found [here](train_example/README.md).

### Debugging Notes
1. If your code hangs, examine your log files. If you find `OSError: [Errno 12] Cannot allocate memory`, then try
    1. Setting `data_cache_mb` (in the constructor of the `SpeechDataLoader`) to a lower value. If you did not specify
    this manually, the default is `4096`.
    2. Setting `n_transform_proc` (in the constructor of the `SpeechDataLoader`) to a lower value. If you did not
    specify this manually, the default is `math.ceil(os.n_cpu() / num_replicas) - 1`, where `num_replicas` is the
    number of distributed replicas (either provided as an argument, or inferred automatically).
    3. Adding additional swap memory to your system.

# Combining Datasets
### Loading from Multiple Datasets
`speech_datasets.SpeechDataLoader` makes it transparent to load multiple datasets into memory at the same time. Simply
specify a list of strings (each formatted as `<dataset>/<split>`), and pass it as the `datasets` argument to the
`speech_datasets.SpeechDataLoader` constructor. For example,
```python
from speech_datasets import SpeechDataLoader

datasets = ["wsj/train_si284", "librispeech/train-other-500",
            "commonvoice/valid_train_en"]
with SpeechDataLoader(datasets, task="asr1", train=True, shuffle=True,
                      precomputed_feats_type="fbank") as loader:
    # do stuff
```
Because the data is loaded one archive at a time, and each archive file (by default) only contains data from a single
split, we suggest you follow one of these two strategies to ensure that any training data loaded from multiple splits
is properly randomized:
1. Explicitly prepare your training data as a combination of splits by navigating to `COMBINE/<asr1|tts1>` and running
   ```shell script
   ./combine_train_data.sh [asr.sh or tts.sh options] <dataset1>/<split1a> <dataset2>/<split2a> ...
   ```
   This script creates & registers a new sub-directory in [`COMBINE`](COMBINE), which contains all the data
   corresponding to this particular combination of data splits. For the example above, you can navigate to
   [`COMBINE/asr1`](COMBINE/asr1) and run
   ```shell script
   ./combine_train_data.sh --stage 2 --stop-stage 5 --speed-perturb-factors "0.9 1.0 1.1" --feats-type fbank \
       wsj/train_si284 librispeech/train-other-500 commonvoice/valid_train_en
   ```
   before using the Python code above.

   Note that `combine_train_data.sh` assumes that Stage 1 (dataset-specific preparation) has already been run for all
   the relevant datasets, and it does not support stages 6-7 (computing a token inventory), as these are handled by
   [`multi_tokenize.sh`](COMBINE/asr1/multi_tokenize.sh) (described below). Mis-specifying these options will result
   in an error message. Additionally, if you have already computed CMVN statistics for all the datasets you wish to
   combine, you may simply invoke [`combine_cmvn_stats.sh`](COMBINE/asr1/combine_cmvn_stats.sh) (described below).

2. Train your model using `DistributedDataParallel`. Each concurrent process will be working on a different archive
   file at any given time. Thus, while each archive file only contains data from a single split, the fact that there
   are multiple concurrent processes will greatly increase the probability that each batch contains utterances from
   multiple different data splits.

### Combining CMVN Statistics
Should you additionally wish to combine pre-computed CMVN statistics from multiple datasets, you can use the scripts
[`COMBINE/asr1/combine_cmvn_stats.sh`](COMBINE/asr1/combine_cmvn_stats.sh) or
[`COMBINE/tts1/combien_cmvn_stats.sh`](COMBINE/tts1/combine_cmvn_stats.sh).
Simply navigate to the relevant directory and invoke
```shell script
./combine_cmvn_stats.sh --cmvn_type <cmvn_type> --feats_type <feats_type> <dataset1>/<split1a> <dataset2>/<split2a> ...
```
where `<dataset>/<split>` represents the same format of specifying dataset splits used by the
`speech_datasets.SpeechDataLoader` class. Note that `multi_cmvn.sh` assumes that stage 5 of the top-level script has
already been run for all the relevant data splits, with the same `<cmvn_type>` and `<feats_type>`.


### Obtaining a Token Inventory Using Multiple Datasets
Finally, if you wish to construct a token inventory from multiple datasets (character, BPE, and/or word), we provide
the scripts [`COMBINE/asr1/multi_tokenize.sh`](COMBINE/asr1/multi_tokenize.sh) and
[`COMBINE/tts1/multi_tokenize.sh`](COMBINE/tts1/multi_tokenize.sh). Navigate to the relevant directory and invoke
```shell script
./multi_tokenize.sh --token_type <token_type> --n_tokens <n_tokens> --nlsyms <nlsyms> <dataset1> <dataset2> ...
``` 
Note that unlike the other API's specified thus far, `multi_tokenize.sh` simply takes a sequence of datasets as inputs,
not datasets and splits. This is because some data preparation recipes place additional text files (e.g. for language
modeling) in hard-to-reach reach locations, so they manually specify them in the `--srctexts` option of the top-level
script. WSJ is one such example. For example, you can call
```
./multi_tokenize.sh --token_type bpe --n_tokens 2000 swbd fisher wsj
```
to learn a BPE inventory that covers the training text data from Switchboard, Fisher, and WSJ. You can find the dumped
token lists & `sentencepiece` model files in `COMBINE/<asr1|tts1>/data/token_list`.


# Adding Support for New Datasets
See the relevant (linked) section of [`CONTRIBUTING.md`](CONTRIBUTING.md#adding-new-datasets) for details.


# Specifying Data Transformations
The high-level interface for data transformations is adapted from ESPNet's code and is defined in 
[`speech_datasets/transform/transformation.py`](speech_datasets/transform/transformation.py).
See [`speech_datasets/transform/README.md`](speech_datasets/transform/README.md) for details.


# Performance
### Benchmarks on ASR Tasks
TODO: add some training numbers w/ ESPNet models

### Latency
TODO: time for one epoch (ESPNet pre-computed feats vs. our pre-computed feats vs. online feats)

##### Notes
1. Experimentally, `hdf5` files are much faster to read when the data are `int` vectors (e.g. raw audio), 
while Kaldi's `mat` format is much faster to read when the data are `float`/`double` matrices (e.g. pre-extracted 
features). As such, these are the default archive formats used for dumping raw audio and pre-computed features,
respectively in `asr.sh`.


# References
Kaldi:
```
@INPROCEEDINGS{Povey_ASRU2011,
    author = {Povey, Daniel and Ghoshal, Arnab and Boulianne, Gilles and Burget, Lukas and Glembek, Ondrej and Goel, Nagendra and Hannemann, Mirko and Motlicek, Petr and Qian, Yanmin and Schwarz, Petr and Silovsky, Jan and Stemmer, Georg and Vesely, Karel},
    keywords = {ASR, Automatic Speech Recognition, GMM, HTK, SGMM},
    month = {Dec},
    title = {The Kaldi Speech Recognition Toolkit},
    booktitle = {IEEE 2011 Workshop on Automatic Speech Recognition and Understanding},
    year = {2011},
    publisher = {IEEE Signal Processing Society},
    location = {Hilton Waikoloa Village, Big Island, Hawaii, US},
    note = {IEEE Catalog No.: CFP11SRW-USB},
}
```

ESPNet:
```
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}

@inproceedings{hayashi2020espnet,
  title={{Espnet-TTS}: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Inoue, Katsuki and Yoshimura, Takenori and Watanabe, Shinji and Toda, Tomoki and Takeda, Kazuya and Zhang, Yu and Tan, Xu},
  booktitle={Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7654--7658},
  year={2020},
  organization={IEEE}
}
```

PyKaldi:
```
@inproceedings{pykaldi,
  title = {PyKaldi: A Python Wrapper for Kaldi},
  author = {Dogan Can and Victor R. Martinez and Pavlos Papadopoulos and
            Shrikanth S. Narayanan},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP),
             2018 IEEE International Conference on},
  year = {2018},
  organization = {IEEE}
}
```
