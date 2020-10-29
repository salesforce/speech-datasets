# Example Code
This directory provides an example which trains a transformer encoder-decoder model on the
`train-clean-100` and `train-clean-360` splits of LibriSpeech, and it evaluates the model's
performance on the `dev-clean` split.

In order to run this example, you must first prepare an environment and install the `speech_datasets` package,
as detailed [here](../README.md#environment-setup). Next, navigate to [librispeech/asr1](../librispeech/asr1) and
invoke
```shell script
./run.sh --stage 1 --stop_stage 4 --feats_type <raw|fbank|fbank_pitch>
```
This will download, prepare, and extract the relevant features for LibriSpeech, and make the dataset usable with
the `speech_datasets` package. Note that this step will take a long time!

Next, you should navigate to this directory and activate the conda environment by invoking
```
source ../tools/venv/bin/activate && conda deactivate && conda activate <VENV_NAME>
```
(where `<VENV_NAME>` is the name of the conda virtual environment, `datasets` by default if you did not specify it
when setting up your environment as described [here](../README.md#environment-setup)). Now, you can run
[`main.py`](main.py). If you dumped `--feats_type raw`, then you can run
```
python main.py --feats_type <fbank|fbank_pitch>
```
If you instead dumped `--feats_type fbank` or `--feats_type fbank_pitch`, you can instead run
```
python main.py --feats_type <fbank|fbank_pitch> --precomputed_feats
```

The `feats_type` argument to `main.py` will specify whether to use the feature computation configuration
[`fbank.yaml`](resources/fbank.yaml) or [`fbank_pitch.yaml`](resources/fbank_pitch.yaml).
Both compute 80-dimensional filterbank features (optionally pitch as well), apply the appropriate cepstral
mean/variance normalization (using the statistics pre-computed in
[`global_cmvn_fbank.ark`](resources/global_cmvn_fbank.ark) or
[`global_cmvn_fbank_pitch.ark`](resources/global_cmvn_fbank_pitch.ark)), and apply spectral augmentation.

In this example, the data loader will tokenize the text using the provided sentencepiece model 
[`librispeech_bpe2000.model`](resources/librispeech_bpe2000.model). See the `main()` function of
[`main.py`](main.py) for a full example.
