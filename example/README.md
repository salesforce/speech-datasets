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

running this gives me python package error.
I needed to run `pip install pandas ray[tune]`

```
python main.py --feats_type <fbank|fbank_pitch>
```

After dumping raw and running the main.py, I'm getting the following error:
python main.py --feats_type fbank
10/30/2020 14:40:28 - INFO - __main__ - rank=-1 - Initializing data loaders...
2020-10-30 14:40:29,019	INFO services.py:1166 -- View the Ray dashboard at http://127.0.0.1:8265
10/30/2020 14:40:29 - INFO - __main__ - rank=-1 - Initializing model & optimizer...
10/30/2020 14:40:31 - INFO - __main__ - rank=-1 - Starting epoch 1...
  0%|                                                                 | 0/5159 [00:06<?, ?it/s]
Traceback (most recent call last):
  File "main.py", line 216, in <module>
    main()
  File "main.py", line 166, in main
    logits = model(xs, x_lens, ys, y_lens)[:, :-1]
  File "/home/ykang/Repositories/speech-datasets/tools/venv/envs/venv/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/ykang/Repositories/sdl/example/model.py", line 122, in forward
    x, x_mask = self.conv(xs, x_mask)
  File "/home/ykang/Repositories/speech-datasets/tools/venv/envs/venv/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/ykang/Repositories/sdl/example/model.py", line 87, in forward
    x = self.out(x.transpose(1, 2).contiguous().view(n, t, c*d))
  File "/home/ykang/Repositories/speech-datasets/tools/venv/envs/venv/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/ykang/Repositories/speech-datasets/tools/venv/envs/venv/lib/python3.7/site-packages/torch/nn/modules/container.py", line 100, in forward
    input = module(input)
  File "/home/ykang/Repositories/speech-datasets/tools/venv/envs/venv/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/ykang/Repositories/speech-datasets/tools/venv/envs/venv/lib/python3.7/site-packages/torch/nn/modules/linear.py", line 87, in forward
    return F.linear(input, self.weight, self.bias)
  File "/home/ykang/Repositories/speech-datasets/tools/venv/envs/venv/lib/python3.7/site-packages/torch/nn/functional.py", line 1372, in linear
    output = input.matmul(weight.t())
RuntimeError: size mismatch, m1: [2148 x 4864], m2: [5120 x 256] at /opt/conda/conda-bld/pytorch_1579022060824/work/aten/src/THC/generic/THCTensorMathBlas.cu:290

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
