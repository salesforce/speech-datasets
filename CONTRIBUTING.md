# Table of Contents
1. [Adding New Datasets](#adding-new-datasets)
2. [Defining New Data Transformations](#defining-new-data-transformations)
3. [Pre-Computing Different Features](#pre-computing-different-features)

# Adding New Datasets
To add support for a new dataset `<dataset>`, you first need to create a directory `<dataset>/asr1` for ASR, or
`<dataset>/<tts1>` for TTS. Then, invoke [`TEMPLATE/asr1/setup.sh`](TEMPLATE/asr1/setup.sh)` <dataset>/asr1` or
[`TEMPLATE/tts1/setup.sh`](TEMPLATE/tts1/setup.sh)` <dataset>/tts1` to set up the directory for ASR or TTS,
respectively. This creates symbolic links to shared scripts in the [`TEMPLATE`](TEMPLATE) directory, and also copies
over some standard configuration files for feature computation.

Next, you need to create the script `<dataset>/<asr1|tts1>/local/data.sh`, which is responsible for preparing all the
data splits for this dataset in Kaldi's format. For split `<split>` of dataset `<dataset>`, we expect the following
files (as described in Kaldi's official [documentation](https://kaldi-asr.org/doc/data_prep.html)) to be dumped to the
directory `<dataset>/<asr1|tts1>/data/<split>`:
- `wav.scp`: Assigns to each recording a filename (corresponding to a `.wav` file), or an extended filename (a Linux
command directed to an output pipe, which would produce the desired `.wav` file)
- `text`: Associates each utterance ID with its transcript. Note that we expect
    - All transcripts are fully lowercase
    - All non-linguistic symbols representing different kinds of noise are represented by the token `<noise>`
- `utt2spk`: Associates each utterance ID to a speaker ID. We expect that
    - The prefix of the speaker ID is `{<dataset>}`, e.g. `{wsj}011` or `{librispeech}1006-135212`
    - The prefix of the utterance ID is the speaker ID, e.g. `{wsj}011c0202` or `{librispeech}1006-135212-0001`
- `segments`: Defines each utterance ID as a segment of a recording. Optional, used only in cases where recordings
contain multiple utterances (e.g. Switchboard).

We recommend saving any helper scripts to `<dataset>/<asr1|tts1>/local` as well. More details on each of the above
files can be found in Kaldi's official [documentation](https://kaldi-asr.org/doc/data_prep.html). We also suggest that
you update [`TEMPLATE/asr1/db.sh`](TEMPLATE/asr1/db.sh) or [`TEMPLATE/tts1/db.sh`](TEMPLATE/tts1/db.sh) with variables
that specify the absolute path where one can find the downloaded, unprocessed dataset.

Finally, you need to write a top-level `<dataset>/<asr1|tts1>/run.sh` script which invokes either `asr.sh` or 
`tts.sh` with the appropriate dataset-specific arguments. The most important ones are `--fs` (the sampling frequency
in Hz, default `16000`), `--train-sets` (the training splits, required argument), `--dev-eval-sets` (the development &
evaluation splits, optional argument), and `--srctexts` (the `text` files (as described above) to use for training a
language model & compiling a token inventory).

### Notes
1. The sampling frequency `--fs` is the frequency at which the raw audio will be resampled (if necessary) before being
dumped. If you specify `--fs` in `run.sh`, make sure to also change the value of `sample_frequency` in
`conf/fbank.yaml` and `conf/fbank_pitch.yaml`! For example, compare
[`swbd/asr1/run.sh`](swbd/asr1/run.sh) and [`swbd/asr1/conf/fbank.yaml`](swbd/asr1/conf/fbank.yaml) (sampling frequency
`8000`) to [`librispeech/asr1/run.sh`](librispeech/asr1/run.sh) and
[`librispeech/asr1/conf/fbank.yaml`](librispeech/asr1/conf/fbank.yaml) (sampling frequency `16000`). 
If you wanted to up-sample Switchboard to 16kHz, you would change these values to `16000`. 
Conversely, if you wanted to down-sample LibriSpeech to 8kHz, you would change these values to `8000`.

2. Note that `--train-sets` and `--dev-eval-sets` are simply split names (`asr.sh` or `tts.sh` will look for split
`<split>` in the directory `data/<split>`), while `--srctexts` should specify actual filenames.

3. If the recipe depends on some special tools, then write the requirements to the top of `run.sh`:
    ```bash
    # e.g. flac command is required
    if ! which flac &> /dev/null; then
        echo "Error: flac is not installed"
        return 1
    fi
    ```


# Defining New Data Transformations
To define a new data transformation, first create a new class that implements the interface
`speech_datasets.transform.interface.TransformInterface`. Make sure this class is implemented in 
[`speech_datasets/transform`](speech_datasets/transform). Next, register your new class in the `import_alias`
dictionary of [`speech_datasets/transform/transformation.py`](speech_datasets/transform/transformation.py#L25).
Add documentation for this transformation in 
[`speech_datasets/transform/README.md`](speech_datasets/transform/README.md).
You can now specify this transformation in a `yaml` configuration file. 

# Pre-Computing Different Features
Currently, we allow you to pre-compute `fbank` or `fbank_pitch` features, which are specified by the configuration files
`conf/fbank.yaml` and `conf/fbank_pitch.yaml`. To add support for a new feature type `<feats_type>`, you can simply add
a new configuration file `conf/<feats_type>.yaml` which specifies your data transformation. Then, update the following
block of [`TEMPLATE/asr1/asr.sh`](TEMPLATE/asr1/asr.sh#L141) to define a dump directory and configuration file for your
new feature transformation:
```shell script
# Check feature type
# Feel free to update this section to support additional feature transforms!
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/${feats_type}
    feat_conf=none
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/${feats_type}
    feat_conf="conf/${feats_type}.yaml"
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/${feats_type}
    feat_conf="conf/${feats_type}.yaml"
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi
```
Note that these configuration files are exactly the same as those used for specifying data transformations, as described
in [`speech_datasets/transform/README.md`](speech_datasets/transform/README.md)
