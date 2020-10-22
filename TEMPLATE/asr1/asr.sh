#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
max() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -ge "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1          # Processes starts from this stage.
stop_stage=7     # Processes is stopped at this stage.
nj=$(nproc)      # Number of parallel CPU jobs
dumpdir=dump     # Directory to dump features.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=    # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw      # Feature type (raw or fbank or fbank_pitch).
audio_format=wav    # Audio format (only in feats_type=raw, archive_format=sound).
archive_format=     # Audio archive format
                    # (hdf5 (default) or mat or sound for feats_type = raw)
                    # (hdf5 or mat (default) for feats_type = fbank_pitch or fbank)
fs=16000            # Sampling rate (only in feats_type = raw).
cmvn_type=global    # Type of CMVN statistics to compute (global or speaker or utterance) (defaut=global)
                    # (only in feats_type=fbank or feats_type=fbank_pitch)
remove_short_from_test=false  # whether to remove short utterances from test/dev sets (stage 4)
remove_empty_transcripts=true # whether to remove utterances with no transcript (stage 4)

# Tokenization related
token_type=bpe      # Tokenization type (char or bpe or word).
n_tokens=2000       # The maximum number of tokens allowed.
bpemode=unigram     # Mode of BPE (unigram or bpe).
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_char_cover=1.0  # character coverage when modeling BPE
nlsyms="<noise>"    # non-linguistic symbols list, separated by a comma, for (BPE & char)

# [Task dependent] Set the datadir name created by local/data.sh
train_sets=     # Name of training set(s). Multiple items can be specified.
dev_eval_sets=  # Name of development & evaluation set(s). Multiple items can be specified.
srctexts=       # Used for the training of BPE and LM and the creation of a vocabulary list.

help_message=$(cat << EOF
Usage: $0 --train-set <train_set_names> --dev--eval-sets <dev_eval_set_names> --srctexts <srctexts>

Options:
    # General configuration
    --stage       # Processes starts from this stage (default="${stage}").
    --stop_stage  # Processes is stopped at this stage (default="${stop_stage}").
    --nj          # The number of parallel jobs (default="${nj}").
    --dumpdir     # Directory to dump features (default="${dumpdir}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors   # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type      # Feature type (raw or fbank_pitch or fbank, default="${feats_type}").
    --audio_format    # Audio format (only in feats_type=raw, default="${audio_format}").
    --archive_format  # Format in which to store extracted features/audio
                      # If feats_type=raw, valid options are hdf5, sound, mat (default=hdf5)
                      # Otherwise, valid options are hdf5, mat (default=mat)
    --fs              # Sampling rate (only in feats_type=raw, default="${fs}").
    --cmvn_type       # Type of CMVN statistics to compute (global or speaker or utterance, default=${cmvn_type})
                      # (only used when feats_type=fbank or feats_type=fbank_pitch)
    --remove_short_from_test   # whether to remove short utterances from test/dev sets (default=${remove_short_from_test})
    --remove_empty_transcripts # whether to remove utterances with no transcript (default=${remove_empty_transcripts})

    # Tokenization related
    --token_type              # Tokenization type (char or bpe or word, default="${token_type}").
    --n_tokens                # The maximum number of tokens allowed (default="${n_tokens}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").
    --nlsyms                  # Non-linguistic symbol list for BPE/char, separated by a comma. (default="${nlsyms}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_sets    # Name of training set(s) (required).
    --dev_eval_sets # Name of development & evaluation set(s) (optional).
    --srctexts      # Used for the training of BPE and LM and the creation of a vocabulary list (required).
EOF
)

. ./path.sh || exit 1
. ./cmd.sh || exit 1

log "$0 $*"
. utils/parse_options.sh || exit 1

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

# Check required arguments
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 2 ] && [ -z "${train_sets}" ]; then
    log "${help_message}"
    log "Error: --train_sets is required for stages 2-5. Got stage=${stage}, stop_stage=${stop_stage}."
    exit 2
fi
if [ ${stage} -ge 6 ] && [ ${stop_stage} -le 6 ] && [ -z "${srctexts}" ]; then
    log "${help_message}"
    log "Error: --srctexts is required for stages 6. Got stage=${stage}, stop_stage=${stop_stage}"
    exit 2
fi

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

# Check archive format (set default if needed)
if [ -z ${archive_format} ] && [ "${feats_type}" = raw ]; then
    archive_format=hdf5
elif [ -z ${archive_format} ]; then
    archive_format=mat
elif [ "${archive_format}" != "hdf5" ] && [ "${archive_format}" != "sound" ] && [ "${archive_format}" != "mat" ]; then
    log "${help_message}"
    log "Error: not supported: --archive_format ${archive_format}"
    exit 2
fi

# Check audio format (raw only)
if [ ${archive_format} != sound ] && [ ${audio_format} != wav ]; then
    log "[warning]: --audio-format ${audio_format} option unused. Replacing with audio_format=wav"
    audio_format=wav
fi

# Check tokenization type
token_listdir=data/token_list
if [ "${token_type}" = bpe ]; then
    bpedir="${token_listdir}/${token_type}_${bpemode}${n_tokens}"
elif [ "${token_type}" = char ]; then
    bpedir="${token_listdir}/${token_type}"
    bpemode=${token_type}
elif [ "${token_type}" = word ]; then
    bpedir="${token_listdir}/${token_type}${n_tokens}"
    bpemode=${token_type}
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi
token_list="${bpedir}/tokens.txt"
bpeprefix="${bpedir}"/model

# =============================================================================== #
#                  Standard data preparation stages start here                    #
# =============================================================================== #

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    dev0=$(echo ${dev_eval_sets} | cut -f 1 -d" ")
    train0=$(echo ${train_sets} | cut -f 1 -d" ")
    log "Stage 1: Data preparation for data/${train0}, data/${dev0}, etc."
    # [Task dependent] Need to create a new local/data.sh for each new dataset
    local/data.sh ${local_data_opts}
fi

# Create spk2utt for datasets (if needed)
for dset in ${train_sets} ${dev_eval_sets}; do
    if [ ! -f "data/${dset}/utt2spk" ]; then
        log "Dataset ${dset} was not prepared properly. utt2spk missing."
        exit 1
    elif [ ! -f "data/${dset}/spk2utt" ]; then
        log "Dataset ${dset} does not have spk2utt, so creating one now."
        utils/utt2spk_to_spk2utt.pl "data/${dset}/utt2spk" > "data/${dset}/spk2utt"
    fi
done

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ -n "${speed_perturb_factors}" ]; then
        for dset in ${train_sets}; do
            dset_dirs=
            log "Stage 2: Speed perturbation: data/${dset} -> data/${dset}_sp"
            for factor in ${speed_perturb_factors}; do
                if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
                    utils/perturb_data_dir_speed.sh "${factor}" "data/${dset}" "data/${dset}_sp${factor}"
                    dset_dirs+="data/${dset}_sp${factor} "
                else
                    # If speed factor is 1, same as the original
                    dset_dirs+="data/${dset} "
                fi
            done
            utils/combine_data.sh "data/${dset}_sp" ${dset_dirs}
        done
    else
        log "Skip stage 2: Speed perturbation"
    fi
fi

if [ -n "${speed_perturb_factors}" ]; then
    new_train_sets=
    for dset in ${train_sets}; do
        new_train_sets+="${dset}_sp "
    done
    train_sets=$(echo "${new_train_sets}" | sed -E "s/[[:space:]]*$//")
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: ${feats_type} extract: data/ -> ${data_feats}/orig"

    # Generate a shuffled wav.scp for all the training data. This file is used
    # to generate the data archives, and it's easier to randomize batches when
    # the individual archive files aren't clustered by speaker (which happens
    # when the original wav.scp is in sorted order)
    seed=20210719
    for dset in ${train_sets}; do
        if [ -e "data/${dset}/segments" ]; then
            utils/shuffle_list.pl --srand ${seed} "data/${dset}/segments" \
                > "data/${dset}/segments_shuf"
        fi
        utils/shuffle_list.pl --srand ${seed} "data/${dset}/wav.scp" \
            > "data/${dset}/wav_shuf.scp"
        seed=$((seed+1))
    done

    # Don't shuffle the dev & eval data (grouping by speaker is useful)
    for dset in ${dev_eval_sets}; do
        if [ -e "data/${dset}/segments" ]; then
            cp "data/${dset}/segments" "data/${dset}/segments_shuf"
        fi
        cp "data/${dset}/wav.scp" "data/${dset}/wav_shuf.scp"
    done

    for dset in ${train_sets} ${dev_eval_sets}; do
        # 1. Copy datadir
        utils/copy_data_dir.sh data/"${dset}" "${data_feats}/orig/${dset}"
        rm -f ${data_feats}/orig/"${dset}"/{segments,reco2file_and_channel}
        rm -f ${data_feats}/orig/"${dset}"/{${audio_format},wav,feats}.scp

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe,
        # like "cat /some/path |", shouldn't be used in the training process.
        # "dump.sh" can dump such pipe-style-wav to real audio file (if
        # feats_type=raw) and also it can also change the audio-format and
        # sampling rate. It packages the result in the archive format specified.
        maybe_segments=
        if [ -e data/"${dset}"/segments_shuf ]; then
            # "segments" is used for splitting the wav files which are written
            # in wav.scp into utterances. The file format of segments:
            #   <segment_id> <record_id> <start_time> <end_time>
            #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
            # Where the time is written in seconds.
            maybe_segments+="--segments data/${dset}/segments_shuf "
        fi

        # 2. Dump the appropriate output, depending on feats_type
        n_utts=$(<"${data_feats}/orig/${dset}/utt2spk" wc -l)
        max_n_splits=$(max 1 "$(echo "${n_utts} / 1000" | bc)")
        _nj=$(min "${nj}" "${max_n_splits}")
        if [ "${feats_type}" = raw ]; then
            data_format=${audio_format}
        else
            data_format="feats"
        fi
        utils/dump.sh --cmd "${train_cmd}" --data-name "${feats_type}_${dset}" \
            --nj "${_nj}" --fs ${fs} --conf "${feat_conf}" \
            --data_format ${data_format} --archive-format "${archive_format}" \
            --text "data/${dset}/text" --utt2spk "data/${dset}/utt2spk" \
            ${maybe_segments} "data/${dset}/wav_shuf.scp" "${data_feats}/orig/${dset}"

        # 3a. Derive the shape of each utterance matrix
        _nj=$(min "${nj}" "${n_utts}")
        utils/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
            --archive_format "${archive_format}" "${data_feats}/orig/${dset}/${data_format}.scp" \
            "${data_feats}/orig/${dset}/feats_shape"

        # 3b. Generate utt2num_frames from feats_shape
        paste -d " " <(cut -f 1 -d" " "${data_feats}/orig/${dset}/feats_shape") \
            <(cut -f 2 -d" " "${data_feats}/orig/${dset}/feats_shape" | cut -f 1 -d",") \
            > "${data_feats}/orig/${dset}/utt2num_frames"

        # 4. Write various metadata files, and clean up the data directory
        rm -f "data/${dset}/wav_shuf.scp"
        rm -f "data/${dset}/segments_shuf"
        echo ${feats_type} > "${data_feats}/orig/${dset}/feats_type"
        echo ${data_format} > "${data_feats}/orig/${dset}/data_format"
        echo ${archive_format} > "${data_feats}/orig/${dset}/archive_format"
        utils/fix_data_dir.sh "${data_feats}/orig/${dset}"
    done
    for dirname in "${data_feats}"/orig/*; do
        if [ -d "${dirname}" ]; then
            ln -sf "orig/$(basename "${dirname}")" "${data_feats}"
        fi
    done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # Note: short data is removed primarily by removing entries from the
    # wav.scp/feats.scp index file. The files will not be removed, but the data
    # simply won't be loaded at runtime if you don't want it.
    log "Stage 4: Remove short data: ${data_feats}/orig -> ${data_feats}/no_short"
    if [ $remove_short_from_test = true ]; then
        dsets_to_shorten="${train_sets} ${dev_eval_sets}"
    else
        dsets_to_shorten=${train_sets}
        for dset in ${dev_eval_sets}; do
            ln -sf "orig/${dset}" "${data_feats}"
        done
    fi
    for dset in ${dsets_to_shorten}; do

        # Get various metadata about the dataset
        _feats_type="$(<${data_feats}/orig/${dset}/feats_type)"
        _data_format="$(<${data_feats}/orig/${dset}/data_format)"

        # Copy the data directory (validate with the appropriate .scp filename)
        utils/copy_data_dir.sh --validate_opts "--scp-name ${_data_format}.scp" \
            "${data_feats}/orig/${dset}" "${data_feats}/no_short/${dset}"
        cp "${data_feats}/orig/${dset}/feats_type" "${data_feats}/no_short/${dset}/feats_type"
        cp "${data_feats}/orig/${dset}/data_format" "${data_feats}/no_short/${dset}/data_format"
        cp "${data_feats}/orig/${dset}/archive_format" "${data_feats}/no_short/${dset}/archive_format"

        # Define what "short" utterances are (for raw audio vs features),
        if [ "${_feats_type}" = raw ]; then
            min_length=2560
        else
            min_length=10
        fi

        # Remove short utterances
        <"${data_feats}/orig/${dset}/utt2num_frames" \
            awk -v min_length="$min_length" '{ if ($2 > min_length) print $0; }' \
            >"${data_feats}/no_short/${dset}/utt2num_frames"
        <"${data_feats}/orig/${dset}/${_data_format}.scp" \
            utils/filter_scp.pl "${data_feats}/no_short/${dset}/utt2num_frames"  \
            >"${data_feats}/no_short/${dset}/${_data_format}.scp"

        # Remove utterances with no accompanying text (if desired)
        if [ ${remove_empty_transcripts} = true ]; then
            awk ' { if( NF != 1 ) print $0; } ' "${data_feats}/orig/${dset}/text" \
                >"${data_feats}/no_short/${dset}/text"
        fi

        # fix_data_dir.sh leaves only utts which exist in all files
        utils/fix_data_dir.sh "${data_feats}/no_short/${dset}"
        ln -sf "no_short/${dset}" "${data_feats}"
    done
fi

# =============================================================================== #
#   You may wish to MANUALLY specify srctexts and train_sets for these stages to  #
#         aggregate CMVN statistics & texts from multiple training sets           #
# =============================================================================== #

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ "${feats_type}" != fbank ] && [ "${feats_type}" != fbank_pitch ]; then
        log "Skip Stage 5: Compute ${cmvn_type} CMVN stats for training data"
        log "Specify feats_type=fbank or fbank_pitch (feats_type=${feats_type} received) if you want CMVN statistics."

    else
        log "Stage 5: Compute ${cmvn_type} CVMN stats for training data"

        # Compute CMVN stats for all splits
        for dset in ${train_sets} ${dev_eval_sets}; do
            _nj=$(min "${nj}" "$(<"${data_feats}/${dset}/utt2spk" wc -l)")
            utils/compute_cmvn_stats.sh --cmd ${train_cmd} --nj "${_nj}" \
                --spk2utt "${data_feats}/${dset}/spk2utt" \
                --cmvn-type ${cmvn_type} --archive_format ${archive_format} \
                "${data_feats}/${dset}/feats.scp" \
                "${data_feats}/${dset}/${cmvn_type}_cmvn.ark"
        done

        # Combine CMVN stats for only train splits if global, all splits otherwise
        cmvns=
        if [ ${cmvn_type} = global ]; then
            combine_cmvn_sets="${train_sets}"
        else
            combine_cmvn_sets="${train_sets} ${dev_eval_sets}"
        fi
        for dset in ${combine_cmvn_sets}; do
            cmvns+="${data_feats}/${dset}/${cmvn_type}_cmvn.ark "
        done
        python3 -m speech_datasets.bin.combine_cmvn_stats --cmvn_type ${cmvn_type} \
            --output_file ${data_feats}/${cmvn_type}_cmvn.ark ${cmvns}
    fi
fi


# Note: this stage doesn't do much, but it's useful that it's a separate
# step for COMBINE/<asr1|tts1>/multi_tokenize.sh
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Concatenating all source transcripts into dump/srctexts"
    # The src text files are formatted as <utt_id> <transcript>
    # Remove lines with no transcripts (NF == 1),
    mkdir -p dump
    # shellcheck disable=SC2002
    cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' > "dump/srctexts"
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Generate token_list from srctexts using ${token_type} tokens"
    mkdir -p "${bpedir}"

    # The src text files are formatted as <utt_id> <transcript>
    # Discard the utterance ID from each line (delimiting fields by " ", only
    # keep fields 2 and up). This the text we will use to train our BPE model.
    cut -d" " -f 2- < "dump/srctexts" > "${bpedir}"/train.txt

    if [ -n "${nlsyms}" ]; then
        _opts_spm="--user_defined_symbols=${nlsyms}"
    else
        _opts_spm=""
    fi

    # n_tokens - 2 to account for <sos/eos> and <blank>
    python3 -m speech_datasets.bin.spm_train \
        --input="${bpedir}"/train.txt --vocab_size="${n_tokens}" \
        --model_type="${bpemode}" --model_prefix="${bpeprefix}" \
        --character_coverage=${bpe_char_cover} \
        --input_sentence_size="${bpe_input_sentence_size}" \
        --unk_surface="<unk>" ${_opts_spm} --unk_id=0 --bos_id=1 --eos_id=2

    # Manually specify <blank> = 0, <unk> = 1
    echo -e "<blank>\n<unk>" > "${token_list}"
    # First 3 lines of model.vocab are <unk>, <s>, </s>, so get toks from line 4
    tail -n +4 "${bpeprefix}.vocab" | awk '{ print $1 }' >> "${token_list}"
    # Last token is <sos/eos>
    echo "<sos/eos>" >> "${token_list}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
