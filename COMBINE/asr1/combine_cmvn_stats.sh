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

feats_type=fbank   # fbank or fbank_pitch are valid
cmvn_type=global   # global or speaker or utterance are valid

help_message=$(cat <<EOF
Usage: $0 [options] <dataset1>/<split1a> <dataset2>/<split2a> <dataset2>/<split2b> ...

Combines CMVN stats for the specified dataset splits (pre-computed by Stage 5 run.sh for each dataset split specified)
into a single file.

Options:
    --feats_type          # Feature type (fbank or fbank_pitch) (default=${feats_type}).
    --cmvn_type           # Type of CMVN stats to compute (global or speaker or utterance) (default=${cmvn_type}).
EOF
)


. ./path.sh || exit 1
. ./cmd.sh || exit 1

log "$0 $*"
. utils/parse_options.sh || exit 1
if [ $# -eq 0 ]; then
    log "${help_message}"
    log "Error: Please specify dataset splits as positional arguments."
    exit 2
fi

workspace=$PWD
task=$(basename "$(utils/make_absolute.sh "$workspace")")

# Get CMVN's from all the relevant dataset splits
cmvns=
for dset in "$@"; do
    base=$(echo ${dset} | sed -E "s/\/.*//g")
    split=$(echo ${dset} | sed -E "s/.*\///g")
    base_dir="${MAIN_ROOT}/${base}/${task}"
    dset_dir="${base_dir}/dump/${feats_type}"/${split}
    cmvn="${dset_dir}/${cmvn_type}_cmvn.ark"

    if [ ! -d ${base_dir} ]; then
        log "${base} is not a valid dataset for task ${task//1/}"
        exit 1
    elif [ "${base}" = "${dset}" ]; then
        log "Expected dataset to specified as <dataset>/<split>, but got ${dset}"
        exit 1
    elif [ ! -d ${dset_dir} ]; then
        log "Either ${split} is not a valid split for dataset ${base}, or"
        log "${base_dir}/run.sh has not yet been run with feats_type=${feats_type}"
        exit 1
    elif [ ! -f ${cmvn} ]; then
        log "${cmvn_type} CMVN statistics have not been computed for feats_type=${feats_type} for data split ${dset}."
        log "Please run stage 5 of ${base_dir}/${task}/run.sh."
        exit 1
    fi
    cmvns+="${cmvn} "
done

# Combine CMVN's
combo_idx=$(python3 local/combine_datasets.py --task "${task//1/}" --write_dir false "$@")
dumpdir="dump/${feats_type}/no_short/${combo_idx}"
mkdir -p "${dumpdir}"
python3 -m speech_datasets.bin.combine_cmvn_stats --cmvn_type ${cmvn_type} \
    --output_file "${dumpdir}/${cmvn_type}_cmvn.ark" ${cmvns}
