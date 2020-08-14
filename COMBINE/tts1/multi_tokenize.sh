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

# Tokenization related options from asr.sh
token_type=bpe          # Tokenization type (char or bpe).
n_tokens=2000           # The number of BPE vocabulary.
nlsyms="<noise>"        # non-linguistic symbols list, separated by a comma

help_message=$(cat <<EOF
Usage: $0 [options] <dataset1> <dataset2> <dataset3> ...

Produces a token inventory of the given type for all the datasets provided.

Options:
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --n_tokens                # The maximum number of tokens allowed (default="${n_tokens}").
    --nlsyms                  # Non-linguistic symbol list for BPE/char, separated by a comma. (default="${nlsyms}").
EOF
)

. ./path.sh || exit 1
. ./cmd.sh || exit 1

log "$0 $*"
. utils/parse_options.sh || exit 1
if [ $# -eq 0 ]; then
    log "${help_message}"
    log "Error: Please specify datasets as positional arguments."
    exit 2
fi

workspace=$PWD
task=$(basename "$(utils/make_absolute.sh "$workspace")")
run_args="--token-type ${token_type} --n_tokens ${n_tokens} --nlsyms ${nlsyms} "

# Compile srctexts from all the relevant datasets
srctexts=
for dset in "$@"; do
    log "Concatenating all source texts from dataset $dset..."
    dset_dir="${MAIN_ROOT}/${dset}/${task}"
    cd ${dset_dir}
    ./run.sh --stage 6 --stop-stage 6 ${run_args}
    cd ${workspace}
    srctexts+="${dset_dir}/dump/srctexts "
    echo ""
done

# Concatenate all the relevant text data & prepare a token inventory
log "Concatenating all source texts from all datasets..."
mkdir -p dump data
cat $srctexts > dump/srctexts
./run.sh --stage 7 --stop-stage 7 ${run_args}

