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

# Get the datasets we want to use based on the command-line args
train_sets="train-clean-100 train-clean-360 train-other-500 "
dev_sets="dev-clean dev-other "
eval_sets="test-clean test-other "
srctexts=
for dset in ${train_sets}; do
    srctexts+="data/${dset}/text "
done

./asr.sh \
    --fs 16000 \
    --n_tokens 2000 \
    --token_type bpe \
    --train_sets "${train_sets}" \
    --dev_eval_sets "${dev_sets} ${eval_sets}" \
    --srctexts "${srctexts}" \
    --local_data_opts "${eval_sets} ${dev_sets} ${train_sets}" "$@"