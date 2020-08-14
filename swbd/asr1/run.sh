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

train_sets="swbd1_train "
dev_set="swbd1_dev"
eval_sets="eval2000 rt03 "
srctexts="data/swbd1_train/text "

./asr.sh \
    --fs 8000 \
    --n_tokens 2000 \
    --token_type bpe \
    --train_sets "${train_sets}" \
    --dev_eval_sets "${dev_set} ${eval_sets}" \
    --srctexts "${srctexts}" "$@"
