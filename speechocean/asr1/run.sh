#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
dev_set="dev"
eval_sets="test"
srctexts="data/train/text"

./asr.sh \
    --fs 16000 \
    --n_tokens 2000 \
    --token_type bpe \
    --train_sets "${train_set}" \
    --dev_eval_sets "${dev_set} ${eval_sets}" \
    --srctexts "${srctexts}" \
    --local_data_opts "${eval_sets} ${dev_set} ${train_set}" "$@"
