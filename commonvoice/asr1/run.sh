#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=en # en de fr cy tt kab ca zh-TW it fa eu es ru

train_set=valid_train_${lang}
train_dev=valid_dev_${lang}
train_test=valid_test_${lang}

./asr.sh \
    --local_data_opts "--lang ${lang}" \
    --fs 16000 \
    --n_tokens 2000 \
    --token_type bpe \
    --feats_type fbank_pitch \
    --train_sets "${train_set}" \
    --dev_eval_sets "${train_dev} ${train_test}" \
    --srctexts "data/${train_set}/text" "$@"
