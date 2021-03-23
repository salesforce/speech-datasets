#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'unbound variable', -o ... 'error in pipeline'
set -e
set -u
set -o pipefail

train_set="train_si284 "
dev_set="test_dev93 "
eval_sets="test_eval92 "

# Even though data/nlsyms.txt is generated, we don't provide it to asr.sh
# because the only non-linguistic symbol it contains is "<noise>", which is
# which is the default value for nlysms.
./asr.sh \
    --fs 16000 \
    --n_tokens 75 \
    --token_type bpe \
    --train_sets "${train_set}" \
    --dev_eval_sets "${dev_set} ${eval_sets}" \
    --srctexts "data/train_si284/text data/local/other_text/text" "$@"
