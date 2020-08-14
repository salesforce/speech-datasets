#!/bin/bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0
(No options)
EOF
)

if [ $# -ne 0 ]; then
    log "Error: invalid command line arguments"
    log "${help_message}"
    exit 1
fi

. ./path.sh || exit 1
. ./db.sh || exit 1

other_text=data/local/other_text/text
nlsyms=data/nlsyms.txt

# Get WSJ0 raw data from /export/share if needed
if [ -z "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' in db.sh"
    exit 1
elif [ ! -e "${WSJ0}" ]; then
    WSJ0_base=$(dirname "${WSJ0}")
    mkdir -p "${WSJ0_base}"
    {
      tar xzvf /export/share/ybzhou/dataset/LDC/csr_1_senn_LDC93S6B.tgz -C "${WSJ0_base}"
    } || {
      rm -rf "${WSJ0_base}"
      log "Failed to extract WSJ0"
      exit 1
    }
fi

# Get WSJ1 raw data from /export/share if needed
if [ -z "${WSJ1}" ]; then
    log "Fill the value of 'WSJ1' of db.sh"
    exit 1
elif [ ! -e "${WSJ1}" ]; then
    WSJ1_base=$(dirname "${WSJ1}")
    mkdir -p "${WSJ1_base}"
    {
      tar xzvf /export/share/ybzhou/dataset/LDC/csr_senn_LDC94S13B.tgz -C "${WSJ1_base}"
    } || {
      rm -rf "${WSJ1_base}"
      log "Failed to extract WSJ1"
      exit 1
    }
fi

log "local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?"
local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?
log "local/wsj_format_data.sh"
local/wsj_format_data.sh

log "Create the list of non-linguistic symbols: ${nlsyms}"
cut -f 2- -d" " data/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
cat ${nlsyms}

log "Prepare text from lng_modl dir: ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z -> ${other_text}"
mkdir -p "$(dirname ${other_text})"

# NOTE(kamo): Give utterance id to each texts & make everything lowercase
# Also remove utterances with non-linguistic symbols, i.e. lines including "<"
zcat ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
    grep -v "<" | tr "[:upper:]" "[:lower:]" | \
    awk '{ printf("{wsj}lng_%07d %s\n",NR,$0) } ' > ${other_text}
