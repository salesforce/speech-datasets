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

# Extract WSJ0/WSJ1 raw data if needed
WSJ=("${WSJ0}" "${WSJ1}")
WSJ_TGZ=("${WSJ0_TGZ}" "${WSJ1_TGZ}")
for (( i=0; i<2; i++ )); do
    echo ${WSJ[i]}
    if [ -z "${WSJ[i]}" ]; then
        log "Fill the value of 'WSJ${i}' in db.sh"
        exit 1
    elif [ ! -d "${WSJ[i]}" ]; then
        mkdir -p "${WSJ[i]}"
        {
          tar xzvf "${WSJ_TGZ[i]}" -C "${WSJ[i]}"
        } || {
          rm -rf "${WSJ[i]}"
          log "Failed to extract WSJ${i}"
          exit 1
        }
    fi
done

log "local/wsj_data_prep.sh ${WSJ0}/csr_1_senn/??-{?,??}.? ${WSJ1}/csr_senn/??-{?,??}.?"
local/wsj_data_prep.sh "${WSJ0}"/csr_1_senn/??-{?,??}.? "${WSJ1}"/csr_senn/??-{?,??}.?
log "local/wsj_format_data.sh"
local/wsj_format_data.sh

log "Create the list of non-linguistic symbols: ${nlsyms}"
cut -f 2- -d" " data/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
cat ${nlsyms}

log "Prepare text from lng_modl dir: ${WSJ1}/csr_senn/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z -> ${other_text}"
mkdir -p "$(dirname ${other_text})"

# NOTE(kamo): Give utterance id to each texts & make everything lowercase
# Also remove utterances with non-linguistic symbols, i.e. lines including "<"
zcat ${WSJ1}/csr_senn/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
    grep -v "<" | tr "[:upper:]" "[:lower:]" | \
    awk '{ printf("{wsj}lng_%07d %s\n",NR,$0) } ' > ${other_text}
