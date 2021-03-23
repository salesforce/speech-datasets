#!/bin/bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./path.sh || exit 1
. ./db.sh || exit 1

# Extract switchboard-1
if [ -z "${SWBD1}" ]; then
    log "Fill the value of 'SWBD1' in db.sh"
    exit 1
elif [ ! -e "${SWBD1}" ]; then
    mkdir -p "${SWBD1}"
    {
      tar xzvf ${SWBD1_TGZ} -C "${SWBD1}"
    } || {
      log "Failed to extract SWBD1"
      exit 1
    }
fi

# Download switchboard-1 transcripts if needed
if [ ! -d "${SWBD1}/swb_ms98_transcriptions" ]; then
    echo " *** Downloading trascriptions and dictionary ***"
    wget http://www.openslr.org/resources/5/switchboard_word_alignments.tar.gz ||
    wget http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
    tar xzvf switchboard_word_alignments.tar.gz -C "${SWBD1}"
    rm switchboard_word_alignments.tar.gz
else
  log "Directory with transcriptions exists, skipping downloading."
fi

# Prepare the dictionary & the rest of the Switchboard-1 data
log "local/swbd1_prepare_dict.sh ${SWBD1}"
local/swbd1_prepare_dict.sh "${SWBD1}"
log "local/swbd1_data_prep.sh ${SWBD1}"
local/swbd1_data_prep.sh "${SWBD1}"

# Extract & prepare EVAL-2000
if [ "$(echo "${EVAL2000}" | wc -w)" != 2 ]; then
    log "Fill the value of 'EVAL2000' in db.sh (2 items required, hub5e_00 and hub5)"
fi
for (( i=1; i<=2; i++ )); do
  src=$(echo "${EVAL2000_TGZ}" | cut -d " " -f $i)
  dst=$(echo "${EVAL2000}" | cut -d " " -f $i)
  # hub5e is in a sub-directory
  if [ $i = 1 ]; then
    dst=$(dirname "${dst}")
  fi

  if [ ! -e "${dst}" ]; then
      mkdir -p "${dst}"
      {
        tar xzvf "${src}" -C "${dst}"
      } || {
        log "Failed to extract EVAL2000 (part $i)"
        exit 1
      }
  fi
done

# Note: do not quote ${EVAL2000} -- it should contains 2 directories, and eval2000_data_prep.sh requires 2 arguments
log "local/eval2000_data_prep.sh ${EVAL2000}"
local/eval2000_data_prep.sh ${EVAL2000}

# Extract & prepare RT-03
if [ -z "${RT03}" ]; then
    log "Fill the value of 'RT03' in db.sh"
    exit 1
elif [ ! -e "${RT03}" ]; then
    RT03_BASE="$(dirname "${RT03}")"
    mkdir -p "${RT03_BASE}"
    {
      tar xzvf "${RT03_TGZ}" -C "${RT03_BASE}"
    } || {
      log "Failed to extract SWBD1"
      exit 1
    }
fi

log "local/rt03_data_prep.sh ${RT03}"
local/rt03_data_prep.sh ${RT03}

# normalize eval2000 and rt03 texts by
# 1) convert upper to lower
# 2) remove tags (%AH) (%HESITATION) (%UH)
# 3) remove <B_ASIDE> <E_ASIDE>
# 4) remove "(" or ")"
for x in eval2000 rt03; do
    cp data/${x}/text data/${x}/text.org
    paste -d "" \
        <(cut -f 1 -d" " data/${x}/text.org) \
        <(awk '{$1=""; print tolower($0)}' data/${x}/text.org | perl -pe 's| \(\%.*\)||g' \
         | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g") | sed -e 's/\s\+/ /g' > data/${x}/text
    rm data/${x}/text.org
done
