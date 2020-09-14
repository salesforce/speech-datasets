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


# Extract & prepare Fisher
for (( i=1; i<=$(echo "${FISHER_TGZ}" | wc -w); i++ )); do
    src=$(echo "${FISHER_TGZ}" | cut -d " " -f $i)
    dst=$(echo "${FISHER}" | cut -d " " -f $i)
    if [ ! -e "${dst}" ]; then
        mkdir -p "${dst}"
        {
            tar xzvf "${src}" -C "${dst}"
        } || {
            log "Failed to extract FISHER (part $i)"
            exit 1
        }
    fi
done

# Note: do not quote ${FISHER} -- it should contains 4 directories, and fisher_prep.sh all 4
log "local/fisher_data_prep.sh ${FISHER}"
local/fisher_data_prep.sh ${FISHER}