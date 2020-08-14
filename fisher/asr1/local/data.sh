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
if [ "$(echo "${FISHER}" | wc -w)" != 4 ]; then
    log "Fill the value of 'FISHER' in db.sh (4 items required; last 2 should be pre-extracted)"
fi
fisher_sources="/export/share/ybzhou/dataset/LDC/LDC2004T19/fe_03_p1_tran_LDC2004T19.tgz "
fisher_sources+="/export/share/ybzhou/dataset/LDC/LDC2005T19/LDC2005T19.tgz"
for (( i=1; i<=2; i++ )); do
    src=$(echo "${fisher_sources}" | cut -d " " -f $i)
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