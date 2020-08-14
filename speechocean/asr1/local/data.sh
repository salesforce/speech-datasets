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


# Ensure that SPEECHOCEAN data has already been extracted
if [ -z "${SPEECHOCEAN_053}" ]; then
    log "Please fill the value of 'SPEECHOCEAN_053' in db.sh"
    exit 1
elif [ ! -e "${SPEECHOCEAN_053}" ]; then
    log "Make sure speechocean/King-ASR-053 has been properly extracted in ${SPEECHOCEAN_053}"
    exit 1
fi
mkdir -p data/053_train data/053_dev data/053_test
if [ -z "${SPEECHOCEAN_066}" ]; then
    log "Please fill the value of 'SPEECHOCEAN_066' in db.sh"
    exit 1
elif [ ! -e "${SPEECHOCEAN_066}" ]; then
    log "Make sure speechocean/King-ASR-066 has been properly extracted in ${SPEECHOCEAN_066}"
    exit 1
fi
mkdir -p data/066_train data/066_dev data/066_test

# Run the processing for each dataset
dsets="053 066"
splits="test dev train"
for dset in ${dsets}; do
    src=$(eval echo \$SPEECHOCEAN_${dset})
    if [ -z "${src}" ]; then
        log "Fill in the value of 'SPEECHOCEAN_${dset}' in db.sh"
    fi
    if [ ! -e "${src}" ]; then
        log "Make sure that speechocean/King-ASR-${dset} has been properly extracted in ${src}"
    fi

    for split in ${splits}; do
        log "Processing ${split} split of speechocean/King-ASR-${dset}..."
        dst="data/${dset}_${split}"
        prefix="{speechocean-${dset}}"
        spk_list="${src}/spk_${split}"
        python local/process_speechocean.py --srcdir "${src}" --dstdir "${dst}" \
            --data_prefix "${prefix}" --speaker_list "${spk_list}"
        log "Finished processing ${split} split of speechocean/King-ASR-${dset}"
    done

done

# Combine the datasets
for split in ${splits}; do
    log "Combining ${split} splits of all SpeechOcean datasets..."
    dirs=
    for dset in ${dsets}; do
        dirs+="data/${dset}_${split} "
    done
    utils/combine_data.sh "data/${split}" ${dirs}
    utils/fix_data_dir.sh "data/${split}"
    rm -rf ${dirs}
done

log "Finished processing SpeechOcean data"
