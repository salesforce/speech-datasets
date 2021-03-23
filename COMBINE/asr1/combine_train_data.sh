#!/bin/bash
set -euo pipefail
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message="Usage: $0 [asr.sh options] <dataset1>/<split1a> <dataset2>/<split2a> <dataset2>/<split2b> ..."

log "$0 $*"
if [ $# -eq 0 ]; then
    log "$help_message"
    log "Error: at least 1 argument required"
    exit 2
fi

kwargs=()
stage=2
stop_stage=5
while true; do
    case "$1" in
        --stage)
            if [ "$2" -lt 2 ]; then
                log "Specify --stage 2 or higher (got --stage $2)."
                log "We expect stage 1 to be complete for all datasets given."
                exit 2
            else
                stage=$2
            fi
            shift 2
            ;;
        --stop-stage|--stop_stage)
            if [ "$2" -gt 5 ]; then
                log "Specify --stop-stage 5 or lower (got --stop-stage $2)."
                log "Use combine_cmvn_stats.sh to combine CMVN statistics from multiple datasets (stage 5)."
                log "Use multi_tokenize.sh to obtain token inventories from multiple datasets (stages 6-7)."
                exit 2
            else
                stop_stage=$2
            fi
            shift 2
            ;;
        --*) kwargs+=( "$1" "$2" ); shift 2; ;;
        *) break;
    esac
done
kwargs+=( --stage "$stage" --stop_stage "$stop_stage" )

if [ $# -eq 0 ]; then
    log "${help_message}"
    log "Error: Please specify dataset splits as positional arguments."
    exit 2
fi

task=$(basename "$(utils/make_absolute.sh "$PWD")")
idx=$(python local/combine_datasets.py --task "${task//1/}" --write_dir true "$@")
datadir="data/${idx}"
for f in wav.scp segments utt2spk text; do
    sort "${datadir}/${f}" > "${datadir}/${f}.tmp"
    mv "${datadir}/${f}.tmp" "${datadir}/${f}"
done
./run.sh "${kwargs[@]}" --train_sets "${idx}"
