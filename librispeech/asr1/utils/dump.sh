#!/bin/bash
set -euo pipefail
SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Begin configuration section.
nj=$(nproc)
conf=none
data_name=
data_format=
fs=16000
cmd=utils/run.pl
compress=true
archive_format=hdf5 # mat or hdf5
text=
utt2spk=
segments=
# End configuration section.

help_message=$(cat <<EOF
Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
e.g.: $0 data/train exp/make_fbank/train mfcc
Note: <log-dir> defaults to <data-dir>/logs, and <fbank-dir> defaults to <data-dir>/data
Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --archive_format <mat|hdf5>                      # Specify the format of feats file
EOF
)
log "$0 $*"  # Print the command line for logging
. path.sh || exit 1
. utils/parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 4 ]; then
    log "${help_message}"
    exit 1;
fi

scp=$1
data=$(utils/make_absolute.sh $2)
if [ $# -ge 3 ]; then
  logdir=$3
else
  logdir=${data}/logs
fi
if [ $# -ge 4 ]; then
  dumpdir=$(utils/make_absolute.sh $4)
else
  dumpdir=${data}/data
fi

# use "data_name" as part of data_name of the archive.
if [ -z ${data_name} ]; then
    log "Please provide a data_name for the features being extracted by supplying the --data-name option"
    exit 2
fi

mkdir -p ${dumpdir} || exit 1;
mkdir -p ${logdir} || exit 1;

if [ -f ${data}/${data_format}.scp ]; then
  mkdir -p ${data}/.backup
  log "$0: moving $data/${data_format}.scp to $data/.backup"
  mv ${data}/${data_format}.scp ${data}/.backup
fi

utils/validate_data_dir.sh --no-text --no-feats --no-scp ${data} || exit 1;

split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav.${n}.scp"
done

utils/split_scp.pl ${scp} ${split_scps} || exit 1;

extra_opt=
if [ -n "${text}" ]; then
    extra_opt+="--text-file ${text} "
fi
if [ -n "${utt2spk}" ]; then
    extra_opt+="--utt2spk-file ${utt2spk} "
fi

if [ "${archive_format}" == hdf5 ]; then
    ext=h5
else
    ext=ark
fi

if [ -n "${segments}" ] && [ -f "${segments}" ]; then
    log "$0 [info]: segments file exists: using that."
    split_segments=""
    for n in $(seq ${nj}); do
        split_segments="${split_segments} ${logdir}/segments.${n}"
    done

    utils/split_scp.pl ${segments} ${split_segments}
    for split in ${split_segments}; do
        sort -k1,1 -u < ${split} > ${split}.tmp
        mv ${split}.tmp  ${split}
    done

    ${cmd} JOB=1:${nj} ${logdir}/make_${data_name}.JOB.log \
        dump.py \
            --feature-config ${conf} ${extra_opt} --sample-frequency ${fs} \
            --compress=${compress} --archive-format ${archive_format} \
            --segment=${logdir}/segments.JOB scp:${scp} \
            ark,scp:${dumpdir}/${data_name}.JOB.${ext},${dumpdir}/${data_name}.JOB.scp

else
    log "$0: [info]: no segments file exists: assuming ${scp} indexed by utterance."
    split_scps=""
    for n in $(seq ${nj}); do
      split_scps="${split_scps} ${logdir}/wav.${n}.scp"
    done

    utils/split_scp.pl ${scp} ${split_scps}
    for split in ${split_scps}; do
        sort -k1,1 -u < ${split} > ${split}.tmp
        mv ${split}.tmp  ${split}
    done

    ${cmd} JOB=1:${nj} ${logdir}/make_${data_name}.JOB.log \
        dump.py \
            --feature-config ${conf} ${extra_opt} --sample-frequency ${fs} \
            --compress=${compress} --archive-format ${archive_format} \
            scp:${logdir}/wav.JOB.scp \
            ark,scp:${dumpdir}/${data_name}.JOB.${ext},${dumpdir}/${data_name}.JOB.scp
fi

# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat ${dumpdir}/${data_name}.${n}.scp || exit 1;
done > ${data}/${data_format}.scp || exit 1

rm -f ${logdir}/wav.*.scp ${logdir}/segments.* 2>/dev/null

n_feat=$(wc -l < ${data}/${data_format}.scp)
n_utt=$(wc -l < ${data}/utt2spk)
if [ ${n_feat} -ne ${n_utt} ]; then
    log "It seems not all of the feature files were successfully ($n_feat != $n_utt);"
    log "consider using utils/fix_data_dir.sh $data"
fi

log "Succeeded dumping ${data_name} [elapsed=${SECONDS}s]"
