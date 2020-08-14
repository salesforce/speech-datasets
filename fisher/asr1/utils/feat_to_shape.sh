#!/bin/bash
set -euo pipefail

# Begin configuration section.
nj=4
cmd=utils/run.pl
verbose=0
archive_format=
preprocess_conf=
# End configuration section.

help_message=$(cat << EOF
Usage: $0 [options] <input-scp> <output-scp> [<log-dir>]
e.g.: $0 data/train/feats.scp data/train/shape.scp data/train/logs
Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --archive_format <mat|hdf5|sound>                # Specify the format of feats file
  --preprocess-conf <json>                         # Apply preprocess to feats when creating shape.scp
  --verbose <num>                                  # Default: 0
EOF
)

echo "$0 $*" 1>&2 # Print the command line for logging
. path.sh || exit 1
. utils/parse_options.sh || exit 1;

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

scp=$1
outscp=$2
data=$(dirname ${scp})
if [ $# -eq 3 ]; then
  logdir=$3
else
  logdir=${data}/logs
fi
mkdir -p ${logdir}

nj=$((nj<$(<"${scp}" wc -l)?nj:$(<"${scp}" wc -l)))
split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/feats.${n}.scp"
done

utils/split_scp.pl ${scp} ${split_scps}

if [ -n "${preprocess_conf}" ]; then
    preprocess_opt="--preprocess-conf ${preprocess_conf}"
else
    preprocess_opt=""
fi
if [ -n "${archive_format}" ]; then
    filetype_opt="--filetype ${archive_format}"
else
    filetype_opt=""
fi

${cmd} JOB=1:${nj} ${logdir}/feat_to_shape.JOB.log \
    feat_to_shape.py --verbose ${verbose} ${preprocess_opt} ${filetype_opt} \
    scp:${logdir}/feats.JOB.scp ${logdir}/shape.JOB.scp

# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat ${logdir}/shape.${n}.scp
done > ${outscp}

rm -f ${logdir}/feats.*.scp 2>/dev/null
