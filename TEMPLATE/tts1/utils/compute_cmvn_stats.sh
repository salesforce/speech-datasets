#!/bin/bash
set -euo pipefail

# Begin configuration section.
nj=4
cmd=utils/run.pl
archive_format=hdf5
cmvn_type=global
spk2utt=
# End configuration section.

help_message=$(cat << EOF
Usage: $0 [options] <input-scp> <output-ark> [logdir]
e.g.: $0 data/train/feats.scp data/train/shape.scp data/train/logs
Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --archive_format <mat|hdf5>                      # Specify the format of feats file
  --cmvn-type                                      # cmvn_type (global or speaker or utterance)
  --spk2utt                                        # speaker -> utterance file
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
cmvnark=$2
data=$(dirname ${scp})
if [ $# -eq 3 ]; then
  logdir=$3
else
  logdir=${data}/logs
fi
mkdir -p ${logdir}

split_scps=
split_cmvn=
for n in $(seq ${nj}); do
    split_cmvn+="${logdir}/cmvn.${n}.ark "
    split_scps+="${logdir}/feats.${n}.scp "
done
utils/split_scp.pl ${scp} ${split_scps} || exit 1


maybe_spk2utt=
if [ -n "${spk2utt}" ] && [ "${cmvn_type}" = speaker ]; then
    maybe_spk2utt="--spk2utt ${spk2utt}"
fi

${cmd} JOB=1:${nj} ${logdir}/compute_cmvn_stats.JOB.log \
    compute_cmvn_stats.py --filetype ${archive_format} ${maybe_spk2utt} \
    --cmvn-type ${cmvn_type} "scp:${logdir}/feats.JOB.scp" "${logdir}/cmvn.JOB.ark"

python3 -m speech_datasets.bin.combine_cmvn_stats --cmvn_type ${cmvn_type} \
    --output_file ${cmvnark} ${split_cmvn} || exit 1

rm -f ${split_scps} ${split_cmvn}
