#!/bin/bash
set -euo pipefail
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Copyright 2017 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo "$0 $*"  # Print the command line for logging

cmd=utils/run.pl
nj=$(nproc)
filetype='hdf5'  # mat or hdf5
cmvn_type='global' # global or utterance or speaker
help_message="Usage: $0 <scp> <logdir> <dumpdir>"

. ./path.sh || exit 1
. utils/parse_options.sh || exit 1

if [ $# != 3 ]; then
    log "${help_message}"
    exit 2
fi

scp=$1
logdir=$2
dumpdir=$(utils/make_absolute.sh $3)

if [ ${filetype} = mat ]; then
    ext=ark
elif [ ${filetype} = hdf5 ]; then
    ext=h5
else
    log "Received --filetype '${filetype}', but only 'mat' and 'hdf5' are valid"
    exit 2
fi

if [ ${cmvn_type} != global ] && [ ${cmvn_type} != utterance ] && [ ${cmvn_type} != speaker ]; then
    log "Received --cmvn_type '${cmvn_type}', but only 'global', 'utterance', 'speaker'' are valid"
fi

srcdir=$(dirname "$scp")
cmvnark=$srcdir/cmvn.ark
maybe_utt2spk=
if [ -f $srcdir/utt2spk ]; then
    maybe_utt2spk+="--utt2spk $srcdir/utt2spk "
fi
maybe_spk2utt=
if [ -f $srcdir/spk2utt ]; then
    maybe_spk2utt+="--spk2utt $srcdir/spk2utt "
fi

mkdir -p ${logdir}
mkdir -p ${dumpdir}

# compute CMVN stats
python -m speech_datasets.bin.compute_cmvn_stats \
    --in-filetype ${filetype} ${maybe_spk2utt}  \
    --cmvn-type ${cmvn_type} scp:${scp} ${cmvnark}

echo $cmvn_type > $srcdir/cmvn_type

# split scp file
split_scps=""
for n in $(seq ${nj}); do
    split_scps="$split_scps $logdir/feats.$n.scp"
done

utils/split_scp.pl ${scp} ${split_scps} || exit 1;

# apply CMVN to features & dump them
${cmd} JOB=1:${nj} ${logdir}/apply_cmvn.JOB.log \
    apply_cmvn.py --norm-vars true --in-filetype ${filetype} --out-filetype ${filetype} \
        --cmvn-type ${cmvn_type} ${maybe_utt2spk} ${cmvnark} scp:${logdir}/feats.JOB.scp \
        ark,scp:${dumpdir}/feats.JOB.${ext},${dumpdir}/feats.JOB.scp \
    || exit 1

# concatenate scp files
for n in $(seq ${nj}); do
    cat ${dumpdir}/feats.${n}.scp || exit 1;
done > ${dumpdir}/feats.scp || exit 1

# remove temp scps
rm ${dumpdir}/feats.*.scp 2>/dev/null
rm ${logdir}/feats.*.scp 2>/dev/null
log "Succeeded applying CMVN to features for training."
