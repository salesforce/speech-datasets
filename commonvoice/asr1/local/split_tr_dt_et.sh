#!/bin/bash

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

perdt=10 # percent for dev set
peret=10 # percent for eval set

. utils/parse_options.sh

if [ $# != 4 ]; then
    log "Usage: $0 <src-data-dir> <dest-trdata-dir> <dest-dtdata-dir> <dest-etdata-dir>";
    exit 1;
fi

sdata=$1
trdata=$2
dtdata=$3
etdata=$4

tmpdata=$trdata/tmp
mkdir -p $tmpdata
mkdir -p $dtdata
mkdir -p $etdata

# make a unique prompts files
# some transcripts have multiple spaces and need tr -s " " to remove them
cut -f 2- -d" " $sdata/text | tr -s " " | sort | uniq > $tmpdata/prompts
num_prompt=$(wc -l $tmpdata/prompts | awk '{print $1}')

num_dt=$(echo "$num_prompt * $perdt / 100" | bc)
num_et=$(echo "$num_prompt * $peret / 100" | bc)
log "number of dev set prompts: $num_dt"
log "number of eval set prompts: $num_et"

# dt
utils/shuffle_list.pl $tmpdata/prompts | head -n $num_dt > $tmpdata/dt_prompts
# et
utils/shuffle_list.pl $tmpdata/prompts | head -n $(echo "$num_dt + $num_et" | bc) \
    | tail -n $num_et > $tmpdata/et_prompts
# tr
nrest=$(echo "$num_dt + $num_et + 1" | bc)
utils/shuffle_list.pl $tmpdata/prompts | tail -n +$nrest > $tmpdata/tr_prompts
log "number of train set prompts: $(wc -l $tmpdata/tr_prompts | awk '{print $1}')"

# it takes very long time when # prompts is large
cat $sdata/text | local/filter_text.py -f $tmpdata/dt_prompts | awk '{print $1}' | sort > $tmpdata/dt.ids
log "finished text extraction for dev set #utt = $(wc -l $tmpdata/dt.ids | awk '{print $1}')"
cat $sdata/text | local/filter_text.py -f $tmpdata/et_prompts | awk '{print $1}' | sort > $tmpdata/et.ids
log "finished text extraction for eval set #utt = $(wc -l $tmpdata/et.ids | awk '{print $1}')"
cat $tmpdata/dt.ids $tmpdata/et.ids | sort > $tmpdata/dtet.ids
cat $sdata/text | awk '{print $1}' | sort > $tmpdata/all.ids
diff $tmpdata/all.ids $tmpdata/dtet.ids | awk '/^</{print $2}' | sort > $tmpdata/tr.ids
log "finished text extraction for train set #utt = $(wc -l $tmpdata/tr.ids | awk '{print $1}')"

log "dev data: $(reduce_data_dir.sh $sdata $tmpdata/dt.ids $dtdata)"
utils/fix_data_dir.sh $dtdata

log "eval data: $(reduce_data_dir.sh $sdata $tmpdata/et.ids $etdata)"
utils/fix_data_dir.sh $etdata

log "train data: $(reduce_data_dir.sh $sdata $tmpdata/tr.ids $trdata)"
utils/fix_data_dir.sh $trdata
