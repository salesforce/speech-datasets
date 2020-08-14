#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Copyright   2014  Johns Hopkins University (author: Daniel Povey)
#             2017  Luminar Technologies, Inc. (author: Daniel Galvez)
#             2017  Ewald Enzinger
# Apache 2.0

# Adapted from egs/mini_librispeech/s5/local/download_and_untar.sh (commit 1cd6d2ac3a935009fdc4184cb8a72ddad98fe7d9)

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 3 ]; then
  log "Usage: $0 [--remove-archive] <data-base> <url> <filename>"
  log "e.g.: $0 /export/data/ https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz cv_corpus_v1.tar.gz"
  log "With --remove-archive it will remove the archive after successfully un-tarring it."
  exit 1
fi

data=$1
url=$2
filename=$3
filepath="$data/$filename"
workspace=$PWD

if [ ! -d "$data" ]; then
  log "$0: no such directory $data"
  exit 1;
fi

if [ -z "$url" ]; then
  log "$0: empty URL."
  exit 1;
fi

if [ -f $data/$filename.complete ]; then
  log "$0: data was already successfully extracted, nothing to do."
  exit 0;
fi


if [ -f $filepath ]; then
  size=$(/bin/ls -l $filepath | awk '{print $5}')
  size_ok=false
  if [ "$filesize" -eq "$size" ]; then size_ok=true; fi;
  if ! $size_ok; then
    log "$0: removing existing file $filepath because its size in bytes ($size)"
    log "does not equal the size of the archives ($filesize)."
    rm $filepath
  else
    log "$filepath exists and appears to be complete."
  fi
fi

if [ ! -f $filepath ]; then
  if ! which wget >/dev/null; then
    log "$0: wget is not installed."
    exit 1;
  fi
  log "$0: downloading data from $url.  This may take some time, please be patient."

  cd $data
  if ! wget --no-check-certificate $url; then
    log "$0: error executing wget $url"
    exit 1;
  fi
  cd $workspace
fi

cd $data

if ! tar -xzf $filename; then
  log "$0: error un-tarring archive $filepath"
  exit 1;
fi

cd $workspace

touch $data/$filename.complete

log "$0: Successfully downloaded and un-tarred $filepath"

if $remove_archive; then
  log "$0: removing $filepath file since --remove-archive option was supplied."
  rm $filepath
fi
