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

# all utterances are FLAC compressed
if ! which flac >&/dev/null; then
   log "Please install 'flac' on ALL worker nodes!"
   exit 1
fi

if [ "$#" -eq 0 ]; then
    log "Usage: $0 <dataset1-name> <dataset2-name> ..."
    log "e.g.: $0 train-clean-100 train-clean-360 dev-clean test-clean"
    exit 1
fi

# Ensure that LIBRISPEECH data has already been extracted
if [ -z "${LIBRISPEECH}" ]; then
    log "Fill in the value of 'LIBRISPEECH' in db.sh"
    exit 1
fi
mkdir -p "${LIBRISPEECH}"
base_url=www.openslr.org/resources/12


spk_file="${LIBRISPEECH}/LibriSpeech/SPEAKERS.TXT"
[ ! -f "$spk_file" ] && log "$0: expected file $spk_file to exist" && exit 1

for dset in "$@"; do
    log "Downloading dataset ${dset} (if needed)"
    local/download_and_untar.sh "${LIBRISPEECH}" "${base_url}" "${dset}"

    log "Preparing dataset ${dset}"
    src=$LIBRISPEECH/LibriSpeech/$dset
    dst=data/$dset
    mkdir -p "$dst" || exit 1

    [ ! -d "$src" ] && log "$0: no such directory $src" && exit 1

    # Remove existing files
    wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm "$wav_scp"
    trans=$dst/text; [[ -f "$trans" ]] && rm $trans
    utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm "$utt2spk"
    spk2gender=$dst/spk2gender; [[ -f "$spk2gender" ]] && rm "$spk2gender"

    for reader_dir in $(find -L "${src}" -mindepth 1 -maxdepth 1 -type d | sort); do
        reader=$(basename "${reader_dir}")
        if ! [ "${reader}" -eq "${reader}" ]; then  # not integer.
            log "$0: unexpected subdirectory name $reader" && exit 1
        fi

        reader_gender=$(egrep "^${reader}[ ]+\|" ${spk_file} | awk -F'|' '{gsub(/[ ]+/, ""); print tolower($2)}')
        if [ "$reader_gender" != 'm' ] && [ "${reader_gender}" != 'f' ]; then
            log "Unexpected gender: '${reader_gender}'" && exit 1
        fi

        for chapter_dir in $(find -L $reader_dir/ -mindepth 1 -maxdepth 1 -type d | sort); do
          chapter=$(basename $chapter_dir)
          if ! [ "$chapter" -eq "$chapter" ]; then
              log "$0: unexpected chapter-subdirectory name $chapter" && exit 1
          fi

          find -L $chapter_dir/ -iname "*.flac" | sort | xargs -I% basename % .flac | \
              awk -v "dir=$chapter_dir" '{printf "{librispeech}%s flac -c -d -s %s/%s.flac |\n", $0, dir, $0}' \
               >> $wav_scp|| exit 1

          chapter_trans=$chapter_dir/${reader}-${chapter}.trans.txt
          [ ! -f  $chapter_trans ] && log "$0: expected file $chapter_trans to exist" && exit 1
          cat $chapter_trans | tr "[:upper:]" "[:lower:]" | sed -E "s/^/\{librispeech\}/" >> $trans

          # NOTE: For now we are using per-chapter utt2spk. That is each chapter is considered
          #       to be a different speaker. This is done for simplicity and because we want
          #       e.g. the CMVN to be calculated per-chapter
          awk -v "reader={librispeech}$reader" -v "chapter=$chapter" \
            '{printf "{librispeech}%s %s-%s\n", $1, reader, chapter}' < $chapter_trans >> $utt2spk || exit 1

          # reader -> gender map (again using per-chapter granularity)
          echo "{librispeech}${reader}-${chapter} $reader_gender" >> $spk2gender
        done
    done

    spk2utt=$dst/spk2utt
    utils/utt2spk_to_spk2utt.pl $utt2spk > $spk2utt || exit 1

    ntrans=$(wc -l <$trans)
    nutt2spk=$(wc -l <$utt2spk)
    ! [ "$ntrans" -eq "$nutt2spk" ] && \
      log "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1

    log "$(utils/validate_data_dir.sh --no-feats "${dst}")" || exit 1
    log "$0: Successfully prepared dataset $dset"
done

exit 0
