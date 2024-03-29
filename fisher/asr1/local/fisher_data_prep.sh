#!/bin/bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Copyright 2013  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

stage=0
. utils/parse_options.sh || exit 1

if [ $# -eq 0 ]; then
  log "$0 <fisher-dir-1> [<fisher-dir-2> ...]"
  log " e.g.: $0 /export/corpora3/LDC/LDC2004T19 /export/corpora3/LDC/LDC2005T19\\"
  log " /export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13"
  log " (We also support a single directory that has the contents of all of them)"
  exit 1;
fi


# Check that the arguments are all absolute pathnames.
for dir in $*; do
  case $dir in /*) ;; *)
      log "$0: all arguments must be absolute pathnames."; exit 1;
  esac
done

# First check we have the right things in there...
tmpdir=`pwd`/data/local
links=$tmpdir/links
rm -rf $links 2>/dev/null
mkdir -p $links || exit 1;
for subdir in fe_03_p1_sph1  fe_03_p1_sph3  fe_03_p1_sph5  fe_03_p1_sph7 \
  fe_03_p2_sph1  fe_03_p2_sph3  fe_03_p2_sph5  fe_03_p2_sph7 fe_03_p1_sph2 \
  fe_03_p1_sph4  fe_03_p1_sph6  fe_03_p1_tran  fe_03_p2_sph2  fe_03_p2_sph4 \
  fe_03_p2_sph6  fe_03_p2_tran; do
  found_subdir=false
  for dir in $*; do
    if [ -d $dir/$subdir ]; then
      found_subdir=true
      ln -s $dir/$subdir $links/$subdir
    else
      new_style_subdir=$(echo $subdir | sed s/fe_03_p1_sph/fisher_eng_tr_sp_d/)
      if [ -d $dir/$new_style_subdir ]; then
        found_subdir=true
        ln -s $dir/$new_style_subdir $links/$subdir
      fi
    fi
  done
  if ! $found_subdir; then
    log "$0: could not find the subdirectory $subdir in any of $*"
    exit 1;
  fi
done
log "Verified all subdirectories"

. ./path.sh # Needed for MAIN_ROOT
sph2pipe=$MAIN_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   log "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

# (1) Get transcripts in one file, and clean them up ...

if [ $stage -le 0 ]; then

  find $links/fe_03_p1_tran/data $links/fe_03_p2_tran/data -iname '*.txt'  > $tmpdir/transcripts.flist

  for dir in fe_03_p{1,2}_sph{1,2,3,4,5,6,7}; do
    find $links/$dir/ -iname '*.sph'
  done > $tmpdir/sph.flist

  n=`cat $tmpdir/transcripts.flist | wc -l`
  if [ $n -ne 11699 ]; then
    log "Expected to find 11699 transcript files in the Fisher data, found $n"
    exit 1;
  fi
  n=`cat $tmpdir/sph.flist | wc -l`
  if [ $n -ne 11699 ]; then
    log "Expected to find 11699 .sph files in the Fisher data, found $n"
    exit 1;
  fi
  log "Found all transcripts & .sph files."
fi

dir=data/train_fisher
if [ $stage -le 1 ]; then
  mkdir -p $dir

  ## fe_03_00004.sph
  ## Transcpribed at the LDC
  #
  #7.38 8.78 A: an- so the topic is

  echo -n > $tmpdir/text.1 || exit 1;

  perl -e ' 
   use File::Basename;
   ($tmpdir)=@ARGV;
   open(F, "<$tmpdir/transcripts.flist") || die "Opening list of transcripts";
   open(R, "|sort >data/train_fisher/reco2file_and_channel") || die "Opening reco2file_and_channel";
   open(T, ">$tmpdir/text.1") || die "Opening text output";
   while (<F>) {
     $file = $_;
     m:([^/]+)\.txt: || die "Bad filename $_";
     $call_id = $1;
     $mod_call_id = $1 =~ s/^fe_03_/{fisher}/r;
     print R "$mod_call_id-A $mod_call_id A\n";
     print R "$mod_call_id-B $mod_call_id B\n";
     open(I, "<$file") || die "Opening file $_";

     $line1 = <I>;
     $line1 =~ m/# (.+)\.sph/ || die "Bad first line $line1 in file $file";
     $call_id eq $1 || die "Mismatch call-id $call_id vs $1\n";
     while (<I>) {
       if (m/([0-9.]+)\s+([0-9.]+) ([AB]):\s*(\S.+\S|\S)\s*$/) {
         $start = sprintf("%06d", $1 * 100.0);
         $end = sprintf("%06d", $2 * 100.0);
         length($end) > 6 && die "Time too long $end in file $file";
         $side = $3; 
         $words = $4;
         $utt_id = "${mod_call_id}-$side-$start-$end";
         print T "$utt_id $words\n" || die "Error writing to text file";
       }
     }
   }
   close(R); close(T) ' $tmpdir || exit 1;
   log "Consolidated all transcripts into $tmpdir/text.1"
fi

if [ $stage -le 2 ]; then
  sort $tmpdir/text.1 | grep -v '((' | \
    awk '{if (NF > 1){ print; }}' | \
    sed 's:\[laugh\]:<noise>:g' | \
    sed 's:\[laughter\]:<noise>:g' | \
    sed 's:\[sigh\]:<noise>:g' | \
    sed 's:\[cough\]:<noise>:g' | \
    sed 's:\[sneeze\]:<noise>:g' | \
    sed 's:\[sigh\]:<noise>:g' | \
    sed 's:\[mn\]:<noise>:g' | \
    sed 's:\[breath\]:<noise>:g' | \
    sed 's:\[lipsmack\]:<noise>:g' | \
    sed 's:\[noise\]:<noise>:g' | \
    sed 's:\[\[skip\]\]::g' | \
    sed 's:\[pause\]::g' > $tmpdir/text.2
  cp $tmpdir/text.2 $dir/text
  log "Standardized transcriptions for different kinds of noises to get $tmpdir/text"

  # create segments file and utt2spk file...
  ! cat $dir/text | perl -ane 'm:([^-]+)-([AB])-(\S+): || die "Bad line $_;"; print "$1-$2-$3 $1-$2\n"; ' > $dir/utt2spk  \
     && log "Error producing utt2spk file" && exit 1;

  cat $dir/text | perl -ane 'm:((\S+-[AB])-(\d+)-(\d+))\s: || die; $utt = $1; $reco = $2; $s = sprintf("%.2f", 0.01*$3);
                 $e = sprintf("%.2f", 0.01*$4); print "$utt $reco $s $e\n"; ' > $dir/segments

  utils/utt2spk_to_spk2utt.pl <$dir/utt2spk > $dir/spk2utt
  log "Created utt2spk, spk2utt, and segments files"
fi

if [ $stage -le 3 ]; then
  for f in `cat $tmpdir/sph.flist`; do
    # convert to absolute path
    utils/make_absolute.sh $f
  done > $tmpdir/sph_abs.flist
  
  cat $tmpdir/sph_abs.flist | perl -ane 'm:/([^/]+)\.sph$: || die "bad line $_; ";
                                         $x = $1 =~ s/^fe_03_/{fisher}/r;
                                         print "$x $_"; ' > $tmpdir/sph.scp

  cat $tmpdir/sph.scp | awk -v sph2pipe=$sph2pipe \
      '{printf("%s-A %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2);
        printf("%s-B %s -f wav -p -c 2 %s |\n", $1, sph2pipe, $2);}' | \
    sort -k1,1 -u  > $dir/wav.scp || exit 1;

  log "Created sph.scp and wav.scp files"
fi

utils/fix_data_dir.sh $dir
rm -r $tmpdir

log "Fisher data preparation succeeded"

