#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

# This is needed for certain pods (ffmpeg-3 doesn't exist anymore & messes up apt gets)
rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-ffmpeg-3*
apt-get remove libflac8 -y
apt-get update -y
apt-get upgrade -y
apt-get autoremove -y

# The actual apt installs we need
apt-get install -y apt-utils
apt-get install -y python-pip virtualenv  # This is pip for python2
apt-get install -y emacs less
apt-get install -y gawk
apt-get install -y man
apt-get install -y build-essential libfontconfig1 automake
apt-get install -y sox flac ffmpeg libasound2-dev libsndfile1-dev
apt-get install -y libfftw3-dev libopenblas-dev libgflags-dev libgoogle-glog-dev
apt-get install -y gfortran python3
apt-get install -y bc
apt-get install -y wget
