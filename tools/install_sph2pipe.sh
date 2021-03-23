#!/bin/bash
set -euo pipefail

if [ $# != 1 ]; then
    echo "Usage: $0 <dir>"
    exit 1;
fi
pwd=$PWD
dir=$1

if [ ! -e sph2pipe_v2.5.tar.gz ]; then
  wget --no-check-certificate https://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz
fi

tar xzvf sph2pipe_v2.5.tar.gz -C $dir
rm sph2pipe_v2.5.tar.gz

cd $dir/sph2pipe_v2.5
gcc -o sph2pipe *.c -lm
cd $pwd
