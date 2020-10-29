FROM pytorch/pytorch:latest
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    apt-utils wget bc gawk vim emacs build-essential locales libfontconfig1 automake \
    sox flac ffmpeg libasound2-dev libsndfile1-dev \
    libfftw3-dev libopenblas-dev libgflags-dev libgoogle-glog-dev gfortran \
    python3 python3-dev python3-pip python3-numpy python3-setuptools
RUN apt update
RUN apt install -y openssh-server openssh-client

# Default to utf-8 encodings in python
# Can verify in container with:
# python -c 'import locale; print(locale.getpreferredencoding(False))'
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
