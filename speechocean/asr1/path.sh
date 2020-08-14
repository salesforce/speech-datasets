MAIN_ROOT=$(dirname "$(dirname "${PWD}")")
export LC_ALL=C

if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi

# Activate local virtual environment for development
error_msg="Virtual environment not set up properly! Navigate to $MAIN_ROOT and run 'make clean all'"
if [ -e $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh ] && [ -e $MAIN_ROOT/tools/conda.done ]; then
    VENV_NAME=$(cat "${MAIN_ROOT}/tools/conda.done")
    source $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh && conda deactivate
    if conda env list | (grep -q -E "${VENV_NAME}\s"); then
        conda activate "${VENV_NAME}"
    else
        echo "${error_msg}" && exit 1
    fi
else
    echo "${error_msg}" && exit 1
fi

# Add binary scripts to the path, to allow them to be run easily
export PATH=$MAIN_ROOT/speech_datasets/bin:$PATH
export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"
