#!/bin/bash
set -euo pipefail

if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi
CONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

n_required_args=5
if [ $# -lt $n_required_args ] ; then
    echo "Usage: $0 <python-version> <conda> <venv_dir> <venv_name> <package_root> [conda install args]*"
    exit 1;
fi
PYTHON_VERSION="$1"
CONDA="$2"
VENV_DIR="$3"
VENV_NAME="$4"
PACKAGE_ROOT="$5"
shift $n_required_args

# Download conda if an installation isn't specified
if [ -z ${CONDA} ]; then
    CONDA="${VENV_DIR}/bin/conda"
    if [ ! -e miniconda.sh ]; then
        wget --tries=3 "${CONDA_URL}" -O miniconda.sh
    fi
    if [ ! -e "$(pwd)/${VENV_DIR}" ]; then
        bash miniconda.sh -b -p "$(pwd)/${VENV_DIR}"
    fi
else
    ln -sf "$(${CONDA} info --base)" "${VENV_DIR}"
fi

# Check if environment alreay exists
if ${CONDA} env list | (! grep -q -E "${VENV_NAME}\s"); then
    ${CONDA} create -y -n "${VENV_NAME}" "python=${PYTHON_VERSION}"
else
    read -r -p "Enviroment ${VENV_NAME} already exists. Continue setup anyways? (y/n) " choice
    case $choice in
        y|Y|yes|Yes ) echo "Continuing to set up environment ${VENV_NAME}." ;;
        * ) echo "Either pick a different value for VENV_NAME, or remove the ${CONDA} environment ${VENV_NAME} before re-running this script." && exit 1 ;;
    esac
fi

# Activate conda environment & check Python version
source "${VENV_DIR}/etc/profile.d/conda.sh" && conda deactivate && conda activate "${VENV_NAME}"
INSTALLED_PYTHON_VERSION=$(python -V | grep -Eo "[[:digit:].]*")
if [ ${INSTALLED_PYTHON_VERSION} != ${PYTHON_VERSION} ]; then
    echo "Enviroment ${VENV_NAME} is Python ${INSTALLED_PYTHON_VERSION}, but Python ${PYTHON_VERSION} requested."
    read -r -p "Continue setup with Python ${INSTALLED_PYTHON_VERSION} anyways? (y/n) " choice
    case $choice in
        y|Y|yes|Yes ) echo "Continuing to set up environment ${VENV_NAME}." ;;
        * ) echo "Either pick a different value for VENV_NAME, or change PYTHON_VERSION to ${INSTALLED_PYTHON_VERSION} before re-running this script." && exit 1 ;;
    esac
fi

conda install -y setuptools pip conda -c anaconda
conda update -y -n "${VENV_NAME}" conda

# Install any conda dependencies (specified via command line)
while (( "$#" )); do
    echo ""
    echo "conda install -y $1"
    conda install -y $1
    shift
done

# Install the speech_datasets package in editable mode
pip install -e "${PACKAGE_ROOT}"
