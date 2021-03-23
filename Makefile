# Use shell /bin/bash instead of /bin/sh so the source command can be used
SHELL := /bin/bash
# Use the default conda unless a specific install is specified. If there is
# no conda, we will download a fresh one and use it to set up the virtual env.
CONDA :=
VENV_NAME := datasets
# The python version installed in the conda setup
PYTHON_VERSION := 3.7.9
# PyTorch version: 1.2.0, 1.3.0, 1.3.1, 1.4.0, 1.5.0, 1.5.1 (>= 1.2.0 required)
# 1.5.0 and later do not work with PyKaldi...
TORCH_VERSION := 1.4.0

ifeq ($(CONDA),)
    CONDA := $(shell which conda)
endif
ifeq ($(TORCH_VERSION),)
    pytorch := pytorch
else
    pytorch := pytorch=$(TORCH_VERSION)
endif

ifneq ($(shell which nvidia-smi),) # 'nvcc' found
    CUDA_VERSION := $(shell nvcc --version | grep "release" | sed -E "s/.*release ([0-9.]*).*/\1/")
    CONDA_PYTORCH := $(pytorch) cudatoolkit=$(CUDA_VERSION) -c pytorch
else
    CUDA_VERSION :=
    CONDA_PYTORCH := $(pytorch) cpuonly -c pytorch
endif
# Install CPU version of PyKaldi, so we can run feature extraction on CPU while training on GPU
CONDA_PYKALDI := -c pykaldi pykaldi-cpu

.PHONY: all clean

all: conda sph2pipe check_install example

tools/conda.done:
# Only install PyTorch if the PyTorch version is non-empty
	tools/install_anaconda.sh $(PYTHON_VERSION) "$(CONDA)" tools/venv $(VENV_NAME) . "$(CONDA_PYTORCH)" "$(CONDA_PYKALDI)"
	@echo $(VENV_NAME) > tools/conda.done

conda: tools/conda.done

tools/sph2pipe.done:
	tools/install_sph2pipe.sh tools
	touch tools/sph2pipe.done

sph2pipe: tools/sph2pipe.done

check_install: conda
ifneq ($(strip $(CUDA_VERSION)),)
	source tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate $(shell cat tools/conda.done) && python tools/check_install.py
else
	source tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate $(shell cat tools/conda.done) && python tools/check_install.py --no-cuda
endif

example: conda
	source tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate $(shell cat tools/conda.done) && pip install -r example/requirements.txt

clean: clean_conda
	rm -rf tools/*.done

clean_conda:
	rm -rf *.egg-info
	rm -rf tools/venv
	rm -f tools/miniconda.sh
	find . -iname "*.pyc" -delete
