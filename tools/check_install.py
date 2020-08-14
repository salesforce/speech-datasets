#!/usr/bin/env python3

"""Script to check whether the installation is done correctly."""

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from distutils.version import LooseVersion
import importlib
import logging
import sys

logger = logging.getLogger(__name__)


# NOTE: add the libraries which are not included in install_requires
MANUALLY_INSTALLED_LIBRARIES = [
    ("kaldi", None),                # pykaldi is installed via conda
    ("speech_datasets", "0.1.0")    # make sure this package is installed :)
]

# NOTE: manually check torch installation & its version
COMPATIBLE_TORCH_VERSIONS = (
    "1.2.0",
    "1.3.0",
    "1.3.1",
    "1.4.0",
    "1.5.0",
    "1.5.1"
)


def main(args):
    """Check the installation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disable cuda-related tests",
    )
    args = parser.parse_args(args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info(f"python version = {sys.version}")

    library_list = []

    # check torch installation at first
    try:
        import torch

        logger.info(f"pytorch version = {torch.__version__}")
        if LooseVersion(torch.__version__) < LooseVersion("1.2.0"):
            logger.warning(f"torch>=1.2.0 required, but torch {torch.__version__} received.")
            logger.warning("please re-run this script with an appropriate version of torch")
            sys.exit(1)
        elif torch.__version__ not in COMPATIBLE_TORCH_VERSIONS:
            logger.warning(f"torch {torch.__version__} is not tested. please be careful.")
    except ImportError:
        logger.warning("torch is not installed.")
        logger.warning("please try to setup again and then re-run this script.")
        sys.exit(1)

    library_list.extend(MANUALLY_INSTALLED_LIBRARIES)

    # check library availability
    logger.info("start checking library availability...")
    logger.info("# libraries to be checked = %d" % len(library_list))
    is_correct_installed_list = []
    for idx, (name, version) in enumerate(library_list):
        try:
            importlib.import_module(name)
            logger.info("--> %s is installed." % name)
            is_correct_installed_list.append(True)
        except ImportError as e:
            logger.warning("--> %s is not installed." % name)
            logger.warning("--> import %s failed with exception:\n" % name + str(e))
            is_correct_installed_list.append(False)
    logger.info("library availableness check done.")
    logger.info(
        "%d / %d libraries are correctly installed."
        % (sum(is_correct_installed_list), len(library_list))
    )

    if len(library_list) != sum(is_correct_installed_list):
        logger.warning("please try to setup again and then re-run this script.")
        sys.exit(1)

    # check library version
    num_version_specified = sum(
        [True if v is not None else False for n, v in library_list]
    )
    logger.info("library version check start.")
    logger.info("# libraries to be checked = %d" % num_version_specified)
    is_correct_version_list = []
    for idx, (name, version) in enumerate(library_list):
        if version is not None:
            vers = importlib.import_module(name).__version__
            if vers is not None:
                is_correct = vers in version
                if is_correct:
                    logger.info("--> %s version is matched (%s)." % (name, vers))
                    is_correct_version_list.append(True)
                else:
                    logger.warning(
                        "--> %s version is incorrect (%s is not in %s)."
                        % (name, vers, str(version))
                    )
                    is_correct_version_list.append(False)
            else:
                logger.info(
                    "--> %s has no version info, but version is specified." % name
                )
                logger.info("--> maybe it is better to reinstall the latest version.")
                is_correct_version_list.append(False)
    logger.info("library version check done.")
    logger.info(
        "%d / %d libraries are correct version."
        % (sum(is_correct_version_list), num_version_specified)
    )

    if sum(is_correct_version_list) != num_version_specified:
        logger.info("please try to setup again and then re-run this script.")
        sys.exit(1)

    # check cuda availableness
    if args.no_cuda:
        logger.info("cuda availableness check skipped.")
    else:
        logger.info("cuda availableness check start.")
        import torch

        try:
            assert torch.cuda.is_available()
            logger.info("--> cuda is available in torch.")
        except AssertionError:
            logger.warning("--> it seems that cuda is not available in torch.")
        try:
            assert torch.backends.cudnn.is_available()
            logger.info("--> cudnn is available in torch.")
        except AssertionError:
            logger.warning("--> it seems that cudnn is not available in torch.")
        try:
            assert torch.cuda.device_count() > 1
            logger.info(
                f"--> multi-gpu is available (#gpus={torch.cuda.device_count()})."
            )
        except AssertionError:
            logger.warning("--> it seems that only single gpu is available.")
            logger.warning("--> maybe your machine has only one gpu.")
        logger.info("cuda availableness check done.")

    logger.info("installation check is done.")


if __name__ == "__main__":
    main(sys.argv[1:])
