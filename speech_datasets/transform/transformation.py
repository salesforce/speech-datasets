import copy
import io
import logging
import sys
from typing import Union

import yaml

from speech_datasets.utils import dynamic_import
from speech_datasets.transform.interface import TransformInterface

logger = logging.getLogger(__name__)
PY2 = sys.version_info[0] == 2

if PY2:
    from collections import Sequence
    from funcsigs import signature
else:
    # The ABCs from 'collections' will stop working in 3.8
    from collections.abc import Sequence
    from inspect import signature


# TODO(karita): inherit TransformInterface
import_alias = dict(
    identity="speech_datasets.transform.transformation:Identity",
    time_warp="speech_datasets.transform.spec_augment:TimeWarp",
    time_mask="speech_datasets.transform.spec_augment:TimeMask",
    freq_mask="speech_datasets.transform.spec_augment:FreqMask",
    spec_augment="speech_datasets.transform.spec_augment:SpecAugment",
    speed_perturbation="speech_datasets.transform.perturb:SpeedPerturbation",
    volume_perturbation="speech_datasets.transform.perturb:VolumePerturbation",
    noise_injection="speech_datasets.transform.perturb:NoiseInjection",
    bandpass_perturbation="speech_datasets.transform.perturb:BandpassPerturbation",
    rir_convolve="speech_datasets.transform.perturb:RIRConvolve",
    delta="speech_datasets.transform.add_deltas:AddDeltas",
    cmvn="speech_datasets.transform.cmvn:CMVN",
    fbank="speech_datasets.transform.spectrogram:LogMelSpectrogram",
    fbank_pitch="speech_datasets.transform.spectrogram:FbankPitch",
    spectrogram="speech_datasets.transform.spectrogram:Spectrogram",
)


class Identity(TransformInterface):
    """Identity Function"""

    def __call__(self, x):
        return x


class Transformation(object):
    """Apply some functions to the mini-batch

    Examples:
        >>> import numpy as np
        >>> process = [{"type": "fbank", "n_mels": 80, "samp_freq": 16000},
        ...            {"type": "cmvn", "stats": "data/train/cmvn.ark", "norm_vars": True},
        ...            {"type": "delta", "window": 2, "order": 2}]
        >>> transform = Transformation(process)
        >>> bs = 10
        >>> xs = [np.random.randn(100, 80).astype(np.float32)
        ...       for _ in range(bs)]
        >>> xs = transform(xs)
    """

    def __init__(self, conf: Union[list, str] = None, precomputed_feats: str = None):
        if conf is not None:
            if isinstance(conf, list):
                conf = copy.deepcopy(conf)
            else:
                with io.open(conf, encoding="utf-8") as f:
                    conf = yaml.safe_load(f) or []
                    assert isinstance(conf, list), type(conf)
        else:
            conf = []

        i0 = 0
        if precomputed_feats is not None:
            for i, process in enumerate(conf):
                assert isinstance(process, dict), type(process)
                if dict(process)["type"] == precomputed_feats:
                    i0 = i + 1
        if i0 > 0:
            skip = "\n" + "\n".join(str(x) for x in conf[:i0])
            logger.warning(f"{precomputed_feats} is pre-computed, so skipping "
                           f"the following steps in conf: {skip}")

        self.functions = []
        for process in conf[i0:]:
            assert isinstance(process, dict), type(process)
            opts = dict(process)
            process_type = opts.pop("type")
            class_obj = dynamic_import(process_type, import_alias)
            # TODO(karita): assert issubclass(class_obj, TransformInterface)
            try:
                self.functions.append(class_obj(**opts))
            except TypeError:
                try:
                    signa = signature(class_obj)
                except ValueError:
                    # Some function, e.g. built-in function, are failed
                    pass
                else:
                    logger.error(
                        "Expected signature: {}({})".format(
                            class_obj.__name__, signa
                        )
                    )
                raise

    def is_null(self):
        return len(self.functions) == 0 or all(isinstance(f, Identity) for f in self.functions)

    def __repr__(self):
        rep = "\n" + "\n".join(
            "    {}: {}".format(i, f) for i, f in enumerate(self.functions)
        )
        return "{}({})".format(self.__class__.__name__, rep)

    def __call__(self, x, uttid_list=None, **kwargs):
        """Return new mini-batch

        :param Union[Sequence[np.ndarray], np.ndarray] x:
        :param Union[Sequence[str], str] uttid_list:
        :return: batch:
        :rtype: List[np.ndarray]
        """
        if not isinstance(x, Sequence):
            is_batch = False
            x = [x]
        else:
            is_batch = True

        if isinstance(uttid_list, str):
            uttid_list = [uttid_list for _ in range(len(x))]

        for idx, func in enumerate(self.functions):
            # Derive only the args which the func has
            try:
                param = signature(func).parameters
            except ValueError:
                # Some function, e.g. built-in function, are failed
                param = {}
            _kwargs = {k: v for k, v in kwargs.items() if k in param}
            try:
                if uttid_list is not None and "uttid" in param:
                    x = [func(x, u, **_kwargs) for x, u in zip(x, uttid_list)]
                else:
                    x = [func(x, **_kwargs) for x in x]
            except Exception:
                logger.fatal(
                    "Catch a exception from {}th func: {}".format(idx, func)
                )
                raise

        if is_batch:
            return x
        else:
            return x[0]
