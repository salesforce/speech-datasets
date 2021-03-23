import io
import logging
import os

import numpy as np

from speech_datasets.transform.interface import TransformInterface
from speech_datasets.utils import get_root
from speech_datasets.utils.readers import read_cmvn_stats

logger = logging.getLogger(__name__)


class CMVN(TransformInterface):
    def __init__(self, cmvn_type: str, stats: str = None, norm_means=True,
                 norm_vars=False, utt2spk: str = None, reverse=False,
                 std_floor=1.0e-20):
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.reverse = reverse
        self.std_floor = std_floor

        assert cmvn_type in ["global", "speaker", "utterance"], cmvn_type
        self.accept_uttid = (cmvn_type != "global")
        self.cmvn_type = cmvn_type
        if cmvn_type != "utterance":
            assert stats is not None, "stats required if cmvn_type != 'utterance'"
            try:
                self.stats_file = stats
                stats_dict = read_cmvn_stats(self.stats_file, cmvn_type)
            except FileNotFoundError:
                self.stats_file = os.path.join(get_root(), stats)
                stats_dict = read_cmvn_stats(self.stats_file, cmvn_type)
        else:
            if stats is not None:
                logger.warning("stats file is not used if cmvn_type is 'utterance'")
            self.stats_file = None
            stats_dict = {}

        if cmvn_type == "speaker":
            assert utt2spk is not None, "utt2spk required if cmvn_type is 'speaker'"
            self.utt2spk = {}
            with io.open(utt2spk, "r", encoding="utf-8") as f:
                for line in f:
                    utt, spk = line.rstrip().split(None, maxsplit=1)
                    self.utt2spk[utt] = spk
        else:
            if utt2spk is not None:
                logger.warning("utt2spk is only used if cmvn_type is 'speaker'")
            self.utt2spk = None

        # Kaldi makes a matrix for CMVN which has a shape of (2, feat_dim + 1),
        # and the first vector contains the sum of feats and the second is
        # the sum of squares. The last value of the first, i.e. stats[0,-1],
        # is the number of samples for this statistics.
        self.bias = {}
        self.scale = {}
        for spk, stats in stats_dict.items():
            # Var[x] = E[x^2] - E[x]^2
            mean = stats.sum / stats.count
            var = stats.sum_squares / stats.count - mean * mean
            std = np.maximum(np.sqrt(var), std_floor)
            self.bias[spk] = -mean
            self.scale[spk] = 1 / std

    def __repr__(self):
        return (
            "{name}(stats_file={stats_file}, "
            "norm_means={norm_means}, norm_vars={norm_vars}, "
            "reverse={reverse})".format(
                name=self.__class__.__name__,
                stats_file=self.stats_file,
                norm_means=self.norm_means,
                norm_vars=self.norm_vars,
                reverse=self.reverse,
            )
        )

    def __call__(self, x, uttid=None):
        if self.cmvn_type == "global":
            bias = self.bias[None]
            scale = self.scale[None]
        elif self.cmvn_type == "speaker":
            spk = self.utt2spk[uttid]
            bias = self.bias[spk]
            scale = self.scale[spk]
        else:  # self.cmvn_type == "utterance"
            mean = x.mean(axis=0)
            mse = (x ** 2).sum(axis=0) / x.shape[0]
            bias = -mean
            scale = 1 / np.maximum(np.sqrt(mse - mean ** 2), self.std_floor)

        if not self.reverse:
            if self.norm_means:
                x = np.add(x, bias)
            if self.norm_vars:
                x = np.multiply(x, scale)

        else:
            if self.norm_vars:
                x = np.divide(x, scale)
            if self.norm_means:
                x = np.subtract(x, bias)

        return x
