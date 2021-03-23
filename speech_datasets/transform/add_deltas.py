import numpy as np

from speech_datasets.transform.interface import FuncTrans


def delta(feat, window):
    assert window > 0
    delta_feat = np.zeros_like(feat)
    for i in range(1, window + 1):
        delta_feat[:-i] += i * feat[i:]
        delta_feat[i:] += -i * feat[:-i]
        delta_feat[-i:] += i * feat[-1]
        delta_feat[:i] += -i * feat[0]
    delta_feat /= 2 * sum(i ** 2 for i in range(1, window + 1))
    return delta_feat


def add_deltas(x, window=2, order=2):
    """
    :param x: Features
    :param window: size of the window to use to approximate time derivative computation
    :param order: highest order time derivative to compute
    :return: Features, concatenated with all the relevant derivatives
    """
    feats = [x]
    for _ in range(order):
        feats.append(delta(feats[-1], window))
    return np.concatenate(feats, axis=1)


class AddDeltas(FuncTrans):
    _func = add_deltas
    __doc__ = add_deltas.__doc__
