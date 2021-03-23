from distutils.util import strtobool
from typing import Optional, Tuple, Union

import humanfriendly
import numpy as np
from typeguard import check_argument_types


class CMVNStats(object):
    def __init__(self, count, sum, sum_squares):
        self.count = count
        self.sum = sum
        self.sum_squares = sum_squares

    def __iadd__(self, other):
        self.count += other.count
        self.sum += other.sum
        self.sum_squares += other.sum_squares
        return self

    @classmethod
    def from_numpy(cls, stats):
        stats = np.copy(stats)
        assert len(stats) == 2, stats.shape
        # If feat has >2 dims, only use the first one for count
        count = stats[0, -1].flatten()[0]
        return cls(count=count, sum=stats[0, :-1], sum_squares=stats[1, :-1])

    def to_numpy(self):
        shape = (2, self.sum.shape[0] + 1, *self.sum.shape[1:])
        arr = np.empty(shape, dtype=np.float64)
        arr[0, :-1] = self.sum
        arr[1, :-1] = self.sum_squares
        arr[0, -1] = self.count
        arr[1, -1] = 0.0
        return arr


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def int_or_none(value: str) -> Optional[int]:
    """int_or_none.

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=int_or_none)
        >>> parser.parse_args(['--foo', '456'])
        Namespace(foo=456)
        >>> parser.parse_args(['--foo', 'none'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'null'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'nil'])
        Namespace(foo=None)

    """
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return int(value)


def float_or_none(value: str) -> Optional[float]:
    """float_or_none.

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=float_or_none)
        >>> parser.parse_args(['--foo', '4.5'])
        Namespace(foo=4.5)
        >>> parser.parse_args(['--foo', 'none'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'null'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'nil'])
        Namespace(foo=None)

    """
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return float(value)


def humanfriendly_or_none(value) -> Optional[float]:
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return humanfriendly.parse_size(value)


def str2int_tuple(integers: str) -> Optional[Tuple[int, ...]]:
    """

    >>> str2int_tuple('3,4,5')
    (3, 4, 5)

    """
    assert check_argument_types()
    if integers.strip() in ("none", "None", "NONE", "null", "Null", "NULL"):
        return None
    return tuple(map(int, integers.strip().split(",")))


def str_or_int(value: str) -> Union[str, int]:
    try:
        return int(value)
    except ValueError:
        return value


def str_or_none(value: str) -> Optional[str]:
    """str_or_none.

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=str_or_none)
        >>> parser.parse_args(['--foo', 'aaa'])
        Namespace(foo='aaa')
        >>> parser.parse_args(['--foo', 'none'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'null'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'nil'])
        Namespace(foo=None)

    """
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return value
