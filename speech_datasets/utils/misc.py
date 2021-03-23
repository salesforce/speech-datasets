import importlib
import inspect
from os.path import abspath, dirname
import torch


def get_root():
    """This file is ROOT/speech_datasets/utils/misc.py, so return ROOT."""
    return dirname(dirname(dirname(abspath(__file__))))


def check_kwargs(func, kwargs, name=None):
    """check kwargs are valid for func

    If kwargs are invalid, raise TypeError as same as python default
    :param function func: function to be validated
    :param dict kwargs: keyword arguments for func
    :param str name: name used in TypeError (default is func name)
    """
    try:
        params = inspect.signature(func).parameters
    except ValueError:
        return
    if name is None:
        name = func.__name__
    for k in kwargs.keys():
        if k not in params:
            raise TypeError(f"{name}() got an unexpected keyword argument '{k}'")


def dynamic_import(import_path, alias=None):
    """dynamic import module and class

    :param str import_path: syntax 'module_name:class_name'
        e.g., 'speech_datasets.transform.add_deltas:AddDeltas'
    :param dict alias: shortcut for registered class
    :return: imported class
    """
    alias = dict() if alias is None else alias
    if import_path not in alias and ":" not in import_path:
        raise ValueError(
            "import_path should be one of {} or "
            'include ":", e.g. "speech_datasets.transform.add_deltas:AddDeltas" : '
            "{}".format(set(alias), import_path)
        )
    if ":" not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(":")
    m = importlib.import_module(module_name)
    return getattr(m, objname)


def set_deterministic_pytorch(seed, cudnn_deterministic=True):
    """Ensures pytorch produces deterministic results based on the seed."""
    # See https://github.com/pytorch/pytorch/issues/6351 about cudnn.benchmark
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = (not cudnn_deterministic)
