"""Initialize sub package & declare general util functions.."""
import importlib as _importlib
import inspect as _inspect
from os.path import dirname as _dirname, abspath as _abspath
import torch as _torch


def get_root():
    """This file is ROOT/dataset/utils/__init__.py, so return ROOT."""
    return _dirname(_dirname(_dirname(_abspath(__file__))))


def check_kwargs(func, kwargs, name=None):
    """check kwargs are valid for func

    If kwargs are invalid, raise TypeError as same as python default
    :param function func: function to be validated
    :param dict kwargs: keyword arguments for func
    :param str name: name used in TypeError (default is func name)
    """
    try:
        params = _inspect.signature(func).parameters
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
    m = _importlib.import_module(module_name)
    return getattr(m, objname)


def set_deterministic_pytorch(seed, cudnn_deterministic=True):
    """Ensures pytorch produces deterministic results based on the seed."""
    # See https://github.com/pytorch/pytorch/issues/6351 about cudnn.benchmark
    _torch.manual_seed(seed)
    _torch.backends.cudnn.deterministic = cudnn_deterministic
    _torch.backends.cudnn.benchmark = (not cudnn_deterministic)
