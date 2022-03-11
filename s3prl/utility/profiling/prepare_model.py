


import torch
import fairseq
import logging
import numpy as np

from compute_madd import compute_madd

import importlib
import inspect


# NOTE(cfyeh): all modules under fairseq.modules are considered as leaf modules.
LEAF_MODULES = tuple([
    cls for _, cls in inspect.getmembers(
        importlib.import_module("fairseq.modules"), inspect.isclass
    )
])


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def is_leaf_module(module):
    assert isinstance(module, torch.nn.Module)
    if len(list(module.children())) == 0:
        return True
    if isinstance(module, LEAF_MODULES):
        return True
    return False


def prepare_model(model, *args, **kwargs):
    """
    prepare_model() adds profiling attributes (such as "madd") to the modules
    in the model and augments the origial __call__() method with profiling.
    """
    assert isinstance(model, torch.nn.Module)

    def _register_buffer(_module):
        assert isinstance(_module, torch.nn.Module)
    
        if not is_leaf_module(module=_module):
            return
    
        _module.register_buffer('madd', torch.zeros(1).long())

    class_to_call = {}  # storing the original `__call__()` for each module.
    model.eval()
    model.apply(_register_buffer)  # add attributes to modules

    def new_call(module, *_args, **_kwargs):
        assert module.__class__ in class_to_call

        # call the original call() to get output tensors.
        output = class_to_call[module.__class__](module, *_args, **_kwargs)

        # compute madd for the module and add to it as registered buffer.
        madd = compute_madd(module, output, *_args, **_kwargs)
        module.madd = torch.from_numpy(np.array([madd], dtype=np.int64))

        return output

    def _replace_call(module):
        if is_leaf_module(module=module) and module.__class__ not in class_to_call:
            class_to_call[module.__class__] = module.__class__.__call__
            module.__class__.__call__ = new_call
        for child in module.children():
            _replace_call(module=child)

    _replace_call(module=model)

    return model