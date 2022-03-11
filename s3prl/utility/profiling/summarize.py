# Time:			March 11 2022 15:39:48
# Author:		Ching-Feng Yeh
# cite from:    https://github.com/andrewyeh/fairseq/blob/7d82b6caa5a06c5ff81bc197f3136ba18db0cacb/examples/speech_recognition/new/profiling/summarize.py

import torch
import logging
import pandas as pd

from prepare_model import is_leaf_module


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def summarize(model, *args, **kwargs):
    """
    summarize() provides a report of:
      1) number of parameters (#params)
      2) number of multiply-add operations (MAdd)
    for the given model model.
    """
    assert isinstance(model, torch.nn.Module)

    def _collect_leaf_modules(_module):
        m = []
        if is_leaf_module(module=_module):
            m.append(_module)
            return m
        for child in _module.children():
            m.extend(_collect_leaf_modules(_module=child))
        return m

    modules = _collect_leaf_modules(_module=model)

    data = []
    for module in modules:
        name = module.__class__.__module__ + '.' + module.__class__.__qualname__
        size = sum([p.numel() for p in module.parameters()])
        madd = module.madd.item()
        data.append((name, size, madd))

    data_frame = pd.DataFrame(data)
    data_frame.columns = ["module", "#params", "MAdd"]

    total = pd.Series(
                [
                    "-" * max([len(name) for name in data_frame["module"]]),
                    data_frame["#params"].sum(),
                    data_frame["MAdd"].sum(),
                ],
                index=["module", "#params", "MAdd"],
                name="total",
            )

    data_frame = data_frame.append(total)

    data_frame['MAdd'] = data_frame['MAdd'].apply(lambda x: '{:,}'.format(x))
    data_frame['#params'] = data_frame['#params'].apply(lambda x: '{:,}'.format(x))

    return data_frame