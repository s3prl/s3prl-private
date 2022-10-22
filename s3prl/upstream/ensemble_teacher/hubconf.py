

from .expert import UpstreamExpert as _UpstreamExpert


def ensemble_teacher(ckpt, *args, **kwargs):

    return _UpstreamExpert(ckpt, *args, **kwargs)
