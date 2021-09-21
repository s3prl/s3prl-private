from .expert import UpstreamExpert as _UpstreamExpert


def wav2vec2_hug(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)


def wav2vec2_hug_base_960_ctc(*args, **kwargs):
    kwargs['ckpt'] = 'facebook/wav2vec2-base-960h'
    return wav2vec2_hug(*args, **kwargs)
