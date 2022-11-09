# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/whisper/expert.py ]
#   Synopsis     [ the whisper wrapper ]
#   Author       [ OpenAI ]
"""*********************************************************************************************"""


from .expert import UpstreamExpert as _UpstreamExpert


def whisper_tiny(*args, **kwargs):
    return _UpstreamExpert(name='tiny', *args, **kwargs)


def whisper_base(*args, **kwargs):
    return _UpstreamExpert(name='base', *args, **kwargs)


def whisper_small(*args, **kwargs):
    return _UpstreamExpert(name='small', *args, **kwargs)


def whisper_medium(*args, **kwargs):
    return _UpstreamExpert(name='medium', *args, **kwargs)


def whisper_large(*args, **kwargs):
    return _UpstreamExpert(name='large', *args, **kwargs)