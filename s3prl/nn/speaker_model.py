"""
Speaker verification models

Authors:
  * Haibin Wu 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import (
    AttentiveStatisticsPooling,
    SelfAttentivePooling,
    TemporalAveragePooling,
    TemporalStatisticsPooling,
)

XVECTOR_TDNNS_LENGTH_REDUCTION = 14


__all__ = [
    "TDNN",
    "XVectorBackbone",
    "SpeakerEmbeddingExtractor",
    "SuperbXvector",
]


class TDNN(nn.Module):
    """
    TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
    Affine transformation not applied globally to all frames but smaller windows with local context
    batch_norm: True to include batch normalisation after the non linearity

    Context size and dilation determine the frames selected
    (although context size is not really defined in the traditional sense)
    For example:
        context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
        context size 3 and dilation 2 is equivalent to [-2, 0, 2]
        context size 1 and dilation 1 is equivalent to [0]
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        context_size: int,
        dilation: int,
        dropout_p: float = 0.0,
        stride: int = 1,
        batch_norm: bool = True,
    ):
        super().__init__()
        self._indim = input_size
        self._outdim = output_size
        self.context_size = context_size
        self.dilation = dilation
        self.stride = stride
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_size * context_size, output_size)
        self.nonlinearity = nn.ReLU()
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_size)
        if dropout_p:
            self.drop = nn.Dropout(p=dropout_p)

    @property
    def input_size(self) -> int:
        return self._indim

    @property
    def output_size(self) -> int:
        return self._outdim

    def forward(self, x):
        """
        Args:
            x (torch.FloatTensor): (batch, seq_len, input_size)

        Returns:
            torch.FloatTensor: (batch, seq_len, output_size)
        """

        _, _, d = x.shape
        assert (
            d == self.input_size
        ), "Input size was wrong. Expected ({}), got ({})".format(self.input_size, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_size),
            stride=(1, self.input_size),
            dilation=(self.dilation, 1),
        )

        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.nonlinearity(x)

        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x


class XVectorBackbone(nn.Module):
    """
    The TDNN layers the same as in https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf
    This model only include the blocks before the pooling layer
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1500,
        dropout_p: float = 0.0,
        batch_norm: False = True,
    ):
        super().__init__()
        self._indim = input_size
        self._outdim = output_size

        self.module = nn.Sequential(
            TDNN(
                input_size=input_size,
                output_size=512,
                context_size=5,
                dilation=1,
                dropout_p=dropout_p,
                batch_norm=batch_norm,
            ),
            TDNN(
                input_size=512,
                output_size=512,
                context_size=3,
                dilation=2,
                dropout_p=dropout_p,
                batch_norm=batch_norm,
            ),
            TDNN(
                input_size=512,
                output_size=512,
                context_size=3,
                dilation=3,
                dropout_p=dropout_p,
                batch_norm=batch_norm,
            ),
            TDNN(
                input_size=512,
                output_size=512,
                context_size=1,
                dilation=1,
                dropout_p=dropout_p,
                batch_norm=batch_norm,
            ),
            TDNN(
                input_size=512,
                output_size=output_size,
                context_size=1,
                dilation=1,
                dropout_p=dropout_p,
                batch_norm=batch_norm,
            ),
        )

    @property
    def input_size(self) -> int:
        return self._indim

    @property
    def output_size(self) -> int:
        return self._outdim

    def forward(self, x):
        """
        Args:
            x (torch.FloatTensor): (batch, seq_len, input_size)

        output:
            torch.FloatTensor: (batch, seq_len, output_size)
        """
        # FIXME: This module should take x_len and output y_len
        x = self.module(x)
        return x


class SpeakerEmbeddingExtractor(nn.Module):
    """
    The generic speaker embedding extractor model

    Args:
        input_size (int): input_size
        output_size (int): output_size
        backbone (str): currently only support 'XVector'
        polling_type (str): available pooling class name in :obj:`s3prl.nn.pooling`
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1500,
        backbone: str = "XVector",
        pooling_type: str = "TemporalAveragePooling",
    ):
        super().__init__()
        self._indim = input_size
        self._outdim = output_size

        # TODO: add other backbone model; Pay attention to self.offset
        if backbone == "XVector":
            self.backbone = XVectorBackbone(
                input_size=input_size, output_size=output_size
            )
            self.offset = XVECTOR_TDNNS_LENGTH_REDUCTION
        else:
            raise ValueError("{} backbone type is not defined".format(backbone))

        if pooling_type == "TemporalAveragePooling" or pooling_type == "TAP":
            self.pooling = TemporalAveragePooling(self.backbone.output_size)

        elif pooling_type == "TemporalStatisticsPooling" or pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling(self.backbone.output_size)

        elif pooling_type == "SelfAttentivePooling" or pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(self.backbone.output_size)

        elif pooling_type == "AttentiveStatisticsPooling" or pooling_type == "ASP":
            self.pooling = AttentiveStatisticsPooling(self.backbone.output_size)

        else:
            raise ValueError("{} pooling type is not defined".format(pooling_type))

        self._outdim = self.pooling.output_size

    @property
    def input_size(self) -> int:
        return self._indim

    @property
    def output_size(self) -> int:
        return self._outdim

    def forward(self, x, xlen=None):
        """
        input:
            x: size (batch, seq_len, input_size)
        output:
            x: size (batch, output_size)
        """

        x = self.backbone(x)

        if xlen is not None:
            xlen = torch.LongTensor([max(item - self.offset, 0) for item in xlen])
        else:
            xlen = torch.LongTensor([x.shape(1)] * x.shape(0))

        x = self.pooling(x, xlen)

        return x


class _UtteranceExtractor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._indim = input_size
        self._outdim = output_size

        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(output_size, output_size)
        self.act_fn = nn.ReLU()

    @property
    def input_size(self):
        return self._indim

    @property
    def output_size(self):
        return self._outdim

    def forward(self, x_BxH):
        hid_BxH = self.linear1(x_BxH)
        hid_BxH = self.act_fn(hid_BxH)

        if self.training:
            hid_BxH = self.linear2(hid_BxH)
            hid_BxH = self.act_fn(hid_BxH)

        return hid_BxH


class SuperbXvector(nn.Module):
    """
    The Xvector used in the SUPERB Benchmark with the exact default arguments

    Args:
        input_size (int): The input feature size, usually is the output size of upstream models
        output_size (int): (default, 512) The size of the speaker embedding
        hidden_size (int): (default, 512) The major hidden size in the network
        aggregation_size (int): (default, 1500) The output size of the x-vector, which is usually large
        dropout_p (float): (default, 0.0) The dropout rate
        batch_norm (bool): (default, False) Use batch norm for TDNN layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 512,
        hidden_size: int = 512,
        aggregation_size: int = 1500,
        dropout_p: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size

        self.projector = nn.Linear(input_size, hidden_size)
        self.tdnns = XVectorBackbone(
            hidden_size, aggregation_size, dropout_p=dropout_p, batch_norm=batch_norm
        )
        latest_size = self.tdnns.output_size

        self.pooling = TemporalStatisticsPooling(latest_size)
        latest_size = self.pooling.output_size

        self.affine = _UtteranceExtractor(latest_size, output_size)

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x, x_len):
        """
        Args:
            x (torch.FloatTensor): (batch_size, seq_len, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Returns:
            torch.FloatTensor:

            (batch_size, output_size)
        """

        x = self.projector(x)

        x = self.tdnns(x)
        x_len = x_len - XVECTOR_TDNNS_LENGTH_REDUCTION
        assert (
            x_len <= 0
        ).sum() == 0, "The input sequence is too short for the X-vector model"

        x = self.pooling(x, x_len)
        x = self.affine(x)
        return x
