import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from s3prl import Output

from . import NNModule
from .pooling import (
    AttentiveStatisticsPooling,
    SelfAttentivePooling,
    TemporalAveragePooling,
    TemporalStatisticsPooling,
)


class TDNN(NNModule):
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
        super().__init__()

        self.kernel = nn.Linear(input_size * context_size, output_size)
        self.nonlinearity = nn.ReLU()
        if self.arguments.batch_norm:
            self.bn = nn.BatchNorm1d(output_size)
        if self.arguments.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, x):
        """
        input:
            x: with size (batch, seq_len, input_size)
        output:
            x: with size (batch, seq_len, output_size)
        """

        _, _, d = x.shape
        assert (
            d == self.input_size
        ), "Input size was wrong. Expected ({}), got ({})".format(self.input_size, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.arguments.context_size, self.input_size),
            stride=(1, self.input_size),
            dilation=(self.arguments.dilation, 1),
        )

        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.nonlinearity(x)

        if self.arguments.dropout_p:
            x = self.drop(x)

        if self.arguments.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x


"""
ECAPA-TDNN
"""


class SEModule(NNModule):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(NNModule):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class ECAPA_TDNN(NNModule):
    """
    ECAPA-TDNN model as in https://arxiv.org/abs/2005.07143
    Reference code: https://github.com/TaoRuijie/ECAPA-TDNN
    This model only include the blocks before the pooling layer
    """

    def __init__(self, input_size=80, output_size=1536, C=1024, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * C, output_size, kernel_size=1)

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, x):
        """
        input:
            x: size (batch, seq_len, input_size)
        output:
            x: size (batch, seq_len, output_size)
        """

        x = self.conv1(x.transpose(1, 2).contiguous())
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)
        x = x.transpose(1, 2).contiguous()

        return Output(output=x)


class XVector(NNModule):
    def __init__(self, input_size: int, output_size: int = 1500, **kwargs):
        super().__init__()
        """
        XVector model as in https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf
        This model only include the blocks before the pooling layer
        """
        self.module = nn.Sequential(
            TDNN(input_size=input_size, output_size=512, context_size=5, dilation=1),
            TDNN(input_size=512, output_size=512, context_size=3, dilation=2),
            TDNN(input_size=512, output_size=512, context_size=3, dilation=3),
            TDNN(input_size=512, output_size=512, context_size=1, dilation=1),
            TDNN(input_size=512, output_size=output_size, context_size=1, dilation=1),
        )

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, x):
        """
        input:
            x: size (batch, seq_len, input_size)
        output:
            x: size (batch, seq_len, output_size)
        """

        x = self.module(x)

        return Output(output=x)


class SpeakerEmbeddingExtractor(NNModule):
    def __init__(
        self,
        input_size: int,
        output_size: int = 1500,
        backbone: str = "XVector",
        pooling_type: str = "TAP",
        **kwargs
    ):
        super().__init__()

        # TODO: add other backbone model; Pay attention to self.offset
        if self.arguments.backbone == "XVector":
            self.backbone = XVector(input_size=input_size, output_size=output_size)
            self.offset = 14

        elif self.arguments.backbone == "ECAPA-TDNN":
            self.backbone = ECAPA_TDNN(input_size=input_size, output_size=output_size)
            self.offset = 0

        else:
            raise ValueError(
                "{} backbone type is not defined".format(self.arguments.backbone)
            )

        pooling_type = self.arguments.pooling_type
        if pooling_type == "TemporalAveragePooling" or pooling_type == "TAP":
            self.pooling = TemporalAveragePooling(
                input_size=self.backbone.output_size,
                output_size=self.backbone.output_size,
            )

        elif pooling_type == "TemporalStatisticsPooling" or pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling(
                input_size=self.backbone.output_size,
                output_size=2 * self.backbone.output_size,
            )
            self.arguments.output_size = 2 * self.backbone.output_size

        elif pooling_type == "SelfAttentivePooling" or pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(
                input_size=self.backbone.output_size,
                output_size=self.backbone.output_size,
            )

        elif pooling_type == "AttentiveStatisticsPooling" or pooling_type == "ASP":
            self.pooling = AttentiveStatisticsPooling(
                input_size=self.backbone.output_size,
                output_size=2 * self.backbone.output_size,
            )
            self.arguments.output_size = 2 * self.backbone.output_size

        else:
            raise ValueError("{} pooling type is not defined".format(pooling_type))

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, x, xlen=None):
        """
        input:
            x: size (batch, seq_len, input_size)
        output:
            x: size (batch, output_size)
        """

        x = self.backbone(x).slice(1)

        if xlen is not None:
            xlen = torch.LongTensor([max(item - self.offset, 0) for item in xlen])
        else:
            xlen = [x.shape[1]] * x.shape[0]
            xlen = torch.LongTensor(xlen)

        x = self.pooling(x, xlen)

        return Output(output=x)
