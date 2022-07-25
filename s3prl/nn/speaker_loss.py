import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from s3prl import Output

from . import NNModule


class softmax(NNModule):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)
        self.criertion = nn.CrossEntropyLoss()

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, x, label):
        """
        Args:
            x (torch.Tensor): (batch_size, input_size)
            label (torch.LongTensor): (batch_size, )

        Return:
            loss (torch.float)
            logit (torch.Tensor): (batch_size, )
        """

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.input_size

        x = F.normalize(x, dim=1)
        x = self.fc(x)
        loss = self.criertion(x, label)

        return Output(loss=loss, logit=x)


class amsoftmax(NNModule):
    def __init__(
        self, input_size: int, output_size: int, margin: float = 0.2, scale: float = 30
    ):
        super(amsoftmax, self).__init__()

        self.W = torch.nn.Parameter(
            torch.randn(input_size, output_size), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, x, label):
        """
        Args:
            x (torch.Tensor): (batch_size, input_size)
            label (torch.LongTensor): (batch_size, )

        Return:
            loss (torch.float)
            logit (torch.Tensor): (batch_size, )
        """

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.input_size

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda:
            label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(
            1, label_view, self.arguments.margin
        )
        if x.is_cuda:
            delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.arguments.scale * costh_m
        loss = self.ce(costh_m_s, label)

        return Output(loss=loss, logit=costh_m_s)


class aamsoftmax(NNModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        margin: float = 0.3,
        scale: float = 15,
        easy_margin=False,
        **kwargs
    ):
        super(aamsoftmax, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = input_size
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(output_size, input_size), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, x, label):
        """
        Args:
            x (torch.Tensor): (batch_size, input_size)
            label (torch.LongTensor): (batch_size, )

        Return:
            loss (torch.float)
        """

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        return Output(loss=loss)
