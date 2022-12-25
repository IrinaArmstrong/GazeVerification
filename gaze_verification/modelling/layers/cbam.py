import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked
from typing import Tuple, Optional


@typechecked
class CBAM(nn.Module):
    """
    Full Convolutional Bloch Attention Mechanism module (CBAM).

    The CBAM module takes as input a tensor of feature maps of shape [C x H x W]
    and apply two self-attention mechanisms consecutively.

    The first attention mechanism (Channel Attention) is applied channel-wise,
    in that we want to select the channels (or features)
    that are the more relevant independently from any spatial considerations.
    Implemented in ChannelAttention class.

    The second attention mechanism (Spatial Attention) is applied along the two spatial dimensions.
    We want to select the more relevant locations in the feature maps independently
    from the channels.
    Implemented in SpatialAttention class.

    Full implementation is taken from:
    https://github.com/elbuco1/CBAM

    Paper: https://arxiv.org/abs/1807.06521
    """

    def __init__(self, gate_channels: int,
                 reduction_ratio: int,
                 pool_types: Optional[Tuple[str, ...]] = ('avg', 'max'),
                 no_spatial: bool = False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels,
                                       reduction_ratio,
                                       pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


@typechecked
class ChannelGate(nn.Module):
    """
    The implementation of the Channel Attention block.
    """

    def __init__(self, gate_channels: int, reduction_ratio: int = 16,
                 pool_types: Optional[Tuple[str, ...]] = ('avg', 'max')):
        super(ChannelGate, self).__init__()
        self._gate_channels = gate_channels
        self._mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self._pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self._pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self._mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self._mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool1d(x, 2, x.size(2), stride=x.size(2))
                channel_att_raw = self._mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self._mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)
        return x * scale


# Channel Attention Module
class Flatten(nn.Module):
    """
    Layer for flattening n-dimensional input tensors to [dim_0, -1] shape.
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor: torch.Tensor) -> torch.Tensor:
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


@typechecked
class SpatialGate(nn.Module):
    """
    The implementation of the Spatial Attention mechanism.
    as 1d convolution over Max+Avg Pooled input with sigmoid activation in the end.
    """

    def __init__(self, kernel_size: int = 3, stride: int = 1, add_relu: bool = False):
        super(SpatialGate, self).__init__()
        self.kernel_size = kernel_size
        self._compress = ChannelPool()
        self._spatial = BasicConv(2, 1,
                                  self.kernel_size,
                                  stride=stride,
                                  padding=(self.kernel_size - 1) // 2,
                                  relu=add_relu)

    def forward(self, x):
        x_compress = self._compress(x)
        x_out = self._spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class BasicConv(nn.Module):
    """
    Spatial Attention Module (SAM).

    Applies a 1D convolution over an input signal with
    following batch normalization and ReLu over output.
    """

    def __init__(self, in_planes: int, out_planes: int,
                 kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1,
                 groups: int = 1, add_relu: bool = True,
                 bn: bool = True, bias: bool = False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if add_relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    """
    Make stack of Max Pooling and Average Pooling operations subsequently over input data.
    """

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)



