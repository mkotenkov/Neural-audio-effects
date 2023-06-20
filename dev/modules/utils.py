import IPython
from torch import nn
import torch
from torch import Tensor
import torch.nn.functional as F


class Conv1dCausal(nn.Module):  # Conv1d with cache
    """Causal convolution (padding applied to only left side)"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int = 1,
                 bias: bool = True) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # input_len == output_len when stride=1
        self.in_channels = in_channels
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              (kernel_size,),
                              (stride,),
                              padding=0,
                              dilation=(dilation,),
                              bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.padding, 0))  # standard zero padding
        x = self.conv(x)
        return x


class FiLM(nn.Module):
    def __init__(self,
                 cond_dim: int,  # dim of conditioning input
                 num_features: int,  # dim of the conv channel
                ) -> None:
        super().__init__()
        self.num_features = num_features
        self.adaptor = nn.Linear(cond_dim, 2 * num_features)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.unsqueeze(-1)
        b = b.unsqueeze(-1)
        x = (x * g) + b  # Then apply conditional affine
        return x