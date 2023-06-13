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


def window(audio, buffer_size, drop_last=True):
    assert audio.ndim == 3  # (batch_size, ch, n_samples)
    s = audio.shape[2] // buffer_size
    for i in range(s):
        start = buffer_size * i
        end = buffer_size * (i + 1)
        yield audio[:, :, start:end]
    if len(audio) > buffer_size * s and not drop_last: yield audio[:, :, buffer_size * s:]


def pair_window(audio, buffer_size):
    for prev, curr in zip(window(audio, buffer_size), window(audio[buffer_size:], buffer_size)):
        if len(prev) == len(curr): yield prev, curr


def sonify(audio, sr):
    IPython.display.display(IPython.display.Audio(data=audio, rate=sr))
