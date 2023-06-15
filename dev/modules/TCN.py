from typing import List

from .TCNBlock import TCNBlock
from .utils import *

class TCN(nn.Module):
    def __init__(self,
                 n_cores,
                 cond_size,
                 buffer_size,
                 channels: List[int],
                 dilations: List[int],
                 in_ch: int = 1,
                 out_ch: int = 1,
                 kernel_size: int = 13) -> None:
        super().__init__()
        self.channels = channels  # intermediate channels
        self.in_ch = in_ch  # input channels
        self.out_ch = out_ch  # output channels
        self.kernel_size = kernel_size
        self.out_net = nn.Conv1d(channels[-1], out_ch, kernel_size=(1,), stride=(1,), bias=False)
        self.n_blocks = len(channels)
        assert len(dilations) == self.n_blocks
        self.dilations = dilations
        # only supports stride=1 for now
        strides = [1] * self.n_blocks
        self.strides = strides

        self.blocks = nn.ModuleList()
        block_out_ch = None
        for idx, (curr_out_ch, dil, stride) in enumerate(zip(channels, dilations, strides)):
            if idx == 0:
                block_in_ch = in_ch
            else:
                block_in_ch = block_out_ch
            block_out_ch = curr_out_ch

            self.blocks.append(TCNBlock(
                n_cores,
                cond_size,
                buffer_size,
                block_in_ch,
                block_out_ch,
                kernel_size,
                dil,
                stride
            ))

    def forward(self, audio: Tensor, cond: Tensor) -> Tensor:
        assert audio.ndim == 3  # (batch_size, in_ch, samples)

        for block in self.blocks:
            audio = block(audio, cond)
        audio = self.out_net(audio)
        return audio

    def calc_receptive_field(self) -> int:
        """Compute the receptive field in samples."""
        assert all(_ == 1 for _ in self.strides)  # TODO(cm): add support for dsTCN
        assert self.dilations[0] == 1  # TODO(cm): add support for >1 starting dilation
        rf = self.kernel_size
        for dil in self.dilations[1:]:
            rf = rf + ((self.kernel_size - 1) * dil)
        return rf