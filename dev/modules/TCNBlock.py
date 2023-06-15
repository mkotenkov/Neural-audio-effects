from .IntelligentMerge import IntelligentMerge
from .GBiasReg import GBiasReg
from .utils import *


class TCNBlock(nn.Module):
    def __init__(self,
                 n_waves,
                 cond_size,
                 buffer_size,
                 min_freq,
                 max_freq,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 stride: int = 1) -> None:
        super().__init__()
        self.cond_size = cond_size

        # audio
        self.act = nn.PReLU()
        self.audio_conv = Conv1dCausal(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=True,
        )

        # result
        self.merge = IntelligentMerge(
            a_channels=in_ch,
            b_channels=out_ch,
            out_channels=out_ch,
            hidden_size=64,
            act_func=nn.Tanh()
        )


    def forward(self, audio: Tensor, cond: Tensor) -> Tensor:
        assert audio.ndim == 3  # (batch_size, in_ch, samples)
        assert cond.ndim == 2  # (batch_size, cond_size)
        assert cond.shape[1] == self.cond_size

        audio_in =  audio
        audio = self.act(self.audio_conv(audio))

        result = self.merge(audio_in, audio)

        return result
