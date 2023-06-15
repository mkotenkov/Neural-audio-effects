from .IntelligentMerge import IntelligentMerge
from .CondWaveLearner import CondWaveLearner
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

        # info
        max_freq = max_freq / dilation
        self.wave_learner = CondWaveLearner(
            n_waves=n_waves,
            n_channels=out_ch,
            cond_size=cond_size,
            buffer_size=buffer_size,
            min_freq=min_freq,
            max_freq=max_freq
        )

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


        # self.bias_regression = GBiasReg(in_ch, buffer_size, n_layers=6)



        # result
        self.merge = IntelligentMerge(
            a_channels=out_ch * n_waves,
            b_channels=out_ch,
            out_channels=out_ch,
            hidden_size=64,
            act_func=nn.Tanh()
        )


    def forward(self, audio: Tensor, cond: Tensor) -> Tensor:
        assert audio.ndim == 3  # (batch_size, in_ch, samples)
        assert cond.ndim == 2  # (batch_size, cond_size)
        assert cond.shape[1] == self.cond_size

        info = self.wave_learner(cond)

        audio_in =  audio
        audio = self.act(self.audio_conv(audio))
        audio += self.res(audio_in)

        result = self.merge(info, audio)
        return result
