from .IntelligentMerge import IntelligentMerge
from .WaveLearner import WaveLearner
from .utils import *


class TCNBlock(nn.Module):
    def __init__(self,
                 n_cores,
                 cond_size,
                 buffer_size,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 stride: int = 1) -> None:
        super().__init__()
        self.cond_size = cond_size
        self.in_ch = in_ch

        # info
        self.wave_learner = WaveLearner(
            n_cores=n_cores,
            n_channels=out_ch,
            out_size=buffer_size
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

        # result
        self.merge = IntelligentMerge(
            in_channels=in_ch + out_ch * 2,
            out_channels=out_ch,
            hidden_size=64,
            act_func=nn.Tanh(),
            cond_size=cond_size
        )

        self.film = FiLM(cond_size, out_ch)

    def forward(self, audio: Tensor, cond: Tensor) -> Tensor:
        assert audio.ndim == 3  # (batch_size, in_ch, samples)
        assert cond.ndim == 2  # (batch_size, cond_size)
        assert cond.shape[1] == self.cond_size

        batch_size = audio.shape[0]
        info = torch.stack([self.wave_learner()] * batch_size)

        audio_in = audio

        audio = self.audio_conv(audio)
        audio = self.film(audio, cond)
        audio = self.act(audio)

        to_merge = torch.cat([audio_in, audio, info], dim=1)

        result = self.merge(to_merge, cond)

        return result
