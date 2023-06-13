from modules import IntelligentMerge, TransformBlock

from modules.utils import *


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

        max_freq = max_freq / dilation
        self.wave_learner = CondWaveLearner_NoWeights(
            n_waves=n_waves,
            n_channels=out_ch,
            cond_size=cond_size,
            buffer_size=buffer_size,
            min_freq=min_freq,
            max_freq=max_freq
        )

        self.merge = IntelligentMerge(
            a_channels=out_ch * n_waves,
            b_channels=out_ch,
            out_channels=out_ch,
            hidden_size=64,
            act_func=nn.Tanh()
        )

        self.transformer = TransformBlock(in_ch, out_ch, kernel_size, dilation, stride)

    def forward(self, audio: Tensor, cond: Tensor) -> Tensor:
        assert audio.ndim == 3  # (batch_size, in_ch, samples)

        info = self.wave_learner(cond)
        audio = self.transformer(info, audio)

        result = self.merge(info, audio)

        return result