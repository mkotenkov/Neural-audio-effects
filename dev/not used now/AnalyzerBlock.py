import torch
from torch import nn
from torch import Tensor

from dev.modules.CondWaveLearner import CondWaveLearner
from dev.modules.IntelligentMerge import IntelligentMerge

class AnalyzerBlock(nn.Module):
    def __init__(self, wave_learner_params : dict, a_channels, b_channels) -> None:
        super().__init__()

        self.wave_learner = CondWaveLearner(**wave_learner_params)

        self.merge = IntelligentMerge(
            a_channels=a_channels,
            b_channels=b_channels,
            out_channels=1,
            hidden_size=64,
            act_func=nn.Tanh()
        )

    def forward(self, audio: Tensor, cond: Tensor) -> Tensor:
        learned_wave = self.wave_learner(cond)
        info = self.merge(audio, learned_wave)
        return info


buffer_size = 512

wave_learner_params = dict(
    n_waves=100,
    buffer_size=buffer_size,
    min_freq=0.5,
    max_freq=100
)

analyzer_block = AnalyzerBlock(wave_learner_params)
audio_example = torch.randn(1, 2, buffer_size)
cond = torch.zeros(1)

print(analyzer_block(audio_example, cond).shape)

