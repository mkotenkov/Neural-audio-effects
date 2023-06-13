from modules.utils import *

class TransformBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 stride: int = 1) -> None:
        super().__init__()
        self.act = nn.PReLU()
        self.audio_conv = Conv1dCausal(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=True,
        )
        self.merge_info_and_audio = nn.Conv1d(out_ch + 1, out_ch, kernel_size=(1,), bias=False) # TODO info now has 1 ch
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=(1,), bias=False)

    def forward(self, info: Tensor, audio: Tensor) -> Tensor:
        audio_in = audio

        audio = self.audio_conv(audio)
        audio = self.act(audio)

        stacked = torch.cat([info, audio], dim=1)

        audio_result = self.merge_info_and_audio(stacked) # TODO maybe more efficient way to merge info and audio exists
        audio_result += self.res(audio_in)

        return audio_result


n_samples = 48000
n_oscs = 3

t = torch.arange(n_samples) / n_samples
t = t.repeat(n_oscs, 1)
freq = nn.Parameter(torch.ones(n_oscs), requires_grad=True)
print(freq)
sin = torch.sin(freq.reshape(-1, 1) * t)


print(sin)
a = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
print(a)

r = torch.randn(48000)
r = r[None, None, :]
print(r.shape)