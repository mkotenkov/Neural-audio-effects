"""maybe some problems here (maybe not)"""

import torch
import torch.nn as nn


class CondWaveLearner(nn.Module):
    def __init__(self, n_channels, n_waves, cond_size, min_freq, max_freq, buffer_size):
        super().__init__()
        self.n_waves = n_waves
        self.n_channels = n_channels
        self.buffer_size = buffer_size
        self.cond_size = cond_size
        self.t = nn.Parameter(torch.arange(self.buffer_size) / self.buffer_size).requires_grad_(False)

        linspace = torch.linspace(start=min_freq, end=max_freq, steps=n_waves).to(torch.float32) * 2 * torch.pi
        self.freqs = nn.Parameter(linspace.repeat(n_channels, 1))
        self.biases = nn.Parameter(torch.zeros(n_channels, n_waves))

        self.condition_adaptor = nn.Sequential(
            nn.Linear(cond_size, 128), nn.Tanh(),
            nn.Linear(128, n_waves * 2)
        ).requires_grad_(False)

    def forward(self, cond):
        assert cond.ndim == 2, cond.shape  # (batch_size, cond_size)
        assert cond.shape[1] == self.cond_size, f"{cond.shape[1]} != {self.cond_size}"

        b_mod, f_mod = torch.chunk(self.condition_adaptor(cond), chunks=2, dim=1)

        biases = b_mod[:, None, :] + self.biases
        freqs = f_mod[:, None, :] + self.freqs

        batch_size = cond.shape[0]
        t = torch.stack([torch.stack([self.t.repeat(self.n_waves, 1)] * self.n_channels)] * batch_size)
        mul = t * freqs[..., None] + biases[..., None]

        sines = torch.sin(mul)
        sines = sines.flatten(end_dim=1)
        sines = sines.squeeze()
        return sines


if __name__ == '__main__':
    wave_learner = CondWaveLearner(
        n_waves=20,
        n_channels=2,
        cond_size=3,
        buffer_size=44100,
        min_freq=1,
        max_freq=20
    )

    cond = torch.ones(32, 3)
    res = wave_learner(cond)
    print(res.shape)
