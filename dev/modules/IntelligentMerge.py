import torch
from torch import nn


class IntelligentMerge(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, act_func, cond_size):
        super().__init__()
        self.in_channels = in_channels
        self.cond_size = cond_size

        self.model = nn.Sequential(
            nn.Linear(in_channels, hidden_size), act_func,
            nn.Linear(hidden_size, out_channels)
        )

    def forward(self, x, cond):
        assert x.ndim == 3, f"{x.shape}"  # (batch_size, channels, n_samples)
        assert x.shape[1] == self.in_channels, f"{x.shape}"
        assert cond.ndim == 2, cond.shape  # (batch_size, cond_size)
        assert cond.shape[1] == self.cond_size, f"{cond.shape[1]} != {self.cond_size}"

        batch_size = x.shape[0]

        cat = x.permute(0, 2, 1).flatten(end_dim=1)

        out = self.model(cat)

        batches = out.chunk(batch_size)
        stacked = torch.stack(batches)
        result = stacked.permute(0, 2, 1)

        return result