import torch
from torch import nn


class IntelligentMerge(nn.Module):
    def __init__(self, a_channels, b_channels, out_channels, hidden_size, act_func):
        super().__init__()
        self.a_channels = a_channels
        self.b_channels = b_channels
        self.out_channels = out_channels

        self.model = nn.Sequential(
            nn.Linear(a_channels + b_channels, hidden_size), act_func,
            nn.Linear(hidden_size, out_channels)
        )

    def forward(self, a, b):
        assert a.ndim == 3 and b.ndim == 3, f"{a.shape}, {b.shape}"  # (batch_size, channels, n_samples)
        assert a.shape[1] == self.a_channels and b.shape[1] == self.b_channels, f"{a.shape}, {b.shape}"
        assert a.shape[0] == b.shape[0] and a.shape[2] == b.shape[2], f"{a.shape}, {b.shape}"

        batch_size = a.shape[0]

        cat = torch.cat([a, b], dim=1)
        cat = cat.permute(0, 2, 1).flatten(end_dim=1)

        out = self.model(cat)

        batches = out.chunk(batch_size)
        stacked = torch.stack(batches)
        result = stacked.permute(0, 2, 1)

        return result

if __name__ == '__main__':
    merge = IntelligentMerge(
        a_channels=2,
        b_channels=1,
        out_channels=1,
        hidden_size=64,
        act_func=nn.Tanh()
    )

    a_example = torch.randn(2, 2, 7)
    b_example = torch.randn(2, 1, 7)

    res = merge(a_example, b_example)

    print("a:")
    print(a_example, "\n")

    print("b:")
    print(b_example, "\n")

    print("res:")
    print(res)
    print(res.shape)
