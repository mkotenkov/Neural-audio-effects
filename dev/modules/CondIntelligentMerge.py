import torch
from torch import nn


class CondIntelligentMerge(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, act_func, cond_size):
        super().__init__()
        self.in_channels = in_channels
        self.cond_size = cond_size

        self.model = nn.Sequential(
            nn.Linear(in_channels + cond_size, hidden_size), act_func,
            nn.Linear(hidden_size, out_channels)
        )

    def forward(self, x, cond):
        assert x.ndim == 3, f"{x.shape}"  # (batch_size, channels, n_samples)
        assert x.shape[1] == self.in_channels, f"{x.shape}"
        assert cond.ndim == 2, cond.shape  # (batch_size, cond_size)
        assert cond.shape[1] == self.cond_size, f"{cond.shape[1]} != {self.cond_size}"

        batch_size = x.shape[0]
        n_samples = x.shape[-1]

        cond = torch.stack([cond] * n_samples).permute(1, 2, 0)
        cat = torch.cat([x, cond], dim=1)
        cat = cat.permute(0, 2, 1).flatten(end_dim=1)

        out = self.model(cat)

        batches = out.chunk(batch_size)
        stacked = torch.stack(batches)
        result = stacked.permute(0, 2, 1)

        return result


if __name__ == '__main__':
    merge = CondIntelligentMerge(
        in_channels=3,
        out_channels=3,
        hidden_size=64,
        act_func=nn.Tanh(),
        cond_size=3
    )

    a_example = torch.randn(2, 1, 7)
    b_example = torch.randn(2, 2, 7)
    x = torch.cat([a_example, b_example], dim=1)
    cond = torch.Tensor([[1, 1, 1],
                         [2, 3, 4]])

    res = merge(x, cond)

    print("a:")
    print(a_example, "\n")

    print("b:")
    print(b_example, "\n")

    print("res:")
    # print(res)
    print(res.shape)
