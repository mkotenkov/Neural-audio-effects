"""maybe some problems here (maybe not)"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveLearner(nn.Module):
    def __init__(self, n_cores, n_channels, out_size):
        super().__init__()
        self.out_size = out_size
        self.n_channels = n_channels
        self.cores = nn.Parameter(torch.zeros(n_channels, n_cores, requires_grad=True))

    def forward(self):
        cores = self.cores.reshape(1, self.n_channels, -1)
        result =  F.interpolate(cores, size=self.out_size, mode="linear", align_corners=False)[0]
        return result
