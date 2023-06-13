import torch
import torch.nn as nn

class GBiasReg(nn.Module):
    def __init__(self, in_channels, buffer_size, n_layers):
        super().__init__()
        self.buffer_size = buffer_size

        layers = []
        for i in range(n_layers):
            in_ch = in_channels if i == 0 else 1
            layers.extend(
                [nn.Conv1d(in_channels=in_ch, out_channels=1, kernel_size=13, padding=6, dilation=1),
                 nn.ReLU(), nn.MaxPool1d(2)])
        self.feature_extractor = nn.Sequential(*layers)
        self.extractor_out_size = self.buffer_size // 2 ** n_layers

        # self.classifier = nn.Sequential(
        #     nn.Linear(self.extractor_out_size, 1), nn.Sigmoid()
        # )

    def forward(self, x):
        features = self.feature_extractor(x).flatten(start_dim=1)
        # global_bias_percentage = self.classifier(features)
        # global_bias_samples = int(global_bias_percentage * self.buffer_size)
        return features


if __name__ == '__main__':
    reg = GBiasReg(9, 48000, 9)

    inp = torch.randn(32, 9, 48000)
    print(reg(inp).shape)