import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

from modules.TCN import TCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
        x, y,
        n_iters=1000,
        lr=0.002,  # 0.002
        slice_len=240000,
        dilation_growth=10,
        n_layers=3,
        n_channels=17,
        info_channels=20,
        cond_size=1,
        dilations=None,
        channels=None,
        stereo=False
):
    slice_len = min(x.shape[-1], slice_len)
    x = x[None, :, :] if stereo else x[None, 0:1, :]
    y = y[None, :, :] if stereo else y[None, 0:1, :]

    # build the model
    model = TCN(
        wave_learner_params=dict(
            n_waves=info_channels,
            n_channels=2,
            cond_size=cond_size,
            buffer_size=slice_len,
            min_freq=1,
            max_freq=20
        ),
        channels=[n_channels] * n_layers if channels is None else channels,
        dilations=[dilation_growth ** idx for idx in range(n_layers)] if dilations is None else dilations,
        in_ch=2 if stereo else 1,
        out_ch=2 if stereo else 1)

    # print info
    rf = model.calc_receptive_field()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params * 1e-3:0.3f} k")
    print(f"Receptive field: {rf} samples or {(rf / sample_rate) * 1e3:0.1f} ms")

    # train
    map(lambda item: item.to(device), [x, y, model])
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=1e-10)
    losses = []
    for _ in trange(n_iters):
        optimizer.zero_grad()
        start_idx = torch.randint(0, x.shape[-1] - slice_len, (1,))[0]
        x_crop = x[..., start_idx:start_idx + slice_len]
        y_crop = y[..., start_idx:start_idx + slice_len]
        y_hat = model(x_crop, torch.ones(1, cond_size, device=device))
        loss = F.mse_loss(y_hat[..., rf:], y_crop[..., rf:])
        loss.backward()
        losses.append(loss.detach().cpu())
        optimizer.step()
        scheduler.step()
    plt.title("Training loss over iterations")
    plt.plot(losses)
    return model
