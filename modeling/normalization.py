import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        ch = y.size(1)
        sigma, mu = torch.split(y.unsqueeze(-1).unsqueeze(-1), [ch // 2, ch // 2], dim=1)

        x_mu = x.mean(dim=[2, 3], keepdim=True)
        x_sigma = x.std(dim=[2, 3], keepdim=True)

        return sigma * ((x - x_mu) / x_sigma) + mu


class EFDM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x, y):
        """
            x: Content feature maps, dim: (batch, channel, height, width)
            y: Projected style information, dim: (batch, channel, height * width)
        """
        assert x.size() == y.size(), f"x and y should have the same size, but x: {x.size()}, y: {y.size()}"
        B, C, W, H = x.size()
        
        _, idx_x = torch.sort(x.view(B, C, -1))  # sort conduct a deep copy here.
        val_y, _ = torch.sort(y.view(B, C, -1))  # sort conduct a deep copy here.
        inverse_idx = idx_x.argsort(-1)
        
        out_x = x.view(B, C, -1) + (val_y.gather(-1, inverse_idx) - x.view(B, C, -1).detach())
        
        return out_x.view(B, C, W, H)
