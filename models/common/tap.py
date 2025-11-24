import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentivePooling(nn.Module):
    """
    x: (B, T, D) -> (B, D)
    T: time/segment dimension
    D: feature dim
    """
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        h = torch.tanh(self.W(x))      # (B, T, D)
        att = self.u(h)                # (B, T, 1)
        att = F.softmax(att, dim=1)    # (B, T, 1)
        out = (x * att).sum(dim=1)     # (B, D)
        return out
