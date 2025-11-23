import torch
import torch.nn as nn
import torch.nn.functional as F


class MFBFusion(nn.Module):
    """
    Multimodal Factorized Bilinear Pooling
    a: (B, Da)
    b: (B, Db)
    -> z: (B, out_dim)
    """
    def __init__(self, dim_a: int, dim_b: int, out_dim: int = 512, factor: int = 2, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        self.factor = factor

        self.proj_a = nn.Linear(dim_a, out_dim * factor)
        self.proj_b = nn.Linear(dim_b, out_dim * factor)

        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b):
        # a, b: (B, Da), (B, Db)
        xa = self.proj_a(a)   # (B, out_dim * k)
        xb = self.proj_b(b)   # (B, out_dim * k)

        x = xa * xb           # (B, out_dim * k)

        # sum pooling over factor dimension
        B = x.size(0)
        x = x.view(B, self.out_dim, self.factor)
        x = x.sum(dim=2)      # (B, out_dim)

        # signed sqrt + L2 norm (자주 쓰는 안정화)
        x = torch.sign(x) * torch.sqrt(torch.clamp(x.abs(), min=1e-8))
        x = F.normalize(x, dim=1)

        x = self.dropout(x)
        return x
