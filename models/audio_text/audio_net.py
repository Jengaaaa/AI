import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """
    1D Conv Residual Block
    입력: (B, C, T)
    출력: (B, C_out, T)  (stride=1, padding=same이라 T 유지)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = F.relu(out)
        return out


class TemporalAttention(nn.Module):
    """
    시계열 차원(T)에 대해 attention pooling 수행
    입력: (B, T, D)
    출력: (B, D)
    """
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        u_t = torch.tanh(self.W(x))      # (B, T, D)
        att = self.u(u_t)                # (B, T, 1)
        att = torch.softmax(att, dim=1)  # (B, T, 1)
        out = (x * att).sum(dim=1)       # (B, D)
        return out


class AudioEncoder(nn.Module):
    """
    Raw waveform(16kHz) segment 하나를 인코딩해서 512차원 벡터로 변환
    입력:  x: (B, L_a)
    출력:  (B, feat_dim=512)
    """
    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim

        # 1D CNN (Residual Blocks)
        self.block1 = ResidualBlock1D(1, 64)
        self.block2 = ResidualBlock1D(64, 128)
        self.block3 = ResidualBlock1D(128, 256)
        self.block4 = ResidualBlock1D(256, 512)

        # Conv 뒤에 BiLSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )  # 출력: (B, T, 512)

        # 시간축 Attention + FC -> 512
        self.att = TemporalAttention(512)
        self.fc = nn.Linear(512, feat_dim)

    def forward(self, x):
        """
        x: (B, L_a)
        """
        # (B, 1, T)
        x = x.unsqueeze(1)

        # CNN feature 추출
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # (B, 512, T)

        # LSTM 입력을 위해 (B, T, C)
        x = x.transpose(1, 2)  # (B, T, 512)

        x, _ = self.lstm(x)    # (B, T, 512)

        # Attention pooling
        x = self.att(x)        # (B, 512)

        # 최종 투영
        x = self.fc(x)         # (B, feat_dim)
        return x
