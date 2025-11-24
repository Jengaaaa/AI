import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# 1D Residual Block
# ============================================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # If in/out channel mismatch → projection needed
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
                    if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.proj is not None:
            identity = self.proj(identity)

        out += identity
        return F.relu(out)


# ============================================
# Attention Layer (for feature weighting)
# ============================================
class AttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, 1)

    def forward(self, x):   # x: (batch, time, dim)
        u_t = torch.tanh(self.W(x))
        att = self.u(u_t)     # (batch, time, 1)
        att = F.softmax(att, dim=1)

        out = (x * att).sum(dim=1)   # weighted sum → (batch, dim)
        return out


# ============================================
# Encoder-Decoder BiLSTM
# ============================================
class EncoderDecoderBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.encoder1 = nn.LSTM(input_dim, 1024, bidirectional=True, batch_first=True)
        self.encoder2 = nn.LSTM(2048, 1024, bidirectional=True, batch_first=True)
        self.encoder3 = nn.LSTM(2048, 512, bidirectional=True, batch_first=True)

        self.att = AttentionLayer(1024)

        self.decoder = nn.LSTM(1024, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 512)

    def forward(self, x):
        # x: (batch, time, feat)
        x, _ = self.encoder1(x)
        x, _ = self.encoder2(x)
        x, _ = self.encoder3(x)

        x = self.att(x).unsqueeze(1)  # (batch, 1, 1024)

        x, _ = self.decoder(x)
        x = self.fc(x[:, -1, :])      # final output
        return x  # (batch, 512)


# ============================================
# Full Audio Encoder Model
# ============================================
class AudioEncoder(nn.Module):
    def __init__(self, input_sr=16000):
        super().__init__()

        # 1D CNN Spatial Feature Extractor
        self.res1 = ResidualBlock1D(1, 64)
        self.res2 = ResidualBlock1D(64, 128)
        self.res3 = ResidualBlock1D(128, 256)
        self.res4 = ResidualBlock1D(256, 512)

        # Final temporal dimension flatten
        self.att = AttentionLayer(512)

        self.encdec = EncoderDecoderBLSTM(512)
    
    def forward(self, x):
        # x shape: (batch, audio_len)
        x = x.unsqueeze(1)  # (batch, 1, len)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        # CNN output → (B, C, T) → transpose for LSTM
        x = x.transpose(1, 2)  # (batch, time, 512)

        x = self.att(x).unsqueeze(1)
        x = self.encdec(x)      # (batch, 512)

        return x
