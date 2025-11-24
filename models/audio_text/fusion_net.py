# models/audio_text/fusion_net.py

import torch
import torch.nn as nn

from models.audio_text.audio_net import AudioEncoder
from models.audio_text.text_net import TextEncoder


class SegmentAttentionPool(nn.Module):
    """
    세그먼트 차원(S)에 대해 attention pooling 수행
    입력: (B, S, D)
    출력: (B, D)
    """
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, S, D)
        u_t = torch.tanh(self.W(x))      # (B, S, D)
        att = self.u(u_t).squeeze(-1)    # (B, S)
        att = torch.softmax(att, dim=1)  # (B, S)
        att = att.unsqueeze(-1)          # (B, S, 1)
        out = (x * att).sum(dim=1)       # (B, D)
        return out


class AudioTextDepressionModel(nn.Module):
    """
    입력:
        audio: (B, S, L_a)   - S: num_segments, L_a: audio_len
        text_ids: (B, S, L_t)

    출력:
        pred: (B,)  - 회귀 값 (우울 점수)
    """
    def __init__(
        self,
        num_segments: int = 50,
        audio_encoder: AudioEncoder = None,
        text_encoder: TextEncoder = None,
        feat_dim: int = 512,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.feat_dim = feat_dim

        self.audio_encoder = audio_encoder if audio_encoder is not None else AudioEncoder(feat_dim)
        self.text_encoder = text_encoder if text_encoder is not None else TextEncoder(out_dim=feat_dim)

        # 세그먼트 단위 Attention Pooling
        self.seg_pool_audio = SegmentAttentionPool(feat_dim)
        self.seg_pool_text = SegmentAttentionPool(feat_dim)

        # 멀티모달 Fusion (concat -> 512 proj)
        self.fusion_fc = nn.Linear(feat_dim * 2, 512)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, audio, text_ids):
        """
        audio: (B, S, L_a)
        text_ids: (B, S, L_t)
        """
        B, S, L_a = audio.shape
        _, _, L_t = text_ids.shape

        # ---------- Audio Encoding ----------
        audio_flat = audio.view(B * S, L_a)                 # (B*S, L_a)
        audio_feat_flat = self.audio_encoder(audio_flat)    # (B*S, D)
        audio_feat = audio_feat_flat.view(B, S, -1)         # (B, S, D)

        # ---------- Text Encoding ----------
        text_flat = text_ids.view(B * S, L_t)               # (B*S, L_t)

        # pad_token_id 기반 attention_mask 생성
        pad_id = self.text_encoder.bert.config.pad_token_id
        attention_mask = (text_flat != pad_id).long()       # (B*S, L_t)

        # 모두 PAD인 경우 마스크 보정
        mask_sum = attention_mask.sum(dim=1)
        all_zero = (mask_sum == 0)
        if all_zero.any():
            attention_mask[all_zero] = 1

        text_feat_flat = self.text_encoder(
            input_ids=text_flat,
            attention_mask=attention_mask
        )                                                  # (B*S, D)

        text_feat = text_feat_flat.view(B, S, -1)          # (B, S, D)

        # ---------- Segment-level Attention Pooling ----------
        audio_global = self.seg_pool_audio(audio_feat)     # (B, D)
        text_global = self.seg_pool_text(text_feat)        # (B, D)

        # ---------- Fusion ----------
        fused = torch.cat([audio_global, text_global], dim=-1)  # (B, 2D)
        fused = self.fusion_fc(fused)                           # (B, 512)

        # ---------- Regression ----------
        out = self.regressor(fused).squeeze(-1)                 # (B,)

        return out
