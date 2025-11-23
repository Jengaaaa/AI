# models/audio_text/fusion_net.py

import torch
import torch.nn as nn

from models.audio_text.audio_net import AudioEncoder
from models.audio_text.text_net import TextEncoder
from models.common.tap import TemporalAttentivePooling
from models.common.mfb import MFBFusion


class AudioTextDepressionModel(nn.Module):
    """
    입력:
        audio: (B, S, L_a)   - S: num_segments, L_a: audio_len
        text_ids: (B, S, L_t)
        text_attn_mask: (B, S, L_t) 또는 None

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

        self.audio_encoder = audio_encoder if audio_encoder is not None else AudioEncoder()
        self.text_encoder = text_encoder if text_encoder is not None else TextEncoder()

        self.tap_audio = TemporalAttentivePooling(feat_dim)
        self.tap_text = TemporalAttentivePooling(feat_dim)

        self.mfb = MFBFusion(dim_a=feat_dim, dim_b=feat_dim, out_dim=512, factor=2, dropout=0.1)

        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, audio, text_ids, text_attn_mask=None):
        """
        audio: (B, S, L_a)
        text_ids: (B, S, L_t)
        text_attn_mask: (B, S, L_t) or None
        """
        B, S, L_a = audio.shape
        _, _, L_t = text_ids.shape

        # -------- Audio Encoding --------
        audio_flat = audio.view(B * S, L_a)             # (B*S, L_a)
        audio_feat_flat = self.audio_encoder(audio_flat)   # (B*S, 512)
        audio_feat = audio_feat_flat.view(B, S, -1)     # (B, S, 512)

        # -------- Text Encoding --------
        text_flat = text_ids.view(B * S, L_t)           # (B*S, L_t)

        if text_attn_mask is not None:
            mask_flat = text_attn_mask.view(B * S, L_t)
        else:
            # pad_token_id 가져오기
            pad_id = self.text_encoder.bert.config.pad_token_id
            mask_flat = (text_flat != pad_id).long()

        text_feat_flat = self.text_encoder(text_flat, attention_mask=mask_flat)  # (B*S, 512)
        text_feat = text_feat_flat.view(B, S, -1)                                # (B, S, 512)

        # -------- TAP: Segment-level -> Global --------
        audio_global = self.tap_audio(audio_feat)   # (B, 512)
        text_global = self.tap_text(text_feat)      # (B, 512)

        # -------- MFB Fusion --------
        fused = self.mfb(audio_global, text_global)  # (B, 512)

        # -------- Regression --------
        out = self.regressor(fused).squeeze(-1)      # (B,)

        return out
