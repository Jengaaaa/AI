import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class AttentionLayer(nn.Module):
    """
    시퀀스(hidden states)에 대해 가중합을 구하는 Attention 레이어
    입력: (B, T, D)
    출력: (B, D)
    """
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        """
        x: (B, T, D)
        mask: (B, T)  (1 = 유효 토큰, 0 = 패딩) 또는 None
        """
        u_t = torch.tanh(self.W(x))      # (B, T, D)
        att = self.u(u_t).squeeze(-1)    # (B, T)

        if mask is not None:
            # 패딩 위치에 -inf 가중치 적용
            att = att.masked_fill(mask == 0, float('-inf'))

        att = torch.softmax(att, dim=1)  # (B, T)
        att = att.unsqueeze(-1)          # (B, T, 1)

        out = (x * att).sum(dim=1)       # (B, D)
        return out


class TextEncoder(nn.Module):
    """
    BERT 기반 텍스트 인코더
    - BERT last_hidden_state
    - (옵션) BiLSTM
    - Attention
    - FC -> 512차원
    """
    def __init__(
        self,
        model_name: str = "klue/bert-base",
        out_dim: int = 512,
        use_lstm: bool = True
    ):
        super().__init__()

        # 1) Pretrained BERT 로드
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.use_lstm = use_lstm

        # 2) BiLSTM (선택)
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            att_dim = hidden_size  # 양방향 합쳐서 다시 hidden_size
        else:
            self.lstm = None
            att_dim = hidden_size

        # 3) Attention + FC
        self.att = AttentionLayer(att_dim)
        self.fc = nn.Linear(att_dim, out_dim)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (B, L)
        attention_mask: (B, L) or None
        return: (B, out_dim)
        """

        # attention_mask가 전부 0인 경우(빈 세그먼트) 방지
        if attention_mask is not None:
            # 배치별로 모두 0인 경우 1로 바꿔줌
            mask_sum = attention_mask.sum(dim=1)
            all_zero = (mask_sum == 0)
            if all_zero.any():
                attention_mask = attention_mask.clone()
                attention_mask[all_zero] = 1

        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        seq_output = bert_out.last_hidden_state  # (B, L, H)

        if self.use_lstm:
            seq_output, _ = self.lstm(seq_output)  # (B, L, H)

        # Attention pooling
        pooled = self.att(seq_output, mask=attention_mask)  # (B, H)

        # 512차원으로 투영
        out = self.fc(pooled)          # (B, out_dim)
        return out
