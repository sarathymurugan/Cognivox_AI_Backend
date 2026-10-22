# backend/app/models/tamil_bilstm.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    Specifically chosen for Tamil data to emphasize syntactic elements
    within Tamil agglutinative sentence structure.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.scale = math.sqrt(hidden_dim)

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: (batch, seq_len, hidden_dim)
        batch_size = hidden_states.size(0)
        q = self.query.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        scores = torch.bmm(q, hidden_states.transpose(1, 2)).squeeze(1) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, attn_weights


class TamilBiLSTMModel(nn.Module):
    """
    Tamil Text-Based Model (Proposal Figure 1 - Middle Branch):
    - BiLSTM for Tamil word & morphology sequences
    - Dot-Product Attention for Tamil filler words and pronoun usage
    """
    def __init__(
        self,
        vocab_size: int = 5000,
        embed_dim: int = 64,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.attention = DotProductAttention(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_ids: torch.Tensor):
        # token_ids: (batch, seq_len)
        embedded = self.embedding(token_ids)
        lstm_out, _ = self.bilstm(embedded)  # (batch, seq_len, hidden_dim)
        context, attn_weights = self.attention(lstm_out)
        output_repr = self.norm(context)  # (batch, hidden_dim)

        return output_repr, attn_weights