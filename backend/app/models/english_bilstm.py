# backend/app/models/english_bilstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Bahdanau Additive Attention mechanism.
    Used for English BiLSTM text modeling, Audio CNN frames, and Multimodal Fusion.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: (batch, seq_len, hidden_dim)
        energy = torch.tanh(self.W(hidden_states))
        scores = self.v(energy).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, attn_weights


class EnglishBiLSTMModel(nn.Module):
    """
    English Text-Based Model (Proposal Figure 1 - Top Branch):
    - BiLSTM for contextual sequence modeling
    - Additive Attention for filler words & pronoun usage
    - Latent Topic Coherence layer for semantic organization
    """
    def __init__(
        self,
        vocab_size: int = 5000,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_topics: int = 8
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
        self.attention = AdditiveAttention(hidden_dim)

        # Latent Topic Modeling head (captures semantic coherence alongside syntax)
        self.topic_head = nn.Sequential(
            nn.Linear(hidden_dim, num_topics),
            nn.Softmax(dim=-1)
        )
        self.topic_proj = nn.Linear(hidden_dim + num_topics, hidden_dim)

    def forward(self, token_ids: torch.Tensor):
        # token_ids: (batch, seq_len)
        embedded = self.embedding(token_ids)
        lstm_out, _ = self.bilstm(embedded)  # (batch, seq_len, hidden_dim)
        context, attn_weights = self.attention(lstm_out)

        # Combine syntactic context with latent topic distribution
        topic_dist = self.topic_head(context)
        combined = torch.cat([context, topic_dist], dim=-1)
        output_repr = F.relu(self.topic_proj(combined))  # (batch, hidden_dim)

        return output_repr, attn_weights