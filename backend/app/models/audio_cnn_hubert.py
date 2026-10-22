# backend/app/models/audio_cnn_hubert.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.app.models.english_bilstm import AdditiveAttention


class SpectrogramCNNBranch(nn.Module):
    """
    Audio-Based Model (Proposal Figure 1 - Bottom Branch & Page 4):
    - 2D CNN applied to 128-bin Mel-Spectrograms
    - Frame-level phonetic/prosodic embedding projection
    - Additive Attention mechanism over acoustic frames
    """
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 32))  # Standardizes to (freq=16, time_frames=32)
        )
        self.frame_proj = nn.Linear(32 * 16, out_dim)
        self.attention = AdditiveAttention(out_dim)

    def forward(self, mel_spec: torch.Tensor):
        # mel_spec: (batch, 1, 128, time_frames)
        conv_out = self.conv_layers(mel_spec)  # (batch, 32, 16, 32)
        batch, ch, freq, time = conv_out.shape

        # Reshape into frame-level sequence: (batch, time_frames=32, features=512)
        seq = conv_out.permute(0, 3, 1, 2).reshape(batch, time, ch * freq)
        frame_embeddings = F.relu(self.frame_proj(seq))  # (batch, 32, out_dim)

        # Apply Additive Attention over acoustic frames
        context, attn_weights = self.attention(frame_embeddings)
        return context, attn_weights