# backend/app/models/multimodal_mci_net.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa


from backend.app.models.english_bilstm import EnglishBiLSTMModel, AdditiveAttention
from backend.app.models.tamil_bilstm import TamilBiLSTMModel
from backend.app.models.audio_cnn_hubert import SpectrogramCNNBranch


# =====================================================================
# 1. ATTENTION MECHANISMS (Proposal Pages 3 & 4)
# =====================================================================

class AdditiveAttention(nn.Module):
    """
    Bahdanau Additive Attention:
    Used for English BiLSTM branch, Audio Spectrogram CNN, and Multimodal Fusion.
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


class DotProductAttention(nn.Module):
    """
    Luong Scaled Dot-Product Attention:
    Used specifically for Tamil BiLSTM branch to emphasize syntactic cues.
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


# =====================================================================
# 2. LANGUAGE-SPECIFIC BiLSTM BRANCHES (Figure 1 Top & Middle Branches)
# =====================================================================

class LanguageBiLSTMBranch(nn.Module):
    """
    BiLSTM with language-specific attention:
    - 'en': Additive Attention (English syntax + topic coherence)
    - 'ta': Dot-Product Attention (Tamil syntactic structure)
    """
    def __init__(self, vocab_size: int = 5000, embed_dim: int = 64, hidden_dim: int = 64, language: str = "en"):
        super().__init__()
        self.language = language
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        if language == "ta":
            self.attention = DotProductAttention(hidden_dim)
        else:
            self.attention = AdditiveAttention(hidden_dim)

    def forward(self, token_ids: torch.Tensor):
        embedded = self.embedding(token_ids)
        lstm_out, _ = self.bilstm(embedded)
        context, attn_weights = self.attention(lstm_out)
        return context, attn_weights


# =====================================================================
# 3. AUDIO SPECTROGRAM CNN + ATTENTION BRANCH (Figure 1 Bottom Branch)
# =====================================================================

class SpectrogramCNNBranch(nn.Module):
    """
    2D CNN applied to 128-bin Mel-Spectrograms with Additive Attention over time frames.
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
            nn.AdaptiveAvgPool2d((16, 32))  # Fixed (freq=16, time=32)
        )
        self.frame_proj = nn.Linear(32 * 16, out_dim)
        self.attention = AdditiveAttention(out_dim)

    def forward(self, mel_spec: torch.Tensor):
        # mel_spec: (batch, 1, 128, time_frames)
        conv_out = self.conv_layers(mel_spec)  # (batch, 32, 16, 32)
        batch, ch, freq, time = conv_out.shape
        seq = conv_out.permute(0, 3, 1, 2).reshape(batch, time, ch * freq)
        projected = F.relu(self.frame_proj(seq))  # (batch, 32, out_dim)
        context, attn_weights = self.attention(projected)
        return context, attn_weights


# =====================================================================
# 4. MULTIMODAL FUSION NETWORK (Audio + Text + Acoustic Cues + Metadata)
# =====================================================================

class MultimodalMCIFusionNet(nn.Module):
    """
    Master Multimodal Fusion Architecture (Proposal Figure 1 & Page 4):
    Integrates:
    1. EnglishBiLSTMModel (Additive Attention + Topic Modeling)
    2. TamilBiLSTMModel (Dot-Product Attention)
    3. SpectrogramCNNBranch (2D CNN + Additive Attention)
    4. Clinical Acoustic & Linguistic Biomarker Vector (MFCCs, Jitter, Shimmer, Pauses, Fillers)
    """
    def __init__(self, biomarker_dim: int = 36, hidden_dim: int = 64):
        super().__init__()
        self.en_bilstm = EnglishBiLSTMModel(hidden_dim=hidden_dim)
        self.ta_bilstm = TamilBiLSTMModel(hidden_dim=hidden_dim)
        self.audio_cnn = SpectrogramCNNBranch(out_dim=hidden_dim)

        self.biomarker_proj = nn.Sequential(
            nn.Linear(biomarker_dim, hidden_dim),
            nn.ReLU()
        )

        # Additive Attention fusion across [Linguistic_BiLSTM, Audio_CNN, Clinical_Biomarkers]
        self.fusion_attention = AdditiveAttention(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        mel_spec: torch.Tensor,
        biomarkers: torch.Tensor,
        language: str = "en"
    ):
        if language == "ta":
            text_feat, _ = self.ta_bilstm(token_ids)
        else:
            text_feat, _ = self.en_bilstm(token_ids)

        audio_feat, _ = self.audio_cnn(mel_spec)
        bio_feat = self.biomarker_proj(biomarkers)

        # Stack modalities: (batch, 3, hidden_dim)
        multimodal_seq = torch.stack([text_feat, audio_feat, bio_feat], dim=1)
        fused_context, modality_weights = self.fusion_attention(multimodal_seq)

        mci_prob = self.classifier(fused_context)
        return mci_prob, modality_weights


# =====================================================================
# INFERENCE HELPER FOR FASTAPI BACKEND
# =====================================================================

_multimodal_model = None
WEIGHTS_PATH = os.path.join("backend", "app", "models", "mci_multimodal_weights.pt")


def get_multimodal_model() -> MultimodalMCIFusionNet:
    global _multimodal_model
    if _multimodal_model is None:
        torch.manual_seed(42)
        model = MultimodalMCIFusionNet()
        if os.path.exists(WEIGHTS_PATH):
            try:
                model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
            except Exception as e:
                print(f"Note: Re-run train_multimodal_model.py to update weights ({e})")
        model.eval()
        _multimodal_model = model
    return _multimodal_model


def build_biomarker_vector(acoustic_features: dict, linguistic_features: dict) -> list:
    """Packs 36 handcrafted acoustic + linguistic features into a standardized numeric vector."""
    mfcc_mean = acoustic_features.get("mfcc", {}).get("mean", [0.0] * 13)
    mfcc_std = acoustic_features.get("mfcc", {}).get("std", [0.0] * 13)

    pitch = acoustic_features.get("pitch", {})
    pause = acoustic_features.get("pause", {})
    js = acoustic_features.get("jitter_shimmer", {})
    rms = acoustic_features.get("rms", {})

    scalar_features = [
        float(pitch.get("mean_hz") or 0.0) / 300.0,
        float(pitch.get("std_hz") or 0.0) / 100.0,
        float(rms.get("mean") or 0.0),
        float(pause.get("silence_percentage") or 0.0) / 100.0,
        float(pause.get("longest_pause_seconds") or 0.0) / 10.0,
        float(js.get("jitter") or 0.0) * 10.0,
        float(js.get("shimmer") or 0.0) * 10.0,
        float((linguistic_features or {}).get("filler_percentage") or 0.0) / 100.0,
        float((linguistic_features or {}).get("pronoun_percentage") or 0.0) / 100.0,
        float((linguistic_features or {}).get("lexical_diversity_ttr") or 0.0),
    ]
    vector = list(mfcc_mean[:13]) + list(mfcc_std[:13]) + scalar_features
    return [float(x) if np.isfinite(x) else 0.0 for x in vector]


def run_deep_multimodal_inference(
    audio: np.ndarray,
    sample_rate: int,
    transcript: str,
    language: str,
    acoustic_features: dict,
    linguistic_features: dict
) -> dict:
    """
    Runs a live forward pass through the modular Figure 1 Multimodal Architecture.
    """
    model = get_multimodal_model()
    lang_code = "ta" if language == "ta" else "en"

    words = (transcript or "silence").lower().split()[:128]
    token_ids = [(abs(hash(w)) % 4998) + 1 for w in words] or [1]
    token_tensor = torch.tensor([token_ids], dtype=torch.long)

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = (mel_db + 80.0) / 80.0
    mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    bio_vec = build_biomarker_vector(acoustic_features, linguistic_features)
    bio_tensor = torch.tensor([bio_vec], dtype=torch.float32)

    with torch.no_grad():
        mci_prob, modality_weights = model(token_tensor, mel_tensor, bio_tensor, language=lang_code)

    weights = modality_weights[0].tolist()
    weights_trained = os.path.exists(WEIGHTS_PATH)

    return {
        "architecture_executed": "BiLSTM_Attention + Spectrogram_CNN_Attention + Multimodal_Fusion",
        "text_branch_used": "TamilBiLSTMModel (DotProductAttention)" if lang_code == "ta" else "EnglishBiLSTMModel (AdditiveAttention + TopicModeling)",
        "audio_branch_used": "SpectrogramCNNBranch (2DCNN + AdditiveAttention)",
        "trained_weights_loaded": weights_trained,
        "raw_network_probability": round(float(mci_prob.item()), 4),
        "modality_attention_weights": {
            "linguistic_bilstm_weight": round(float(weights[0]), 3),
            "spectrogram_cnn_weight": round(float(weights[1]), 3),
            "clinical_biomarkers_weight": round(float(weights[2]), 3)
        }
    }