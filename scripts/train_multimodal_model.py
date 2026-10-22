# scripts/train_multimodal_model.py
import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.app.models.multimodal_mci_net import (
    MultimodalMCIFusionNet,
    build_biomarker_vector,
    WEIGHTS_PATH
)
from backend.app.services.feature_service import extract_all_acoustic_features
from backend.app.services.transcription_service import transcribe_audio_bilingual
from backend.app.services.linguistic_service import extract_linguistic_biomarkers
from backend.app.services.scoring_service import calculate_mci_risk_score

DATASET_DIR = "dataset"
METADATA_FILE = os.path.join("dataset", "participants_metadata.csv")

LABEL_MAP = {
    "control": 0.0,
    "stress_drift": 0.5,
    "mci": 1.0
}


def load_label_lookup() -> dict:
    """Reads ground-truth clinical labels from dataset/participants_metadata.csv if available."""
    lookup = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fname = row.get("saved_filename")
                label_str = (row.get("clinical_label") or "control").lower()
                if fname:
                    lookup[fname] = LABEL_MAP.get(label_str, 0.0)
    return lookup


def prepare_training_samples():
    audio_files = [
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith((".wav", ".mp3", ".flac"))
    ]
    label_lookup = load_label_lookup()
    samples = []

    for fname in audio_files:
        fpath = os.path.join(DATASET_DIR, fname)
        print(f"Preparing training tensors for: {fname}")

        audio, sr = librosa.load(fpath, sr=None, mono=True)
        acoustic = extract_all_acoustic_features(audio, sr)
        asr = transcribe_audio_bilingual(fpath)
        ling = extract_linguistic_biomarkers(
            transcript=asr.get("transcript", ""),
            language=asr.get("detected_language", "en")
        )

        # Token tensor
        words = (asr.get("transcript") or "silence").lower().split()[:128]
        token_ids = [(abs(hash(w)) % 4998) + 1 for w in words] or [1]
        token_tensor = torch.tensor([token_ids], dtype=torch.long)

        # Mel-Spectrogram tensor
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_norm = (mel_db + 80.0) / 80.0
        mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 36-d Biomarker vector
        bio_vec = build_biomarker_vector(acoustic, ling)
        bio_tensor = torch.tensor([bio_vec], dtype=torch.float32)

        # Target label: Use ground-truth clinical label if in metadata, else heuristic teacher score
        if fname in label_lookup:
            target_val = label_lookup[fname]
        else:
            heuristic = calculate_mci_risk_score(acoustic, ling)
            target_val = heuristic["mci_risk_score"] / 100.0

        target_tensor = torch.tensor([[float(target_val)]], dtype=torch.float32)
        lang_code = "ta" if ling["language"] == "ta" else "en"

        samples.append((token_tensor, mel_tensor, bio_tensor, lang_code, target_tensor))

    return samples


def train_and_save(epochs: int = 25):
    samples = prepare_training_samples()
    if not samples:
        print("No audio files found in dataset/ to train on.")
        return

    torch.manual_seed(42)
    model = MultimodalMCIFusionNet()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print(f"\nTraining Multimodal MCI Network on {len(samples)} audio samples for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for token_t, mel_t, bio_t, lang, target_t in samples:
            optimizer.zero_grad()
            pred_prob, _ = model(token_t, mel_t, bio_t, language=lang)
            loss = criterion(pred_prob, target_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            avg_loss = total_loss / len(samples)
            print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\nModel weights saved successfully to: {WEIGHTS_PATH}")


if __name__ == "__main__":
    train_and_save()