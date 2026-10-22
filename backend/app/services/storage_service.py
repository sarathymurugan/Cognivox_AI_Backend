# backend/app/services/storage_service.py
import os
import csv
import json
import shutil
from datetime import datetime


SCREENING_LOG_FILE = os.path.join("features", "screening_logs.csv")
COLLECTED_DATASET_DIR = os.path.join("dataset", "collected_samples")
COLLECTED_METADATA_FILE = os.path.join("dataset", "participants_metadata.csv")


def log_screening_result(session_id: str, filename: str, task_type: str, result_payload: dict):
    """Appends every PHC/mobile screening result to features/screening_logs.csv."""
    os.makedirs("features", exist_ok=True)
    file_exists = os.path.exists(SCREENING_LOG_FILE)

    screening = result_payload.get("mci_screening", {})
    ling = result_payload.get("linguistic_biomarkers", {})
    pause = result_payload.get("features", {}).get("pause", {})
    js = result_payload.get("features", {}).get("jitter_shimmer", {})

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "filename": filename,
        "task_type": task_type,
        "detected_language": ling.get("language", "unknown"),
        "age": screening.get("metadata_used", {}).get("age"),
        "gender": screening.get("metadata_used", {}).get("gender"),
        "mci_risk_score": screening.get("mci_risk_score"),
        "risk_category": screening.get("risk_category"),
        "silence_percentage": round(float(pause.get("silence_percentage") or 0.0), 2),
        "longest_pause_sec": round(float(pause.get("longest_pause_seconds") or 0.0), 2),
        "jitter": js.get("jitter"),
        "shimmer": js.get("shimmer"),
        "filler_percentage": ling.get("filler_percentage"),
        "pronoun_percentage": ling.get("pronoun_percentage"),
        "lexical_diversity_ttr": ling.get("lexical_diversity_ttr")
    }

    with open(SCREENING_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_recent_screening_logs(limit: int = 50) -> list:
    """Returns the most recent screening logs for dashboard display."""
    if not os.path.exists(SCREENING_LOG_FILE):
        return []
    with open(SCREENING_LOG_FILE, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    return reader[-limit:][::-1]


def save_consented_dataset_sample(
    wav_file_path: str,
    participant_id: str,
    language: str,
    task_type: str,
    age: int = None,
    gender: str = None,
    mmse_score: float = None,
    clinical_label: str = "unlabeled"
) -> dict:
    """
    Saves consented participant audio into dataset/collected_samples/<language>/
    and logs ground-truth clinical labels (Control / Stress_Drift / MCI) for the >=500 KPI.
    """
    lang_folder = "tamil" if language == "ta" else "english"
    target_dir = os.path.join(COLLECTED_DATASET_DIR, lang_folder)
    os.makedirs(target_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_pid = "".join(c for c in participant_id if c.isalnum() or c in ("-", "_")) or "anon"
    saved_filename = f"{safe_pid}_{lang_folder}_{task_type}_{timestamp}.wav"
    saved_path = os.path.join(target_dir, saved_filename)

    shutil.copy2(wav_file_path, saved_path)

    # Also copy into root dataset/ folder so dataset_builder.py processes it automatically
    shutil.copy2(wav_file_path, os.path.join("dataset", saved_filename))

    meta_exists = os.path.exists(COLLECTED_METADATA_FILE)
    meta_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "participant_id": safe_pid,
        "saved_filename": saved_filename,
        "language": language,
        "task_type": task_type,
        "age": age,
        "gender": gender,
        "mmse_score": mmse_score,
        "clinical_label": clinical_label
    }

    with open(COLLECTED_METADATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=meta_row.keys())
        if not meta_exists:
            writer.writeheader()
        writer.writerow(meta_row)

    return {
        "saved_filename": saved_filename,
        "saved_path": saved_path,
        "metadata": meta_row
    }