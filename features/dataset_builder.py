import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

import csv
import librosa

from backend.app.services.feature_service import (extract_mfcc, extract_pitch, extract_rms_energy, extract_zcr, extract_pause_features, extract_speech_rate, extract_jitter_shimmer)


DATASET_DIR = "dataset"
OUTPUT_DIR = "features"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "dataset_features.csv"
)


def process_audio(file_path):

    print(f"Processing: {file_path}")

    # Load audio once
    audio, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    # Extract features
    mfcc = extract_mfcc(
        audio,
        sample_rate
    )

    pitch = extract_pitch(
        audio,
        sample_rate
    )

    rms = extract_rms_energy(
        audio
    )

    zcr = extract_zcr(
        audio
    )

    pause = extract_pause_features(
        audio,
        sample_rate
    )

    speech_rate = extract_speech_rate(
        audio,
        sample_rate
    )

    jitter_shimmer = extract_jitter_shimmer(
        audio,
        sample_rate
    )

    # Create one flat dictionary
    features = {}

    # File information
    features["file_name"] = os.path.basename(file_path)

    # MFCC mean
    for i, value in enumerate(
        mfcc["mean"],
        start=1
    ):
        features[f"mfcc_{i}_mean"] = value

    # MFCC standard deviation
    for i, value in enumerate(
        mfcc["std"],
        start=1
    ):
        features[f"mfcc_{i}_std"] = value

    # Pitch
    features["pitch_mean_hz"] = pitch["mean_hz"]
    features["pitch_min_hz"] = pitch["min_hz"]
    features["pitch_max_hz"] = pitch["max_hz"]
    features["pitch_std_hz"] = pitch["std_hz"]

    # RMS
    features["rms_mean"] = rms["mean"]
    features["rms_min"] = rms["min"]
    features["rms_max"] = rms["max"]
    features["rms_std"] = rms["std"]

    # ZCR
    features["zcr_mean"] = zcr["mean"]
    features["zcr_min"] = zcr["min"]
    features["zcr_max"] = zcr["max"]
    features["zcr_std"] = zcr["std"]

    # Pause
    features["pause_count"] = pause["pause_count"]
    features["total_pause_duration"] = (
        pause["total_pause_duration_seconds"]
    )
    features["average_pause_duration"] = (
        pause["average_pause_duration_seconds"]
    )
    features["longest_pause"] = (
        pause["longest_pause_seconds"]
    )
    features["silence_percentage"] = (
        pause["silence_percentage"]
    )

    # Speech activity
    features["total_duration"] = (
        speech_rate["total_duration_seconds"]
    )
    features["speech_duration"] = (
        speech_rate["estimated_speech_duration_seconds"]
    )
    features["estimated_pause_duration"] = (
        speech_rate["estimated_pause_duration_seconds"]
    )
    features["speech_percentage"] = (
        speech_rate["speech_percentage"]
    )
    features["pause_percentage"] = (
        speech_rate["pause_percentage"]
    )
    features["speech_pause_ratio"] = (
        speech_rate["speech_to_pause_ratio"]
    )

    # Jitter / Shimmer
    features["jitter"] = (
        jitter_shimmer["jitter"]
    )
    features["shimmer"] = (
        jitter_shimmer["shimmer"]
    )

    return features


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    audio_files = []

    for filename in os.listdir(DATASET_DIR):

        if filename.lower().endswith(
            (".wav", ".mp3", ".flac")
        ):
            audio_files.append(filename)

    if len(audio_files) == 0:

        print(
            "No audio files found in dataset folder."
        )

        return

    all_features = []

    for filename in audio_files:

        file_path = os.path.join(
            DATASET_DIR,
            filename
        )

        try:

            features = process_audio(
                file_path
            )

            all_features.append(
                features
            )

        except Exception as e:

            print(
                f"Error processing {filename}: {e}"
            )

    if len(all_features) == 0:

        print(
            "No features were extracted."
        )

        return

    # Use keys from first record
    fieldnames = all_features[0].keys()

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            all_features
        )

    print()
    print(
        "Feature dataset created successfully."
    )

    print(
        f"Files processed: {len(all_features)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()