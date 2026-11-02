# import librosa
# import librosa.display
# import soundfile as sf
# import matplotlib.pyplot as plt
# import os


# audio, sample_rate = librosa.load(
#     "dataset/mysample1.wav",
#     sr=None
# )

# print(audio)
# print(len(audio))
# print("Shape :", audio.shape)
# print(sample_rate)

# # audio trimming
# trimmed_audio, index = librosa.effects.trim(audio)

# print("Original samples :", len(audio))
# print("Trimmed samples :", len(trimmed_audio))

# base_path = "dataset/trimmed_audio/trimmed.wav"

# if not os.path.exists(base_path):
#     output_path = base_path
# else:
#     counter = 1
#     while True:
#         output_path = f"dataset/trimmed_audio/trimmed{counter}.wav"
#         if not os.path.exists(output_path):
#             break
#         counter += 1

# sf.write(
#     output_path,
#     trimmed_audio,
#     sample_rate
# )

# plt.figure(figsize=(14,4))
# plt.subplot(1,2,1)
# plt.plot(audio)
# plt.xlabel("Samples")
# plt.ylabel("Amplitude")
# plt.title("Original")
# plt.subplot(1,2,2)
# plt.plot(trimmed_audio)
# plt.xlabel("Samples")
# plt.ylabel("Amplitude")
# plt.title("Trimmed")
# plt.tight_layout()
# plt.show()


# def analyze_audio(file_path: str):
#     # Load audio
#     audio, sample_rate = librosa.load(
#         file_path,
#         sr=None,
#         mono=True
#     )

#     # Calculate duration
#     duration = len(audio) / sample_rate

#     # Return audio information
#     return {
#         "sample_rate": sample_rate,
#         "duration_seconds": round(duration, 2),
#         "number_of_samples": len(audio),
#         "channels": 1
#     }






import os
import subprocess
import librosa
import numpy as np
from fastapi import HTTPException


def convert_webm_to_wav(input_path: str, output_path: str):
    """Converts browser-recorded WebM or other formats into 48kHz Mono WAV."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "48000",
                output_path
            ],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as error:
        print("FFmpeg conversion failed:")
        print(error.stderr)
        raise HTTPException(
            status_code=500,
            detail="Unable to convert audio to WAV."
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg is not installed or not found in system PATH."
        )


def load_and_preprocess_audio(file_path: str, trim_silence: bool = True):
    """
    Loads audio in mono, normalizes amplitude, and optionally trims
    leading/trailing silence (keeps internal speech pauses intact).
    """
    audio, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    if len(audio) == 0 or sample_rate <= 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio file is empty."
        )

    # 1. RAW SILENCE GATE (Before normalization amplifies background noise!)
    raw_peak = float(np.max(np.abs(audio)))
    raw_rms = float(np.sqrt(np.mean(audio ** 2)))

    if raw_peak < 0.015 or raw_rms < 0.002:
        raise HTTPException(
            status_code=400,
            detail="No voice detected in recording (audio is silent or too quiet). Please speak clearly into the microphone."
        )

    # 2. Trim leading and trailing silence (top_db=25)
    if trim_silence:
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=25)
        trimmed_duration = len(trimmed_audio) / sample_rate

        # Reject if active sound after trimming is shorter than 1.5 seconds
        if trimmed_duration < 1.5:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient speech detected ({trimmed_duration:.1f}s of sound). Please record at least 3 seconds of continuous speech."
            )
        audio = trimmed_audio

    # 3. Safe Peak Normalization (only runs after confirming real speech exists)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio, sample_rate


def analyze_audio(file_path: str):
    """Returns basic metadata about the audio file."""
    audio, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True
    )
    duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

    return {
        "sample_rate": int(sample_rate),
        "duration_seconds": round(float(duration), 2),
        "number_of_samples": int(len(audio)),
        "channels": 1
    }