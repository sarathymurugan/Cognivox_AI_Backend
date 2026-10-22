import librosa
import numpy as np
import parselmouth


def extract_mfcc(audio, sample_rate: int):

    # Extract 13 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    # Calculate summary statistics
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return {
        "number_of_coefficients": int(mfcc.shape[0]),
        "number_of_frames": int(mfcc.shape[1]),
        "shape": list(mfcc.shape),
        "mean": mfcc_mean.tolist(),
        "std": mfcc_std.tolist()
    }



def extract_pitch(audio, sample_rate: int):
    # Estimate fundamental frequency and identify voiced frames
    f0, voiced_flag, voiced_probability = librosa.pyin(
        audio,
        fmin=75,
        fmax=300,
        sr=sample_rate,
        frame_length=2048,
        hop_length=512
    )

    # Keep only valid pitch values from frames identified as voiced
    valid_mask = (
        voiced_flag
        & np.isfinite(f0)
        & (f0 >= 75)
        & (f0 <= 300)
    )

    valid_pitch = f0[valid_mask]
    valid_probability = voiced_probability[valid_mask]

    if len(valid_pitch) == 0:
        return {
            "mean_hz": None,
            "min_hz": None,
            "max_hz": None,
            "std_hz": None,
            "voiced_frames": 0,
            "voiced_percentage": 0.0,
            "mean_voiced_probability": None
        }

    return {
        "mean_hz": float(np.mean(valid_pitch)),
        "min_hz": float(np.min(valid_pitch)),
        "max_hz": float(np.max(valid_pitch)),
        "std_hz": float(np.std(valid_pitch)),
        "voiced_frames": int(np.sum(valid_mask)),
        "voiced_percentage": float(
            np.mean(valid_mask) * 100
        ),
        "mean_voiced_probability": float(
            np.mean(valid_probability)
        )
    }

def extract_rms_energy(audio):

    # Calculate RMS energy
    rms = librosa.feature.rms(y=audio)[0]

    return {
        "mean": float(np.mean(rms)),
        "min": float(np.min(rms)),
        "max": float(np.max(rms)),
        "std": float(np.std(rms))
    }


def extract_zcr(audio):

    zcr = librosa.feature.zero_crossing_rate(
        y=audio
    )[0]

    return {
        "mean": float(np.mean(zcr)),
        "min": float(np.min(zcr)),
        "max": float(np.max(zcr)),
        "std": float(np.std(zcr))
    }


def _calculate_speech_pause_metrics(audio, sample_rate):
    total_duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

    if total_duration <= 0:
        return {
            "total_duration": 0.0,
            "speech_duration": 0.0,
            "pause_duration": 0.0,
            "speech_percentage": 0.0,
            "pause_percentage": 0.0,
            "speech_pause_ratio": None,
            "pause_durations": []
        }

    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    if len(rms) == 0 or np.max(rms) == 0:
        return {
            "total_duration": float(total_duration),
            "speech_duration": 0.0,
            "pause_duration": float(total_duration),
            "speech_percentage": 0.0,
            "pause_percentage": 100.0,
            "speech_pause_ratio": 0.0,
            "pause_durations": [float(total_duration)]
            if total_duration >= 0.3 else []
        }

    threshold = np.max(rms) * 0.10
    silent_frames = rms < threshold

    # Find consecutive low-energy intervals.
    pause_durations = []
    current_pause_frames = 0

    for is_silent in silent_frames:
        if is_silent:
            current_pause_frames += 1
        elif current_pause_frames > 0:
            duration = current_pause_frames * hop_length / sample_rate

            if duration >= 0.3:
                pause_durations.append(duration)

            current_pause_frames = 0

    if current_pause_frames > 0:
        duration = current_pause_frames * hop_length / sample_rate

        if duration >= 0.3:
            pause_durations.append(duration)

    # Sum only pauses meeting the minimum-duration criterion.
    pause_duration = min(sum(pause_durations), total_duration)
    speech_duration = max(0.0, total_duration - pause_duration)

    speech_percentage = speech_duration / total_duration * 100
    pause_percentage = pause_duration / total_duration * 100

    ratio = (
        speech_duration / pause_duration
        if pause_duration > 0
        else None
    )

    return {
        "total_duration": float(total_duration),
        "speech_duration": float(speech_duration),
        "pause_duration": float(pause_duration),
        "speech_percentage": float(speech_percentage),
        "pause_percentage": float(pause_percentage),
        "speech_pause_ratio": float(ratio) if ratio is not None else None,
        "pause_durations": pause_durations
    }


def extract_pause_features(audio, sample_rate):
    metrics = _calculate_speech_pause_metrics(audio, sample_rate)
    pauses = metrics["pause_durations"]
    total_pause = metrics["pause_duration"]
    total_duration = metrics["total_duration"]

    return {
        "pause_count": len(pauses),
        "total_pause_duration_seconds": float(total_pause),
        "average_pause_duration_seconds": (
            float(np.mean(pauses)) if pauses else 0.0
        ),
        "longest_pause_seconds": (
            float(max(pauses)) if pauses else 0.0
        ),
        "silence_percentage": (
            float(total_pause / total_duration * 100)
            if total_duration > 0 else 0.0
        )
    }


def extract_speech_rate(audio, sample_rate):
    metrics = _calculate_speech_pause_metrics(audio, sample_rate)

    return {
        "total_duration_seconds": metrics["total_duration"],
        "estimated_speech_duration_seconds": metrics["speech_duration"],
        "estimated_pause_duration_seconds": metrics["pause_duration"],
        "speech_percentage": metrics["speech_percentage"],
        "pause_percentage": metrics["pause_percentage"],
        "speech_to_pause_ratio": metrics["speech_pause_ratio"]
    }



def extract_jitter_shimmer(audio, sample_rate):

    # Reject invalid or very short audio
    if (
        audio is None
        or sample_rate <= 0
        or len(audio) < int(sample_rate * 0.1)
        or not np.all(np.isfinite(audio))
        or np.max(np.abs(audio)) == 0
    ):
        return {
            "jitter": None,
            "shimmer": None
        }

    try:
        # Create a Praat sound object
        sound = parselmouth.Sound(
            audio,
            sampling_frequency=sample_rate
        )

        # Identify vocal-cycle points using Praat
        point_process = parselmouth.praat.call(
            sound,
            "To PointProcess (periodic, cc)",
            75,
            300
        )

        # Calculate local jitter
        jitter = parselmouth.praat.call(
            point_process,
            "Get jitter (local)",
            0,
            0,
            1 / 300,
            1 / 75,
            1.3
        )

        # Calculate local shimmer
        shimmer = parselmouth.praat.call(
            [sound, point_process],
            "Get shimmer (local)",
            0,
            0,
            1 / 300,
            1 / 75,
            1.3,
            1.6
        )

        return {
            "jitter": (
                float(jitter)
                if np.isfinite(jitter)
                else None
            ),
            "shimmer": (
                float(shimmer)
                if np.isfinite(shimmer)
                else None
            )
        }

    except Exception as e:
        print(f"Jitter/Shimmer calculation failed: {e}")
        return {
            "jitter": None,
            "shimmer": None
        }


def extract_mel_spectrogram(audio, sample_rate: int):
    """
    Extracts 128-bin Mel-Spectrogram summary statistics for the CNN + Attention branch.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=128,
        fmax=8000
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    return {
        "n_mels": int(mel_db.shape[0]),
        "number_of_frames": int(mel_db.shape[1]),
        "mean_db": float(np.mean(mel_db)),
        "std_db": float(np.std(mel_db)),
        "min_db": float(np.min(mel_db)),
        "max_db": float(np.max(mel_db))
    }


def extract_all_acoustic_features(audio, sample_rate: int):
    """
    Master wrapper that runs all acoustic biomarker extractors once.
    """
    return {
        "mfcc": extract_mfcc(audio, sample_rate),
        "mel_spectrogram": extract_mel_spectrogram(audio, sample_rate),
        "pitch": extract_pitch(audio, sample_rate),
        "rms": extract_rms_energy(audio),
        "zcr": extract_zcr(audio),
        "pause": extract_pause_features(audio, sample_rate),
        "speech_rate": extract_speech_rate(audio, sample_rate),
        "jitter_shimmer": extract_jitter_shimmer(audio, sample_rate)
    }