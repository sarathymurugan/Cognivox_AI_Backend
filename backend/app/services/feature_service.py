import librosa
import numpy as np


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

    # Estimate fundamental frequency (F0)
    pitch = librosa.yin(
        audio,
        fmin=75,
        fmax=300,
        sr=sample_rate
    )

    # Remove invalid values
    valid_pitch = pitch[~np.isnan(pitch)]

    # Handle case where no valid pitch is detected
    if len(valid_pitch) == 0:
        return {
            "mean_hz": None,
            "min_hz": None,
            "max_hz": None,
            "std_hz": None
        }

    return {
        "mean_hz": float(np.mean(valid_pitch)),
        "min_hz": float(np.min(valid_pitch)),
        "max_hz": float(np.max(valid_pitch)),
        "std_hz": float(np.std(valid_pitch))
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

def extract_pause_features(audio, sample_rate):
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)

    # Calculate RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=int(0.025 * sample_rate),
        hop_length=int(0.010 * sample_rate)
    )[0]

    # Use a relative energy threshold
    threshold = np.max(rms) * 0.10

    # Identify low-energy frames
    silent_frames = rms < threshold

    # Convert frames to time
    hop_length = int(0.010 * sample_rate)

    pause_durations = []
    current_pause_frames = 0

    for is_silent in silent_frames:

        if is_silent:
            current_pause_frames += 1

        else:

            if current_pause_frames > 0:

                duration = (
                    current_pause_frames * hop_length
                ) / sample_rate

                if duration >= 0.3:
                    pause_durations.append(duration)

                current_pause_frames = 0

    # Handle pause at the end of the recording
    if current_pause_frames > 0:

        duration = (
            current_pause_frames * hop_length
        ) / sample_rate

        if duration >= 0.3:
            pause_durations.append(duration)

    # No pauses detected
    if len(pause_durations) == 0:

        return {
            "pause_count": 0,
            "total_pause_duration_seconds": 0.0,
            "average_pause_duration_seconds": 0.0,
            "longest_pause_seconds": 0.0,
            "silence_percentage": 0.0
        }

    total_pause_duration = sum(pause_durations)

    return {
        "pause_count": len(pause_durations),
        "total_pause_duration_seconds": float(
            total_pause_duration
        ),
        "average_pause_duration_seconds": float(
            np.mean(pause_durations)
        ),
        "longest_pause_seconds": float(
            np.max(pause_durations)
        ),
        "silence_percentage": float(
            (total_pause_duration / (len(audio) / sample_rate)) * 100
        )
    }

def extract_speech_rate(audio, sample_rate):

    # RMS energy
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # Energy threshold
    threshold = np.max(rms) * 0.10

    # Identify speech frames
    speech_frames = rms >= threshold

    # Count speech frames
    speech_frame_count = np.sum(speech_frames)

    # Estimate speech duration
    speech_duration = (
        speech_frame_count * hop_length
    ) / sample_rate

    # Total duration
    total_duration = len(audio) / sample_rate

    # Pause duration
    pause_duration = total_duration - speech_duration

    pause_duration = max(0.0, pause_duration)

    # Percentages
    speech_percentage = (
        speech_duration / total_duration
    ) * 100

    pause_percentage = (
        pause_duration / total_duration
    ) * 100

    # Speech-to-pause ratio
    if pause_duration > 0:
        speech_pause_ratio = (
            speech_duration / pause_duration
        )
    else:
        speech_pause_ratio = None

    return {
        "total_duration_seconds": float(total_duration),
        "estimated_speech_duration_seconds": float(speech_duration),
        "estimated_pause_duration_seconds": float(pause_duration),
        "speech_percentage": float(speech_percentage),
        "pause_percentage": float(pause_percentage),
        "speech_to_pause_ratio": (
            float(speech_pause_ratio)
            if speech_pause_ratio is not None
            else None
        )
    }

def extract_jitter_shimmer(audio, sample_rate):

    # Estimate fundamental frequency
    f0 = librosa.yin(
        audio,
        fmin=75,
        fmax=300,
        sr=sample_rate
    )

    # Remove invalid values
    valid_f0 = f0[~np.isnan(f0)]

    if len(valid_f0) < 2:
        return {
            "jitter": None,
            "shimmer": None
        }

    # Jitter
    period = 1.0 / valid_f0

    period_difference = np.abs(
        np.diff(period)
    )

    mean_period = np.mean(period)

    jitter = (
        np.mean(period_difference) /
        mean_period
    )

    # RMS amplitude for shimmer estimation
    rms = librosa.feature.rms(
        y=audio,
        frame_length=int(0.025 * sample_rate),
        hop_length=int(0.010 * sample_rate)
    )[0]

    if len(rms) < 2:
        shimmer = None
    else:
        amplitude_difference = np.abs(
            np.diff(rms)
        )

        mean_amplitude = np.mean(rms)

        if mean_amplitude == 0:
            shimmer = None
        else:
            shimmer = (
                np.mean(amplitude_difference) /
                mean_amplitude
            )

    return {
        "jitter": float(jitter),
        "shimmer": (
            float(shimmer)
            if shimmer is not None
            else None
        )
    }