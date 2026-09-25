import librosa

def analyze_audio(file_path: str):
    # Load audio
    audio, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    # Calculate duration
    duration = len(audio) / sample_rate

    # Return audio information
    return {
        "sample_rate": sample_rate,
        "duration_seconds": round(duration, 2),
        "number_of_samples": len(audio),
        "channels": 1
    }