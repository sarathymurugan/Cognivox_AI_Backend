from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import shutil
import librosa
from backend.app.services.feature_service import (extract_mfcc, extract_pitch, extract_rms_energy, extract_zcr, extract_pause_features, extract_speech_rate, extract_jitter_shimmer)
from backend.app.services.audio_service import analyze_audio


app = FastAPI(
    title="MCI Speech Analysis API",
    description="Backend API for AI-based speech biomarker analysis",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "MCI Speech Analysis API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):

    # Check file extension
    allowed_extensions = [".wav", ".mp3", ".flac"]

    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only WAV, MP3, and FLAC audio files are allowed."
        )

    # Create uploads folder
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Create file path
    file_path = os.path.join(
        upload_dir,
        file.filename
    )

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    # Analyze audio
    audio_info = analyze_audio(file_path)

    return {
        "message": "Audio uploaded successfully",
        "filename": file.filename,
        "audio_analysis": audio_info
    }

@app.post("/analyze-audio")
async def analyze_audio_endpoint(
    file: UploadFile = File(...)
):

    allowed_extensions = [".wav", ".mp3", ".flac"]

    file_extension = os.path.splitext(
        file.filename
    )[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only WAV, MP3, and FLAC audio files are allowed."
        )

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(
        upload_dir,
        file.filename
    )

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    # Basic audio analysis
    audio_info = analyze_audio(file_path)

    # Load audio ONCE
    audio, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    # MFCC extraction
    mfcc_features = extract_mfcc(audio, sample_rate)

    # Pitch extraction
    pitch_features = extract_pitch(
        audio,
        sample_rate
    )

    # RMS energy extraction
    rms_features = extract_rms_energy(
        audio
    )

    # Zero-crossing rate extraction
    zcr_features = extract_zcr(
        audio
    )

    # Pause features extraction
    pause_features = extract_pause_features(
        audio,
        sample_rate
    )

    # Speech rate extraction
    speech_rate_features = extract_speech_rate(
        audio,
        sample_rate
    )

    # Jitter and shimmer extraction
    jitter_shimmer_features = extract_jitter_shimmer(
        audio,
        sample_rate
    )

    return {
        "message": "Audio analysis completed successfully",
        "filename": file.filename,
        "audio_analysis": audio_info,
        "features": {
            "mfcc": mfcc_features,
            "pitch": pitch_features,
            "rms": rms_features,
            "zcr": zcr_features,
            "pause": pause_features,
            "speech_rate": speech_rate_features,
            "jitter_shimmer": jitter_shimmer_features
        }
    }