# import os
# import shutil
# import librosa
# import subprocess
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from backend.app.services.feature_service import (extract_mfcc, extract_pitch, extract_rms_energy, extract_zcr, extract_pause_features, extract_speech_rate, extract_jitter_shimmer)
# from backend.app.services.audio_service import analyze_audio
# from app.services.audio_service import convert_webm_to_wav, load_and_preprocess_audio
# from app.services.feature_service import extract_all_acoustic_features
# from app.services.transcription_service import transcribe_audio_bilingual
# from app.services.linguistic_service import extract_linguistic_biomarkers
# from app.services.scoring_service import calculate_mci_risk_score


# app = FastAPI(
#     title="MCI Speech Analysis API",
#     description="Backend API for AI-based speech biomarker analysis",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:8081",
#         "http://127.0.0.1:8081",
#     ],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def convert_webm_to_wav(input_path: str, output_path: str):
#     try:
#         subprocess.run(
#             [
#                 "ffmpeg",
#                 "-y",
#                 "-i",
#                 input_path,
#                 "-vn",
#                 "-ac",
#                 "1",
#                 "-ar",
#                 "48000",
#                 output_path
#             ],
#             check=True,
#             capture_output=True,
#             text=True
#         )

#     except subprocess.CalledProcessError as error:
#         print("FFmpeg conversion failed:")
#         print(error.stderr)

#         raise HTTPException(
#             status_code=500,
#             detail="Unable to convert WebM audio to WAV."
#         )
    
# @app.get("/")
# def root():
#     return {
#         "message": "MCI Speech Analysis API is running"
#     }


# @app.get("/health")
# def health_check():
#     return {
#         "status": "healthy"
#     }


# @app.post("/upload-audio")
# async def upload_audio(file: UploadFile = File(...)):

#     # Check file extension
#     allowed_extensions = [".wav", ".mp3", ".flac", ".webm"]

#     file_extension = os.path.splitext(file.filename)[1].lower()

#     if file_extension not in allowed_extensions:
#         raise HTTPException(
#             status_code=400,
#             detail="Only WAV, MP3, FLAC, and WebM audio files are allowed."
#         )

#     # Create uploads folder
#     upload_dir = "uploads"
#     os.makedirs(upload_dir, exist_ok=True)

#     # Create file path
#     file_path = os.path.join(
#         upload_dir,
#         file.filename
#     )

#     # Save uploaded file
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(
#             file.file,
#             buffer
#         )

#     # Analyze audio
#     audio_info = analyze_audio(file_path)

#     return {
#         "message": "Audio uploaded successfully",
#         "filename": file.filename,
#         "audio_analysis": audio_info
#     }

# @app.post("/analyze-audio")
# async def analyze_audio_endpoint(
#     file: UploadFile = File(...)
# ):

#     allowed_extensions = [".wav", ".mp3", ".flac", ".webm"]

#     file_extension = os.path.splitext(
#         file.filename
#     )[1].lower()

#     if file_extension not in allowed_extensions:
#         raise HTTPException(
#             status_code=400,
#             detail="Only WAV, MP3, FLAC, and WebM audio files are allowed."
#         )

#     upload_dir = "uploads"
#     os.makedirs(upload_dir, exist_ok=True)

#     original_file_path = os.path.join(
#         upload_dir,
#         file.filename
#     )

#     # Save uploaded file
#     with open(original_file_path, "wb") as buffer:
#         shutil.copyfileobj(
#             file.file,
#             buffer
#         )

#         # Convert WebM recording to WAV
#     if file_extension == ".webm":

#         wav_filename = (
#             os.path.splitext(file.filename)[0]
#             + "_converted.wav"
#         )

#         file_path = os.path.join(
#             upload_dir,
#             wav_filename
#         )

#         convert_webm_to_wav(
#             original_file_path,
#             file_path
#         )

#     else:
#         file_path = original_file_path


#     # Basic audio analysis
#     audio_info = analyze_audio(file_path)

#     # Load audio ONCE
#     audio, sample_rate = librosa.load(
#         file_path,
#         sr=None,
#         mono=True
#     )
    
#     # MFCC extraction
#     mfcc_features = extract_mfcc(audio, sample_rate)

#     # Pitch extraction
#     pitch_features = extract_pitch(
#         audio,
#         sample_rate
#     )

#     # RMS energy extraction
#     rms_features = extract_rms_energy(
#         audio
#     )

#     # Zero-crossing rate extraction
#     zcr_features = extract_zcr(
#         audio
#     )

#     # Pause features extraction
#     pause_features = extract_pause_features(
#         audio,
#         sample_rate
#     )

#     # Speech rate extraction
#     speech_rate_features = extract_speech_rate(
#         audio,
#         sample_rate
#     )

#     # Jitter and shimmer extraction
#     jitter_shimmer_features = extract_jitter_shimmer(
#         audio,
#         sample_rate
#     )

#     return {
#         "message": "Audio analysis completed successfully",
#         "filename": file.filename,
#         "audio_analysis": audio_info,
#         "features": {
#             "mfcc": mfcc_features,
#             "pitch": pitch_features,
#             "rms": rms_features,
#             "zcr": zcr_features,
#             "pause": pause_features,
#             "speech_rate": speech_rate_features,
#             "jitter_shimmer": jitter_shimmer_features
#         }
#     }










import os
import uuid
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.services.audio_service import (
    convert_webm_to_wav,
    load_and_preprocess_audio,
    analyze_audio
)
from backend.app.services.feature_service import extract_all_acoustic_features
from backend.app.services.transcription_service import transcribe_audio_bilingual
from backend.app.services.linguistic_service import extract_linguistic_biomarkers
from backend.app.services.scoring_service import calculate_mci_risk_score
from backend.app.services.storage_service import (
    log_screening_result,
    get_recent_screening_logs,
    save_consented_dataset_sample
)
from backend.app.models.multimodal_mci_net import run_deep_multimodal_inference


app = FastAPI(
    title="MCI Speech Analysis API",
    description="Bilingual (Tamil-English) AI Speech Biomarker Screening API for Early MCI Detection",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "MCI Speech Analysis API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/screening-logs")
def fetch_screening_logs(limit: int = 50):
    """Returns recent screening history for PHC / Clinician review."""
    logs = get_recent_screening_logs(limit=limit)
    return {"count": len(logs), "logs": logs}

@app.get("/kpi-report")
def get_kpi_validation_report():
    """Returns the latest CMRG Model Performance & KPI Validation Report."""
    report_path = os.path.join("features", "kpi_evaluation_report.json")
    if not os.path.exists(report_path):
        return {
            "status": "not_generated",
            "message": "Run `python scripts/evaluate_model_kpis.py` to generate the KPI report."
        }
    import json
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    allowed_extensions = [".wav", ".mp3", ".flac", ".webm"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only WAV, MP3, FLAC, and WebM audio files are allowed."
        )

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file_extension == ".webm":
        wav_path = os.path.splitext(file_path)[0] + "_converted.wav"
        convert_webm_to_wav(file_path, wav_path)
        file_path = wav_path

    audio_info = analyze_audio(file_path)

    return {
        "message": "Audio uploaded successfully",
        "filename": file.filename,
        "audio_analysis": audio_info
    }


@app.post("/analyze-audio")
async def analyze_audio_endpoint(
    file: UploadFile = File(...),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    language: Optional[str] = Form(None), # 'ta' for Tamil, 'en' for English
    task_type: Optional[str] = Form("picture_description")
):
    allowed_extensions = [".wav", ".mp3", ".flac", ".webm"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only WAV, MP3, FLAC, and WebM audio files are allowed."
        )

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Use a unique session ID so simultaneous PHC users don't overwrite files
    session_id = str(uuid.uuid4())[:8]
    original_file_path = os.path.join(upload_dir, f"{session_id}_{file.filename}")
    converted_wav_path = None

    try:
        with open(original_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Convert WebM recording to WAV
        if file_extension == ".webm":
            converted_wav_path = os.path.join(upload_dir, f"{session_id}_converted.wav")
            convert_webm_to_wav(original_file_path, converted_wav_path)
            file_path = converted_wav_path
        else:
            file_path = original_file_path

        # 1. Basic audio info & preprocessed waveform
        audio_info = analyze_audio(file_path)
        audio, sample_rate = load_and_preprocess_audio(file_path, trim_silence=True)

        # 2. Extract All Acoustic Biomarkers (MFCC, Mel-Spectrogram, Pitch, RMS, ZCR, Pauses, Jitter/Shimmer)
        acoustic_features = extract_all_acoustic_features(audio, sample_rate)

        # 3. Run Whisper Bilingual ASR (Tamil / English)
        transcription_result = transcribe_audio_bilingual(file_path, preferred_language=language)

        # 4. Extract Bilingual Linguistic Biomarkers (Fillers, Pronouns, Lexical Diversity)
        linguistic_features = extract_linguistic_biomarkers(
            transcript=transcription_result["transcript"],
            language=transcription_result["detected_language"]
        )

        # 5. Deep Learning Forward Pass (Figure 1 BiLSTM + CNN + Attention Fusion)
        dl_inference = run_deep_multimodal_inference(
            audio=audio,
            sample_rate=sample_rate,
            transcript=transcription_result["transcript"],
            language=transcription_result["detected_language"],
            acoustic_features=acoustic_features,
            linguistic_features=linguistic_features
        )

        # 6. Clinical MCI Risk Score & PHC Referral
        risk_assessment = calculate_mci_risk_score(
            acoustic_features=acoustic_features,
            linguistic_features=linguistic_features,
            age=age,
            gender=gender
        )
        risk_assessment["deep_learning_module"] = dl_inference

        response_payload = {
            "message": "Audio analysis completed successfully",
            "session_id": session_id,
            "filename": file.filename,
            "task_type": task_type,
            "audio_analysis": audio_info,
            "transcription": transcription_result,
            "linguistic_biomarkers": linguistic_features,
            "features": acoustic_features,
            "mci_screening": risk_assessment
        }

        # 7. Log screening result to features/screening_logs.csv
        log_screening_result(session_id, file.filename, task_type, response_payload)

        return response_payload

    finally:
        for temp_path in [original_file_path, converted_wav_path]:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


@app.post("/collect-sample")
async def collect_dataset_sample_endpoint(
    file: UploadFile = File(...),
    participant_id: str = Form(...),
    language: str = Form("ta"),  # 'ta' or 'en'
    task_type: str = Form("picture_description"),  # picture_description | recall | narrative
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    mmse_score: Optional[float] = Form(None),
    clinical_label: str = Form("control")  # control | stress_drift | mci
):
    """
    Dedicated endpoint for building the >= 500 participant annotated Tamil-English MCI dataset.
    Standardizes audio to WAV, saves it in dataset/, and logs ground-truth MMSE/clinical labels.
    """
    allowed_extensions = [".wav", ".mp3", ".flac", ".webm"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only WAV, MP3, FLAC, and WebM audio files are allowed."
        )

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    temp_id = str(uuid.uuid4())[:8]
    raw_path = os.path.join(upload_dir, f"{temp_id}_{file.filename}")
    wav_path = os.path.join(upload_dir, f"{temp_id}_standard.wav")

    try:
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Standardize all dataset recordings to 48kHz Mono WAV
        convert_webm_to_wav(raw_path, wav_path)

        saved_info = save_consented_dataset_sample(
            wav_file_path=wav_path,
            participant_id=participant_id,
            language=language if language in ("ta", "en") else "ta",
            task_type=task_type,
            age=age if (age and age > 0) else None,
            gender=gender if (gender and gender != "string") else None,
            mmse_score=mmse_score if (mmse_score and mmse_score > 0) else None,
            clinical_label=clinical_label
        )

        return {
            "message": "Participant sample saved to bilingual MCI dataset",
            "dataset_entry": saved_info
        }

    finally:
        for p in [raw_path, wav_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass