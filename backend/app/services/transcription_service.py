import warnings

warnings.filterwarnings("ignore")

# Lazy-loaded global model instance so Whisper loads only once
_whisper_model = None


def _get_whisper_model(model_size: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            _whisper_model = whisper.load_model(model_size)
        except ImportError:
            _whisper_model = False
    return _whisper_model


def transcribe_audio_bilingual(file_path: str, preferred_language: str = None) -> dict:
    """
    Transcribes Tamil ('ta') or English ('en') speech using Whisper while
    retaining natural disfluencies and filler words ('uh', 'um', 'வந்து', 'அதாவது').
    """

    if preferred_language and preferred_language.strip().lower() not in ("ta", "en"):
        preferred_language = None
        
    model = _get_whisper_model("base")

    # Graceful fallback if openai-whisper is not installed yet
    if not model:
        return {
            "available": False,
            "detected_language": preferred_language or "unknown",
            "transcript": "",
            "segments_count": 0,
            "note": "Install openai-whisper (`pip install openai-whisper`) to enable ASR."
        }

    # Verbatim prompt forces Whisper to keep filler words instead of cleaning them out
    disfluency_prompt = (
        "Um, uh, er, ah, like, you know, I mean... "
        "ஆமா, வந்து, அதாவது, ம்ம், என்னன்னா, சரி..."
    )

    transcribe_kwargs = {
        "initial_prompt": disfluency_prompt,
        "fp16": False
    }
    if preferred_language in ("ta", "en"):
        transcribe_kwargs["language"] = preferred_language

    result = model.transcribe(file_path, **transcribe_kwargs)
    transcript = result.get("text", "").strip()
    detected_lang = result.get("language", preferred_language or "en")

    return {
        "available": True,
        "detected_language": detected_lang,
        "transcript": transcript,
        "segments_count": len(result.get("segments", []))
    }