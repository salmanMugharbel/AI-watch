"""
speech_to_text.py
Transcribes the extracted audio track using Faster-Whisper.
Faster-Whisper is ~4x faster than openai-whisper and runs on CPU with int8.
"""

from __future__ import annotations
from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE


def load_whisper_model():
    """
    Load Faster-Whisper once and cache it.
    Called lazily so import time stays fast.
    """
    from faster_whisper import WhisperModel
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    return model


def transcribe(audio_path: str, model=None) -> dict:
    """
    Transcribe audio and return structured output.

    Returns:
        {
          "full_text": str,           # complete transcript
          "segments": list[dict],     # per-segment start/end/text
          "language": str,            # detected language code
          "duration_sec": float,
        }
    """
    if audio_path is None:
        return {"full_text": "", "segments": [], "language": "unknown", "duration_sec": 0.0}

    if model is None:
        model = load_whisper_model()

    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=None,   # auto-detect language
        vad_filter=True, # voice-activity detection to skip silence
    )

    segments = []
    for seg in segments_gen:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })

    full_text = " ".join(s["text"] for s in segments)

    return {
        "full_text": full_text,
        "segments": segments,
        "language": info.language,
        "duration_sec": info.duration,
    }
