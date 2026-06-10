"""
video_processing.py
Extracts audio (WAV) and key frames (JPEG) from an uploaded video file.
Uses imageio-ffmpeg for a bundled FFmpeg binary — no system install required.
"""

import os
import subprocess
import tempfile
import cv2
import imageio_ffmpeg
from config import FRAME_EXTRACTION_FPS, MAX_FRAMES

# Resolve the bundled FFmpeg binary path once at import time.
# imageio-ffmpeg ships a static binary inside the package, so this works
# on Windows/macOS/Linux without the user installing FFmpeg separately.
_FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: str, output_dir: str) -> str | None:
    """
    Use the bundled FFmpeg binary to pull the audio track as a 16-kHz mono WAV.
    16 kHz mono is exactly what Faster-Whisper expects.
    Returns the path to the WAV file, or None if the video has no audio.
    """
    audio_path = os.path.join(output_dir, "audio.wav")
    cmd = [
        _FFMPEG_BIN, "-y",       # overwrite silently
        "-i", video_path,
        "-vn",                   # drop the video stream
        "-acodec", "pcm_s16le",  # raw PCM – widest compatibility
        "-ar", "16000",          # 16 kHz sample rate
        "-ac", "1",              # mono
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # FFmpeg returns 0 even when there's no audio; check the file was created
    if result.returncode != 0 or not os.path.exists(audio_path):
        return None

    # An empty (header-only) WAV is < 100 bytes
    if os.path.getsize(audio_path) < 100:
        return None

    return audio_path


def extract_frames(video_path: str, output_dir: str) -> list[str]:
    """
    Sample frames from the video at FRAME_EXTRACTION_FPS (default 0.5 fps).
    Capped at MAX_FRAMES so OCR doesn't become the bottleneck.
    Returns a list of saved JPEG paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # How many native frames to skip between each sample
    step = max(1, int(native_fps / FRAME_EXTRACTION_FPS))

    saved_paths: list[str] = []
    frame_idx = 0

    while cap.isOpened() and len(saved_paths) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            out_path = os.path.join(output_dir, f"frame_{len(saved_paths):04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)

        frame_idx += 1

    cap.release()
    return saved_paths


def get_video_metadata(video_path: str) -> dict:
    """Return basic video metadata (duration, fps, resolution)."""
    cap = cv2.VideoCapture(video_path)
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": 0.0,
    }
    if meta["fps"] > 0:
        meta["duration_sec"] = meta["total_frames"] / meta["fps"]
    cap.release()
    return meta


def process_video(video_path: str) -> dict:
    """
    Entry point called by the pipeline.
    Returns a dict with audio_path, frame_paths, metadata, and temp_dir.
    The caller is responsible for cleaning up temp_dir when done.
    """
    temp_dir = tempfile.mkdtemp(prefix="aiwatch_")

    metadata = get_video_metadata(video_path)
    audio_path = extract_audio(video_path, temp_dir)
    frame_paths = extract_frames(video_path, temp_dir)

    return {
        "audio_path": audio_path,       # None if no audio track
        "frame_paths": frame_paths,
        "metadata": metadata,
        "temp_dir": temp_dir,
    }
