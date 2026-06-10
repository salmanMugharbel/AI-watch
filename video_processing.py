"""
video_processing.py
Extracts audio (WAV) and key frames (JPEG) from an uploaded video file.

Key improvement over v1: adaptive frame-sampling rate based on video duration.
Short scam clips (< 30 s) are now sampled at 2 fps so nothing is missed.
First and last frames are always captured regardless of the computed step size.
"""

import os
import subprocess
import tempfile
import cv2
import imageio_ffmpeg
from config import (
    ADAPTIVE_FRAME_SAMPLING,
    FRAME_EXTRACTION_FPS,
    FRAME_FPS_SHORT, FRAME_FPS_MEDIUM, FRAME_FPS_LONG, FRAME_FPS_XLONG,
    MAX_FRAMES,
)

_FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()


# ─── Adaptive FPS lookup ──────────────────────────────────────────────────────

def _target_fps(duration_sec: float) -> float:
    """
    Return the appropriate frame-sampling rate for a video of this duration.
    Shorter videos → denser sampling so short scam pitches are not missed.
    """
    if not ADAPTIVE_FRAME_SAMPLING:
        return FRAME_EXTRACTION_FPS
    if duration_sec < 30:
        return FRAME_FPS_SHORT    # 2.0 fps — every 0.5 s
    if duration_sec < 120:
        return FRAME_FPS_MEDIUM   # 1.0 fps — every second
    if duration_sec < 600:
        return FRAME_FPS_LONG     # 0.5 fps — every 2 s
    return FRAME_FPS_XLONG        # 0.25 fps — every 4 s


# ─── Audio extraction ─────────────────────────────────────────────────────────

def extract_audio(video_path: str, output_dir: str) -> str | None:
    """
    Pull audio track as 16-kHz mono WAV using the bundled FFmpeg binary.
    Returns path to WAV, or None if the video has no audio.
    """
    audio_path = os.path.join(output_dir, "audio.wav")
    cmd = [
        _FFMPEG_BIN, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not os.path.exists(audio_path):
        return None
    if os.path.getsize(audio_path) < 100:   # header-only = no real audio
        return None
    return audio_path


# ─── Frame extraction ─────────────────────────────────────────────────────────

def extract_frames(video_path: str, output_dir: str, duration_sec: float = 0.0) -> list[str]:
    """
    Sample frames at an adaptive rate determined by video duration.
    Always includes the first and last frame of the video so that opening
    claims and closing CTAs are never missed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    native_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if duration_sec <= 0 and native_fps > 0:
        duration_sec = total_frames / native_fps

    fps_target = _target_fps(duration_sec)
    step = max(1, int(native_fps / fps_target))

    # Build the set of frame indices we want to capture
    wanted: set[int] = set()
    idx = 0
    while idx < total_frames and len(wanted) < MAX_FRAMES - 2:
        wanted.add(idx)
        idx += step

    # Always capture first and last frame
    wanted.add(0)
    if total_frames > 1:
        wanted.add(total_frames - 1)

    wanted_sorted = sorted(wanted)

    saved_paths: list[str] = []
    frame_idx = 0
    capture_set = set(wanted_sorted)

    while cap.isOpened() and len(saved_paths) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in capture_set:
            out_path = os.path.join(output_dir, f"frame_{len(saved_paths):04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)

        frame_idx += 1

    cap.release()
    return saved_paths


# ─── Metadata ─────────────────────────────────────────────────────────────────

def get_video_metadata(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    meta = {
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": 0.0,
    }
    if meta["fps"] > 0:
        meta["duration_sec"] = meta["total_frames"] / meta["fps"]
    cap.release()
    return meta


# ─── Entry point ──────────────────────────────────────────────────────────────

def process_video(video_path: str) -> dict:
    temp_dir = tempfile.mkdtemp(prefix="aiwatch_")
    metadata = get_video_metadata(video_path)
    audio_path  = extract_audio(video_path, temp_dir)
    frame_paths = extract_frames(video_path, temp_dir, metadata["duration_sec"])
    return {
        "audio_path":  audio_path,
        "frame_paths": frame_paths,
        "metadata":    metadata,
        "temp_dir":    temp_dir,
    }
