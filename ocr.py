"""
ocr.py
Extracts visible text from sampled video frames using EasyOCR.
EasyOCR supports 80+ languages and works well on overlaid text / banners.
"""

from __future__ import annotations
from config import OCR_LANGUAGES, OCR_GPU


def load_ocr_reader():
    """Load EasyOCR reader once (downloads models on first run ~500 MB)."""
    import easyocr
    reader = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_GPU, verbose=False)
    return reader


def extract_text_from_frame(frame_path: str, reader) -> list[str]:
    """
    Run OCR on a single frame image.
    Returns a list of text strings above a confidence threshold.
    EasyOCR result format: [(bbox, text, confidence), ...]
    """
    results = reader.readtext(frame_path, detail=1, paragraph=False)
    # Keep only results with >= 40% confidence to filter noise
    texts = [text for (_, text, conf) in results if conf >= 0.40 and text.strip()]
    return texts


def extract_text_from_frames(frame_paths: list[str], reader=None) -> dict:
    """
    Run OCR across all sampled frames.
    Deduplicates repeated overlays that appear across multiple consecutive frames.

    Returns:
        {
          "combined_text": str,       # all unique text joined
          "per_frame": list[list[str]], # per-frame raw results
          "unique_lines": list[str],  # deduplicated text lines
        }
    """
    if not frame_paths:
        return {"combined_text": "", "per_frame": [], "unique_lines": []}

    if reader is None:
        reader = load_ocr_reader()

    per_frame: list[list[str]] = []
    seen: set[str] = set()
    unique_lines: list[str] = []

    for path in frame_paths:
        try:
            texts = extract_text_from_frame(path, reader)
        except Exception:
            texts = []

        per_frame.append(texts)

        for t in texts:
            # Normalise for deduplication: lowercase + collapse whitespace
            key = " ".join(t.lower().split())
            if key not in seen and len(key) > 2:
                seen.add(key)
                unique_lines.append(t)

    combined_text = " | ".join(unique_lines)

    return {
        "combined_text": combined_text,
        "per_frame": per_frame,
        "unique_lines": unique_lines,
    }
