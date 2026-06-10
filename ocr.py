"""
ocr.py
Extracts visible text from sampled video frames using EasyOCR.

Improvements over v1:
  1. Optional CLAHE + sharpening preprocessing — improves accuracy on
     low-contrast overlays and stylised fonts common in scam ads.
  2. Dual confidence thresholds — short strings (URLs, phone numbers,
     handles like "t.me/channel") use a lower threshold (0.25) so they
     are never silently dropped by the 0.40 general filter.
"""

from __future__ import annotations
import cv2
import numpy as np
from config import (
    OCR_LANGUAGES, OCR_GPU,
    OCR_PREPROCESSING,
    OCR_SHORT_STRING_CONF, OCR_LONG_STRING_CONF,
    OCR_SHORT_STRING_MAX_LEN,
)


# ─── Frame preprocessing ──────────────────────────────────────────────────────

def _preprocess_frame(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply contrast enhancement and sharpening to improve OCR accuracy.

    CLAHE (Contrast Limited Adaptive Histogram Equalisation) works on the
    luminance channel of the image and lifts text that appears dim or washed
    out without blowing out highlights.  The sharpening kernel then
    accentuates edges so character boundaries are clearer for the OCR model.
    """
    # Convert to LAB so we only enhance the L (luminance) channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Sharpening kernel — emphasises edges without introducing too much noise
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)

    return sharpened


# ─── Model loader ─────────────────────────────────────────────────────────────

def load_ocr_reader():
    """Load EasyOCR reader (downloads models on first run ~500 MB)."""
    import easyocr
    reader = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_GPU, verbose=False)
    return reader


# ─── Per-frame OCR ────────────────────────────────────────────────────────────

def extract_text_from_frame(frame_path: str, reader) -> list[str]:
    """
    Run OCR on a single frame.

    Applies preprocessing if OCR_PREPROCESSING is enabled, then uses
    dual confidence thresholds:
      - Short strings (≤ OCR_SHORT_STRING_MAX_LEN chars): 0.25 threshold
      - Longer strings: 0.40 threshold

    EasyOCR result format: [(bbox, text, confidence), ...]
    """
    if OCR_PREPROCESSING:
        img = cv2.imread(frame_path)
        if img is not None:
            img = _preprocess_frame(img)
            results = reader.readtext(img, detail=1, paragraph=False)
        else:
            results = reader.readtext(frame_path, detail=1, paragraph=False)
    else:
        results = reader.readtext(frame_path, detail=1, paragraph=False)

    texts: list[str] = []
    for (_, text, conf) in results:
        text = text.strip()
        if not text:
            continue
        # Choose threshold based on string length
        threshold = (
            OCR_SHORT_STRING_CONF
            if len(text) <= OCR_SHORT_STRING_MAX_LEN
            else OCR_LONG_STRING_CONF
        )
        if conf >= threshold:
            texts.append(text)

    return texts


# ─── Multi-frame OCR ──────────────────────────────────────────────────────────

def extract_text_from_frames(frame_paths: list[str], reader=None) -> dict:
    """
    Run OCR across all sampled frames and deduplicate repeated overlays.

    Returns:
        {
          "combined_text": str,             # all unique lines joined by " | "
          "per_frame":     list[list[str]], # per-frame raw results
          "unique_lines":  list[str],       # deduplicated text lines
        }
    """
    if not frame_paths:
        return {"combined_text": "", "per_frame": [], "unique_lines": []}

    if reader is None:
        reader = load_ocr_reader()

    per_frame:    list[list[str]] = []
    seen:         set[str]        = set()
    unique_lines: list[str]       = []

    for path in frame_paths:
        try:
            texts = extract_text_from_frame(path, reader)
        except Exception:
            texts = []

        per_frame.append(texts)

        for t in texts:
            key = " ".join(t.lower().split())
            if key not in seen and len(key) > 1:
                seen.add(key)
                unique_lines.append(t)

    combined_text = " | ".join(unique_lines)

    return {
        "combined_text": combined_text,
        "per_frame":     per_frame,
        "unique_lines":  unique_lines,
    }
