"""
vision_analysis.py
Three-layer visual scam detection:

  Layer 1 — YOLOv8 object detection
    Uses pretrained COCO weights to detect real-world objects that correlate
    with scam content: luxury cars, cash displays, phones, laptops, etc.

  Layer 2 — pyzbar QR code detection
    Purpose-built QR/barcode decoder. Far more reliable than YOLO for QR.
    A QR code in a scam video typically routes victims to Telegram or a
    phishing URL — very high-signal indicator.

  Layer 3 — CLIP zero-shot image classification
    openai/clip-vit-base-patch32 classifies each frame against text prompts
    such as "a casino interface" or "a crypto trading chart" with no custom
    training required.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from config import (
    YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_MAX_ANNOTATED_FRAMES,
    CLIP_MODEL, CLIP_CONFIDENCE_THRESHOLD, ENABLE_CLIP_ANALYSIS,
)


# ─── COCO class → risk mapping ────────────────────────────────────────────────
# Only classes that carry scam signal are listed here.
# score  : raw points added to the per-frame tally when class is detected
# category: human-readable group shown in the UI
# label  : display label

YOLO_RISK_MAP: dict[str, dict] = {
    "car":        {"score": 8,  "category": "Luxury Lifestyle",      "label": "Luxury Car / Vehicle"},
    "motorcycle": {"score": 5,  "category": "Luxury Lifestyle",      "label": "Motorcycle"},
    "cell phone": {"score": 12, "category": "Contact CTA",           "label": "Mobile / Messaging Device"},
    "laptop":     {"score": 7,  "category": "Trading Interface",     "label": "Laptop / Screen"},
    "tv":         {"score": 5,  "category": "Screen Display",        "label": "Display Screen"},
    "clock":      {"score": 5,  "category": "Luxury Lifestyle",      "label": "Watch / Clock"},
    "handbag":    {"score": 6,  "category": "Luxury Goods",          "label": "Luxury Handbag"},
    "suitcase":   {"score": 4,  "category": "Luxury Lifestyle",      "label": "Luggage / Travel"},
    "person":     {"score": 2,  "category": "Social Proof",          "label": "Person (social proof)"},
    "wine glass": {"score": 4,  "category": "Luxury Lifestyle",      "label": "Luxury / Party Scene"},
    "bottle":     {"score": 3,  "category": "Luxury Lifestyle",      "label": "Bottle / Party Scene"},
}

# Bounding-box colour per category (BGR for OpenCV)
CATEGORY_COLORS: dict[str, tuple] = {
    "Luxury Lifestyle":  (0, 215, 255),   # gold
    "Contact CTA":       (0, 100, 255),   # orange-red
    "Trading Interface": (255, 180, 0),   # blue-ish
    "Screen Display":    (200, 200, 0),   # cyan
    "Luxury Goods":      (0, 215, 255),   # gold
    "Social Proof":      (180, 180, 180), # grey
    "QR Code":           (0, 0, 255),     # bright red
}

# ─── CLIP zero-shot prompts ────────────────────────────────────────────────────
# Each entry: (text prompt, base_score, display_label)
# base_score is awarded when CLIP confidence ≥ CLIP_CONFIDENCE_THRESHOLD.
CLIP_RISK_PROMPTS: list[tuple[str, int, str]] = [
    ("a casino slot machine or online gambling website",         25, "Casino / Gambling Interface"),
    ("a cryptocurrency trading chart or crypto exchange",        22, "Crypto Trading Interface"),
    ("a luxury lifestyle showing expensive cars jewelry money",  12, "Luxury Lifestyle Display"),
    ("a referral program or bonus invitation screen",            18, "Referral / Invite Screen"),
    ("a financial investment or get-rich-quick advertisement",   20, "Investment Scam Ad"),
    ("a WhatsApp or Telegram messaging app interface",           12, "Messaging App (contact CTA)"),
    ("a QR code payment or link screen",                         15, "QR Code / Payment Screen"),
    ("a normal everyday video with no financial content",         0, "Normal / Clean Content"),
]


# ─── Model loaders ────────────────────────────────────────────────────────────

def load_yolo_model():
    """Load YOLOv8 model. Downloads yolov8n.pt (~6 MB) on first call."""
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)
    return model


def load_clip_model():
    """Load CLIP processor + model (~340 MB download on first call)."""
    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL)
    model.eval()
    return processor, model


# ─── Layer 1: YOLOv8 object detection ─────────────────────────────────────────

def run_yolo_on_frames(
    frame_paths: list[str],
    yolo_model,
) -> tuple[list[dict], list[np.ndarray]]:
    """
    Run YOLOv8 on every frame. Returns:
        detections  : flat list of detection dicts
        annotated   : numpy RGB arrays (one per frame that had ≥1 relevant hit)
    """
    detections: list[dict] = []
    annotated_frames: list[np.ndarray] = []

    for frame_idx, path in enumerate(frame_paths):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        results = yolo_model(path, conf=YOLO_CONFIDENCE, verbose=False)
        result  = results[0]

        frame_hits: list[dict] = []

        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            if cls_name not in YOLO_RISK_MAP:
                continue   # not a scam-relevant class

            conf  = float(box.conf)
            info  = YOLO_RISK_MAP[cls_name]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detection = {
                "frame_idx":  frame_idx,
                "class_name": cls_name,
                "label":      info["label"],
                "category":   info["category"],
                "confidence": round(conf, 3),
                "score":      info["score"],
                "bbox":       (x1, y1, x2, y2),
                "source":     "yolo",
            }
            frame_hits.append(detection)
            detections.append(detection)

        # Draw bounding boxes if any relevant objects found
        if frame_hits and len(annotated_frames) < YOLO_MAX_ANNOTATED_FRAMES:
            annotated = _draw_yolo_boxes(img_bgr.copy(), frame_hits)
            annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    return detections, annotated_frames


def _draw_yolo_boxes(img_bgr: np.ndarray, hits: list[dict]) -> np.ndarray:
    """Draw labelled bounding boxes on a BGR image."""
    for hit in hits:
        x1, y1, x2, y2 = hit["bbox"]
        color = CATEGORY_COLORS.get(hit["category"], (0, 255, 0))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label_text = f"{hit['label']} {hit['confidence']:.0%}"
        # Background rectangle for text readability
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img_bgr, label_text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return img_bgr


# ─── Layer 2: pyzbar QR detection ─────────────────────────────────────────────

def run_qr_detection(frame_paths: list[str]) -> list[dict]:
    """
    Scan frames for QR codes using pyzbar.
    Returns one detection dict per unique decoded value found.
    pyzbar requires the zbar shared library (bundled on macOS/Linux;
    on Windows install 'python-pyzbar' which ships the DLLs).
    """
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        from PIL import Image as PILImage
    except ImportError:
        return []   # pyzbar not installed — skip silently

    seen_data: set[str] = set()
    detections: list[dict] = []

    for frame_idx, path in enumerate(frame_paths):
        try:
            pil_img = PILImage.open(path)
            codes = pyzbar_decode(pil_img)
        except Exception:
            continue

        for code in codes:
            decoded = code.data.decode("utf-8", errors="replace").strip()
            if decoded in seen_data:
                continue
            seen_data.add(decoded)

            detections.append({
                "frame_idx":  frame_idx,
                "class_name": "qr_code",
                "label":      "QR Code",
                "category":   "QR Code",
                "confidence": 1.0,   # pyzbar either finds it or doesn't
                "score":      25,    # high-signal: QR codes route to scam pages
                "decoded":    decoded,
                "source":     "pyzbar",
            })

    return detections


# ─── Layer 3: CLIP zero-shot frame classification ─────────────────────────────

def run_clip_on_frames(
    frame_paths: list[str],
    clip_processor,
    clip_model,
) -> list[dict]:
    """
    Classify each frame against CLIP_RISK_PROMPTS using zero-shot image→text
    similarity. Returns one result dict per frame that exceeds the confidence
    threshold for at least one scam-oriented prompt.
    """
    import torch
    from PIL import Image as PILImage

    prompts = [p for p, _, _ in CLIP_RISK_PROMPTS]
    results: list[dict] = []

    for frame_idx, path in enumerate(frame_paths):
        try:
            pil_img = PILImage.open(path).convert("RGB")
        except Exception:
            continue

        inputs = clip_processor(
            text=prompts,
            images=pil_img,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs   = clip_model(**inputs)
            logits    = outputs.logits_per_image   # shape [1, num_prompts]
            probs     = logits.softmax(dim=-1)[0].tolist()

        for i, (prompt, base_score, display_label) in enumerate(CLIP_RISK_PROMPTS):
            conf = probs[i]
            if conf < CLIP_CONFIDENCE_THRESHOLD:
                continue
            # Skip the "normal content" label — it's a control, not a finding
            if base_score == 0:
                continue

            results.append({
                "frame_idx":   frame_idx,
                "class_name":  "clip_classification",
                "label":       display_label,
                "category":    display_label,
                "confidence":  round(conf, 3),
                # Scale base_score by CLIP confidence (higher conf = more points)
                "score":       round(base_score * conf),
                "prompt":      prompt,
                "source":      "clip",
            })

    return results


# ─── Score aggregation ────────────────────────────────────────────────────────

def _aggregate_visual_score(all_detections: list[dict]) -> tuple[float, list[str]]:
    """
    Convert the flat list of detections into a normalised 0–1 score and
    human-readable reasons.

    Scoring rules:
    - For YOLO / pyzbar: each detection type scores once (prevents one car
      appearing in 30 frames from multiplying the score × 30).
      If the same class appears in > 30 % of frames it gets a persistence bonus.
    - For CLIP: award the highest-confidence score for each label seen.
    """
    total_frames = max(
        max((d["frame_idx"] for d in all_detections), default=0) + 1, 1
    )

    # --- YOLO + pyzbar: score each unique class once + persistence bonus ---
    yolo_pyzbar = [d for d in all_detections if d["source"] in ("yolo", "pyzbar")]
    class_frames: dict[str, set[int]] = {}
    class_info:   dict[str, dict]     = {}

    for d in yolo_pyzbar:
        cls = d["class_name"]
        class_frames.setdefault(cls, set()).add(d["frame_idx"])
        if cls not in class_info or d["confidence"] > class_info[cls]["confidence"]:
            class_info[cls] = d

    yolo_raw = 0.0
    reasons: list[str] = []

    for cls, frames_seen in class_frames.items():
        info        = class_info[cls]
        base_score  = info["score"]
        persistence = len(frames_seen) / total_frames

        # Bonus: up to +50 % extra if class appears in > 30 % of frames
        bonus_multiplier = 1.0 + (0.5 * max(0.0, (persistence - 0.3) / 0.7))
        awarded = base_score * bonus_multiplier
        yolo_raw += awarded

        decoded_note = f' (decoded: {info["decoded"][:60]})' if "decoded" in info else ""
        reasons.append(
            f'{info["label"]} detected in {len(frames_seen)} frame(s)'
            f' — confidence {info["confidence"]:.0%}{decoded_note}'
        )

    # --- CLIP: best confidence per label ---
    clip_hits = [d for d in all_detections if d["source"] == "clip"]
    best_clip: dict[str, dict] = {}
    for d in clip_hits:
        lbl = d["label"]
        if lbl not in best_clip or d["confidence"] > best_clip[lbl]["confidence"]:
            best_clip[lbl] = d

    clip_raw = 0.0
    for lbl, d in best_clip.items():
        clip_raw += d["score"]
        reasons.append(
            f'Visual scene classified as "{lbl}" '
            f'({d["confidence"]:.0%} confidence)'
        )

    # Combine and normalise to 0–1 (soft cap: raw=50 → score=1.0)
    raw_total  = yolo_raw + clip_raw
    normalised = min(raw_total / 50.0, 1.0)

    return normalised, reasons


# ─── Public entry point ───────────────────────────────────────────────────────

def analyze_frames(
    frame_paths: list[str],
    yolo_model=None,
    clip_models: tuple | None = None,
) -> dict:
    """
    Run all three detection layers on the sampled frames.

    Args:
        frame_paths : paths to JPEG frames produced by video_processing.py
        yolo_model  : loaded YOLO model (pass None to load here — slow)
        clip_models : (CLIPProcessor, CLIPModel) tuple or None

    Returns:
        {
          "visual_score"        : 0–100 int,
          "raw_score"           : float,
          "detections"          : list[dict],   all individual hits
          "annotated_frames"    : list[ndarray], RGB images with bboxes
          "qr_codes_found"      : list[str],    decoded QR strings
          "clip_classifications": list[dict],   CLIP results
          "reasons"             : list[str],
        }
    """
    if not frame_paths:
        return {
            "visual_score": 0, "raw_score": 0.0,
            "detections": [], "annotated_frames": [],
            "qr_codes_found": [], "clip_classifications": [], "reasons": [],
        }

    # ── Layer 1: YOLO ─────────────────────────────────────────────────────────
    if yolo_model is None:
        yolo_model = load_yolo_model()

    yolo_detections, annotated_frames = run_yolo_on_frames(frame_paths, yolo_model)

    # ── Layer 2: QR codes ─────────────────────────────────────────────────────
    qr_detections = run_qr_detection(frame_paths)
    qr_codes_found = [d["decoded"] for d in qr_detections if "decoded" in d]

    # ── Layer 3: CLIP ─────────────────────────────────────────────────────────
    clip_detections: list[dict] = []
    if ENABLE_CLIP_ANALYSIS:
        if clip_models is None:
            clip_processor, clip_model_obj = load_clip_model()
        else:
            clip_processor, clip_model_obj = clip_models
        clip_detections = run_clip_on_frames(frame_paths, clip_processor, clip_model_obj)

    all_detections = yolo_detections + qr_detections + clip_detections

    # ── Aggregate score ───────────────────────────────────────────────────────
    normalised, reasons = _aggregate_visual_score(all_detections)
    visual_score = round(normalised * 100)

    return {
        "visual_score":         visual_score,
        "raw_score":            normalised,
        "detections":           all_detections,
        "annotated_frames":     annotated_frames,      # numpy RGB arrays
        "qr_codes_found":       qr_codes_found,
        "clip_classifications": clip_detections,
        "reasons":              reasons,
    }
