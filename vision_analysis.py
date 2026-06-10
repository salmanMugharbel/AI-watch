"""
vision_analysis.py
Three-layer visual scam detection.

Key improvements over v1:
  1. CLIP uses per-prompt cosine similarity (not softmax) — each prompt
     gets an independent absolute score so thresholds are meaningful and
     cross-video comparable.
  2. Expanded CLIP prompt set: 20 prompts across 5 scam categories with
     multiple phrasings per category (CLIP responds better to paraphrasing).
  3. YOLO contextual gate — a YOLO detection only contributes to the score
     when the same frame also scores above YOLO_CLIP_GATE_THRESHOLD on at
     least one CLIP scam prompt.  This eliminates false positives from cars
     in travel vlogs and laptops in tech tutorials.
  4. pyzbar QR decoding unchanged — it was already correct.
"""

from __future__ import annotations
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from config import (
    YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_MAX_ANNOTATED_FRAMES,
    YOLO_CLIP_GATE_THRESHOLD,
    CLIP_MODEL, CLIP_COSINE_THRESHOLD, ENABLE_CLIP_ANALYSIS,
)


# ─── COCO class → risk mapping ────────────────────────────────────────────────
# Scores are intentionally conservative because YOLO detections now require
# CLIP gate confirmation before they contribute.

YOLO_RISK_MAP: dict[str, dict] = {
    "car":        {"score": 10, "category": "Luxury Lifestyle",   "label": "Luxury Car / Vehicle"},
    "motorcycle": {"score":  6, "category": "Luxury Lifestyle",   "label": "Motorcycle"},
    "cell phone": {"score": 14, "category": "Contact CTA",        "label": "Mobile / Messaging Device"},
    "laptop":     {"score":  8, "category": "Trading Interface",  "label": "Laptop / Screen"},
    "tv":         {"score":  6, "category": "Screen Display",     "label": "Display Screen"},
    "clock":      {"score":  6, "category": "Luxury Lifestyle",   "label": "Watch / Clock"},
    "handbag":    {"score":  7, "category": "Luxury Goods",       "label": "Luxury Handbag"},
    "suitcase":   {"score":  5, "category": "Luxury Lifestyle",   "label": "Luggage / Travel"},
    "person":     {"score":  3, "category": "Social Proof",       "label": "Person (social proof)"},
    "wine glass": {"score":  5, "category": "Luxury Lifestyle",   "label": "Luxury Party Scene"},
    "bottle":     {"score":  4, "category": "Luxury Lifestyle",   "label": "Bottle / Party Scene"},
    "tie":        {"score":  4, "category": "Social Proof",       "label": "Formal Attire (authority)"},
}

CATEGORY_COLORS: dict[str, tuple] = {
    "Luxury Lifestyle":  (0, 215, 255),
    "Contact CTA":       (0, 100, 255),
    "Trading Interface": (255, 180, 0),
    "Screen Display":    (200, 200, 0),
    "Luxury Goods":      (0, 215, 255),
    "Social Proof":      (180, 180, 180),
    "QR Code":           (0, 0, 255),
}


# ─── CLIP prompt set ──────────────────────────────────────────────────────────
# Multiple phrasings per scam category improve recall.
# Format: (prompt_text, base_score, display_label, category_group)

CLIP_PROMPTS: list[tuple[str, int, str, str]] = [
    # Gambling
    ("a slot machine or casino gambling website on a screen",            26, "Casino / Slot Interface",        "gambling"),
    ("a sports betting website showing live odds and matches",           22, "Sports Betting Platform",        "gambling"),
    ("a roulette wheel or poker table in a casino",                     24, "Casino Table Games",             "gambling"),
    ("an online gambling app with chips and spin buttons",              24, "Gambling App",                   "gambling"),

    # Crypto / Trading
    ("a candlestick trading chart on a cryptocurrency exchange",        23, "Crypto Trading Chart",           "crypto"),
    ("a Bitcoin or Ethereum price ticker with percentage gains",        22, "Crypto Price Ticker",            "crypto"),
    ("a crypto wallet showing balance and transaction history",         20, "Crypto Wallet Screen",           "crypto"),
    ("a decentralized finance DeFi yield farming dashboard",            20, "DeFi Dashboard",                 "crypto"),

    # Investment / Scam ads
    ("a financial investment opportunity advertisement with big profits",22, "Investment Scam Ad",            "investment"),
    ("a get-rich-quick scheme showing money and luxury lifestyle",       22, "Get-Rich-Quick Ad",             "investment"),
    ("a person holding large amounts of cash near a luxury car",        18, "Cash and Luxury Display",        "investment"),

    # Messaging / Routing
    ("a Telegram group invitation or Telegram channel screen",          15, "Telegram Invite",                "messaging"),
    ("a WhatsApp chat showing a phone number or group invite link",     14, "WhatsApp Contact CTA",           "messaging"),
    ("a QR code displayed prominently on screen",                       18, "QR Code Screen",                 "messaging"),
    ("a referral program screen with bonus invite link",                18, "Referral / Invite Screen",       "messaging"),

    # Luxury lifestyle (context signal)
    ("a luxury sports car like a Lamborghini Ferrari or Rolls-Royce",  14, "Luxury Sports Car",              "luxury"),
    ("a person wearing expensive watches like Rolex with cash",         14, "Luxury Watch and Cash",          "luxury"),
    ("stacks of cash or gold bars as a display of wealth",             16, "Cash / Gold Display",             "luxury"),
    ("a private jet or yacht as a symbol of wealth",                   12, "Luxury Travel Display",           "luxury"),

    # Control — normal content (score 0, used only as a reference)
    ("a normal everyday video with no financial or scam content",        0, "Normal / Clean Content",         "control"),
]

SCAM_PROMPTS = [(p, s, l, g) for p, s, l, g in CLIP_PROMPTS if g != "control"]


# ─── Model loaders ────────────────────────────────────────────────────────────

def load_yolo_model():
    from ultralytics import YOLO
    return YOLO(YOLO_MODEL_PATH)


def load_clip_model():
    from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = HFCLIPModel.from_pretrained(CLIP_MODEL)
    model.eval()
    return processor, model


# ─── Layer 1: YOLOv8 with CLIP gate ──────────────────────────────────────────

def run_yolo_on_frames(
    frame_paths: list[str],
    yolo_model,
    clip_frame_scores: dict[int, float],   # frame_idx → max CLIP scam score
) -> tuple[list[dict], list[np.ndarray]]:
    """
    Run YOLO on every frame.  A detection is only credited when the frame's
    CLIP scam score exceeds YOLO_CLIP_GATE_THRESHOLD, preventing false
    positives from context-free COCO objects in innocent videos.
    """
    detections:       list[dict]       = []
    annotated_frames: list[np.ndarray] = []

    for frame_idx, path in enumerate(frame_paths):
        # Context gate: skip scoring (but still draw boxes) if CLIP did not
        # confirm scam context for this frame
        frame_clip_score = clip_frame_scores.get(frame_idx, 0.0)
        gate_passed      = frame_clip_score >= YOLO_CLIP_GATE_THRESHOLD

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        results     = yolo_model(path, conf=YOLO_CONFIDENCE, verbose=False)
        result      = results[0]
        frame_hits: list[dict] = []

        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            if cls_name not in YOLO_RISK_MAP:
                continue

            conf = float(box.conf)
            info = YOLO_RISK_MAP[cls_name]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detection = {
                "frame_idx":    frame_idx,
                "class_name":   cls_name,
                "label":        info["label"],
                "category":     info["category"],
                "confidence":   round(conf, 3),
                # Score is zero when the CLIP gate is not met — box is drawn
                # in the UI but does not contribute points
                "score":        info["score"] if gate_passed else 0,
                "gate_passed":  gate_passed,
                "bbox":         (x1, y1, x2, y2),
                "source":       "yolo",
            }
            frame_hits.append(detection)
            detections.append(detection)

        if frame_hits and len(annotated_frames) < YOLO_MAX_ANNOTATED_FRAMES:
            annotated = _draw_yolo_boxes(img_bgr.copy(), frame_hits)
            annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    return detections, annotated_frames


def _draw_yolo_boxes(img_bgr: np.ndarray, hits: list[dict]) -> np.ndarray:
    for hit in hits:
        x1, y1, x2, y2 = hit["bbox"]
        # Dim colour when gate not passed to visually distinguish non-scoring boxes
        color = CATEGORY_COLORS.get(hit["category"], (0, 255, 0))
        if not hit.get("gate_passed", True):
            color = tuple(max(c - 100, 0) for c in color)

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label_text = f"{hit['label']} {hit['confidence']:.0%}"
        if not hit.get("gate_passed", True):
            label_text += " [unconfirmed]"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img_bgr


# ─── Layer 2: pyzbar QR detection ────────────────────────────────────────────

def run_qr_detection(frame_paths: list[str]) -> list[dict]:
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        from PIL import Image as PILImage
    except ImportError:
        return []

    seen_data: set[str] = set()
    detections: list[dict] = []

    for frame_idx, path in enumerate(frame_paths):
        try:
            pil_img = PILImage.open(path)
            codes   = pyzbar_decode(pil_img)
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
                "confidence": 1.0,
                "score":      25,
                "gate_passed": True,
                "decoded":    decoded,
                "source":     "pyzbar",
            })

    return detections


# ─── Layer 3: CLIP cosine similarity ─────────────────────────────────────────

def run_clip_on_frames(
    frame_paths: list[str],
    clip_processor,
    clip_model,
) -> tuple[list[dict], dict[int, float]]:
    """
    Score each frame against SCAM_PROMPTS using cosine similarity.

    Returns:
        clip_detections  : list of detection dicts for hits above threshold
        frame_max_scores : dict mapping frame_idx → max cosine score across
                           all scam prompts (used as the YOLO gate signal)
    """
    from PIL import Image as PILImage

    # Pre-compute text features for all scam prompts once
    all_prompt_texts = [p for p, _, _, _ in SCAM_PROMPTS]
    text_inputs = clip_processor(
        text=all_prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=-1)

    clip_detections:  list[dict]       = []
    frame_max_scores: dict[int, float] = {}

    for frame_idx, path in enumerate(frame_paths):
        try:
            pil_img = PILImage.open(path).convert("RGB")
        except Exception:
            frame_max_scores[frame_idx] = 0.0
            continue

        image_inputs = clip_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            image_features = F.normalize(image_features, dim=-1)

        # Independent cosine similarity per prompt (not softmax)
        cosine_scores = (image_features @ text_features.T).squeeze(0).tolist()

        frame_max = 0.0
        for i, (prompt, base_score, display_label, category) in enumerate(SCAM_PROMPTS):
            cos_sim = cosine_scores[i]
            frame_max = max(frame_max, cos_sim)

            if cos_sim < CLIP_COSINE_THRESHOLD:
                continue

            clip_detections.append({
                "frame_idx":   frame_idx,
                "class_name":  "clip_classification",
                "label":       display_label,
                "category":    category,
                "confidence":  round(cos_sim, 3),
                "score":       round(base_score * cos_sim),
                "gate_passed": True,
                "prompt":      prompt,
                "source":      "clip",
            })

        frame_max_scores[frame_idx] = frame_max

    return clip_detections, frame_max_scores


# ─── Score aggregation ────────────────────────────────────────────────────────

def _aggregate_visual_score(all_detections: list[dict]) -> tuple[float, list[str]]:
    """
    Build a normalised 0–1 score and reasons list from all detections.

    YOLO/pyzbar: each unique class scores once + persistence bonus.
    CLIP:        best cosine score per label wins.
    Only detections with gate_passed=True contribute to the score.
    """
    total_frames = max(
        (max((d["frame_idx"] for d in all_detections), default=0) + 1), 1
    )

    # ── YOLO + pyzbar ─────────────────────────────────────────────────────────
    yolo_pyzbar  = [d for d in all_detections if d["source"] in ("yolo", "pyzbar")]
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
        info = class_info[cls]
        if info["score"] == 0:    # gate not passed for this class
            continue

        persistence       = len(frames_seen) / total_frames
        bonus_multiplier  = 1.0 + 0.5 * max(0.0, (persistence - 0.3) / 0.7)
        yolo_raw         += info["score"] * bonus_multiplier

        decoded_note = f' — decoded: "{info["decoded"][:60]}"' if "decoded" in info else ""
        reasons.append(
            f'{info["label"]} detected in {len(frames_seen)} frame(s)'
            f' ({info["confidence"]:.0%} confidence{decoded_note})'
        )

    # ── CLIP ──────────────────────────────────────────────────────────────────
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
            f'Scene classified as "{lbl}" '
            f'(cosine similarity {d["confidence"]:.2f})'
        )

    raw_total  = yolo_raw + clip_raw
    normalised = min(raw_total / 55.0, 1.0)   # soft cap at raw=55 → 1.0

    return normalised, reasons


# ─── Public entry point ───────────────────────────────────────────────────────

def analyze_frames(
    frame_paths: list[str],
    yolo_model=None,
    clip_models: tuple | None = None,
) -> dict:
    """
    Run all three detection layers on sampled frames.

    Returns:
        visual_score, raw_score, detections, annotated_frames,
        qr_codes_found, clip_classifications, reasons
    """
    if not frame_paths:
        return {
            "visual_score": 0, "raw_score": 0.0,
            "detections": [], "annotated_frames": [],
            "qr_codes_found": [], "clip_classifications": [], "reasons": [],
        }

    if yolo_model is None:
        yolo_model = load_yolo_model()

    # ── CLIP first (needed to gate YOLO) ──────────────────────────────────────
    clip_detections:  list[dict]       = []
    frame_max_scores: dict[int, float] = {i: 0.0 for i in range(len(frame_paths))}

    if ENABLE_CLIP_ANALYSIS:
        if clip_models is None:
            clip_processor, clip_model_obj = load_clip_model()
        else:
            clip_processor, clip_model_obj = clip_models
        clip_detections, frame_max_scores = run_clip_on_frames(
            frame_paths, clip_processor, clip_model_obj
        )

    # ── YOLO with CLIP gate ───────────────────────────────────────────────────
    yolo_detections, annotated_frames = run_yolo_on_frames(
        frame_paths, yolo_model, frame_max_scores
    )

    # ── QR codes ─────────────────────────────────────────────────────────────
    qr_detections  = run_qr_detection(frame_paths)
    qr_codes_found = [d["decoded"] for d in qr_detections if "decoded" in d]

    all_detections = yolo_detections + qr_detections + clip_detections

    normalised, reasons = _aggregate_visual_score(all_detections)
    visual_score = round(normalised * 100)

    return {
        "visual_score":         visual_score,
        "raw_score":            normalised,
        "detections":           all_detections,
        "annotated_frames":     annotated_frames,
        "qr_codes_found":       qr_codes_found,
        "clip_classifications": clip_detections,
        "reasons":              reasons,
    }
