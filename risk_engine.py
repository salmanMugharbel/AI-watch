"""
risk_engine.py
Aggregates keyword scores, NLP classification scores, structural pattern
signals, and visual (YOLO/CLIP/QR) scores into a single 0–100 risk score
with human-readable reasons.
"""

from __future__ import annotations
from config import KEYWORD_WEIGHT, NLP_WEIGHT, PATTERN_WEIGHT, VISUAL_WEIGHT, RISK_LEVELS


# Severity multipliers per scam category (more dangerous = higher multiplier)
CATEGORY_SEVERITY: dict[str, float] = {
    "gambling":          1.0,
    "crypto_scam":       1.4,
    "investment_fraud":  1.5,
    "ponzi_pyramid":     1.5,
    "fake_giveaway":     1.3,
    "urgency_pressure":  0.8,
}

CATEGORY_LABELS: dict[str, str] = {
    "gambling":          "Gambling / Betting Advertisement",
    "crypto_scam":       "Cryptocurrency Scam",
    "investment_fraud":  "Investment Fraud",
    "ponzi_pyramid":     "Pyramid / Ponzi Scheme",
    "fake_giveaway":     "Fake Giveaway / Prize Scam",
    "urgency_pressure":  "High-Pressure Urgency Tactics",
}


def _keyword_component(kw_result: dict) -> tuple[float, list[str]]:
    """0–1 score derived from the keyword analysis dict."""
    reasons: list[str] = []
    weighted_sum = 0.0
    total_weight = 0.0

    for cat, data in kw_result["per_category"].items():
        severity = CATEGORY_SEVERITY.get(cat, 1.0)
        cat_score = data["normalized_score"] * severity
        weighted_sum += cat_score
        total_weight += severity

        if data["hits"]:
            label = CATEGORY_LABELS.get(cat, cat)
            kws = ", ".join(f'"{h}"' for h in data["hits"][:5])
            reasons.append(f"{label} keywords detected: {kws}")

    score = min(weighted_sum / max(total_weight, 1.0), 1.0)
    return score, reasons


def _nlp_component(nlp_result: dict) -> tuple[float, list[str]]:
    """0–1 score from zero-shot / Ollama classification."""
    score = nlp_result.get("top_scam_score", 0.0)
    top_label = nlp_result.get("top_label", "")
    reasons: list[str] = []

    if score > 0.5:
        reasons.append(f'NLP model flagged content as: "{top_label}" (confidence {score:.0%})')

    for label, conf in nlp_result.get("scores", {}).items():
        if conf > 0.65 and "legitimate" not in label.lower():
            reasons.append(f'  • {label.capitalize()} — {conf:.0%} confidence')

    return score, reasons


def _pattern_component(
    transcript_text: str,
    ocr_text: str,
    kw_result: dict,
) -> tuple[float, list[str]]:
    """0–1 score from structural signals (URLs, CAPS, multi-category hits)."""
    import re
    reasons: list[str] = []
    signals = 0.0
    max_signals = 5.0

    combined = (transcript_text + " " + ocr_text).lower()

    if re.search(r"(bit\.ly|t\.me|telegram|whatsapp\.com/invite)", combined):
        signals += 1.5
        reasons.append("Suspicious link / invite (Telegram / BitLy) detected")

    if re.search(r"\b(whatsapp|call us|contact us|واتساب|تواصل)\b", combined):
        signals += 1.0
        reasons.append("Direct contact CTA (WhatsApp / phone) detected")

    money_matches = re.findall(r"\$[\d,]+|\d+\s*(?:usd|dollars|ريال|دولار)", combined)
    if len(money_matches) >= 2:
        signals += 1.0
        reasons.append(f"Multiple monetary amounts mentioned: {money_matches[:4]}")

    caps_ratio = sum(1 for c in (transcript_text + ocr_text) if c.isupper()) / max(
        len(transcript_text + ocr_text), 1
    )
    if caps_ratio > 0.35:
        signals += 0.8
        reasons.append("Excessive capitalisation typical of scam ads")

    active_cats = [
        cat for cat, d in kw_result["per_category"].items()
        if d["normalized_score"] > 0.2
    ]
    if len(active_cats) >= 3:
        signals += 1.0
        reasons.append(
            f"Multiple scam categories triggered simultaneously: {', '.join(active_cats)}"
        )

    return min(signals / max_signals, 1.0), reasons


def _visual_component(visual_result: dict | None) -> tuple[float, list[str]]:
    """
    0–1 score from the visual analysis module (YOLOv8 + pyzbar + CLIP).
    Accepts None gracefully so the risk engine works even when vision is skipped.
    """
    if visual_result is None:
        return 0.0, []

    # visual_score is already 0–100; normalise to 0–1
    score = min(visual_result.get("visual_score", 0) / 100.0, 1.0)
    reasons = visual_result.get("reasons", [])

    # Add a note if QR codes were found — always a strong indicator
    qr_codes = visual_result.get("qr_codes_found", [])
    if qr_codes:
        for qr in qr_codes[:3]:
            reasons.append(f'QR code detected (decoded: "{qr[:80]}")')

    return score, reasons


def compute_risk(
    transcript_text: str,
    ocr_text: str,
    analysis: dict,
    visual_result: dict | None = None,
) -> dict:
    """
    Entry point for the risk engine.

    Args:
        transcript_text : Full Whisper transcript.
        ocr_text        : Combined EasyOCR text from frames.
        analysis        : Output from nlp_analysis.analyze().
        visual_result   : Output from vision_analysis.analyze_frames(), or None.

    Returns a complete risk assessment dict consumed by app.py.
    """
    kw_result  = analysis["keyword"]
    nlp_result = analysis["nlp"]

    kw_score,      kw_reasons      = _keyword_component(kw_result)
    nlp_score,     nlp_reasons     = _nlp_component(nlp_result)
    pattern_score, pattern_reasons = _pattern_component(transcript_text, ocr_text, kw_result)
    visual_score,  visual_reasons  = _visual_component(visual_result)

    # ── Weighted combination (four components, weights sum to 1.0) ────────────
    raw = (
        KEYWORD_WEIGHT * kw_score
        + NLP_WEIGHT   * nlp_score
        + PATTERN_WEIGHT * pattern_score
        + VISUAL_WEIGHT  * visual_score
    )
    risk_score = round(min(raw * 100, 100))

    # ── Risk level lookup ─────────────────────────────────────────────────────
    risk_level, color, icon = "Unknown", "grey", "❓"
    for (lo, hi), (level, clr, icn) in RISK_LEVELS.items():
        if lo <= risk_score <= hi:
            risk_level, color, icon = level, clr, icn
            break

    # ── Deduplicated reasons list ─────────────────────────────────────────────
    all_reasons = (
        kw_reasons + nlp_reasons + pattern_reasons
        + visual_reasons + analysis.get("extra_reasons", [])
    )
    seen: set[str] = set()
    deduped: list[str] = []
    for r in all_reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)

    # ── Top scam categories (for bar chart in UI) ─────────────────────────────
    top_categories = sorted(
        [
            (CATEGORY_LABELS.get(cat, cat), round(d["normalized_score"] * 100))
            for cat, d in kw_result["per_category"].items()
            if d["normalized_score"] > 0.05
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "risk_score":       risk_score,
        "risk_level":       risk_level,
        "risk_color":       color,
        "risk_icon":        icon,
        "reasons":          deduped,
        "top_categories":   top_categories,
        "matched_keywords": kw_result["all_matched_keywords"],
        "nlp_scores":       nlp_result.get("scores", {}),
        "component_scores": {
            "keyword": round(kw_score     * 100),
            "nlp":     round(nlp_score    * 100),
            "pattern": round(pattern_score * 100),
            "visual":  round(visual_score  * 100),   # ← new
        },
    }
