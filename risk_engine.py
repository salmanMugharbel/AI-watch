"""
risk_engine.py
Dynamic four-component risk scoring engine.

Improvements over v1:
  1. Adaptive weights   — missing modalities have their weight redistributed
                          to available ones instead of scoring zero.
  2. Dominant-signal    — if any single component is near-certain, the final
     floors               score cannot fall below a configured floor.
  3. Convergence boost  — multiple independent components agreeing gets a
                          multiplier (2 strong = ×1.10, 3 = ×1.20, 4 = ×1.30).
  4. LLM boost          — optional additive points for each indirect scam
                          pattern type detected by the LLM reasoning layer.
  5. Backward-compat    — visual_result and llm_result default to None so
                          existing callers without them still work.
"""

from __future__ import annotations
from config import (
    KEYWORD_WEIGHT, NLP_WEIGHT, PATTERN_WEIGHT, VISUAL_WEIGHT,
    RISK_LEVELS,
    NLP_FLOOR_85, NLP_FLOOR_92, KW_FLOOR_80, VIS_FLOOR_85,
    CONVERGENCE_STRONG_THRESHOLD,
    CONVERGENCE_BOOST_2, CONVERGENCE_BOOST_3, CONVERGENCE_BOOST_4,
    LLM_EUPHEMISM_BOOST, LLM_CODED_LANG_BOOST,
    LLM_IMPLIED_BOOST, LLM_SOCIAL_PROOF_BOOST,
)

CATEGORY_SEVERITY: dict[str, float] = {
    "gambling":          1.0,
    "crypto_scam":       1.4,
    "investment_fraud":  1.5,
    "ponzi_pyramid":     1.5,
    "fake_giveaway":     1.3,
    "urgency_pressure":  0.8,
    "referral_fraud":    1.3,
}

CATEGORY_LABELS: dict[str, str] = {
    "gambling":          "Gambling / Betting Advertisement",
    "crypto_scam":       "Cryptocurrency Scam",
    "investment_fraud":  "Investment Fraud",
    "ponzi_pyramid":     "Pyramid / Ponzi Scheme",
    "fake_giveaway":     "Fake Giveaway / Prize Scam",
    "urgency_pressure":  "High-Pressure Urgency Tactics",
    "referral_fraud":    "Referral / Affiliate Fraud",
}


# ─── Per-component scorers ────────────────────────────────────────────────────

def _keyword_component(kw_result: dict) -> tuple[float, list[str]]:
    reasons: list[str] = []
    weighted_sum = 0.0
    total_weight = 0.0

    for cat, data in kw_result["per_category"].items():
        severity  = CATEGORY_SEVERITY.get(cat, 1.0)
        cat_score = data["normalized_score"] * severity
        weighted_sum += cat_score
        total_weight += severity

        if data["hits"]:
            label = CATEGORY_LABELS.get(cat, cat)
            kws   = ", ".join(f'"{h}"' for h in data["hits"][:5])
            reasons.append(f"{label} keywords detected: {kws}")

    score = min(weighted_sum / max(total_weight, 1.0), 1.0)
    return score, reasons


def _nlp_component(nlp_result: dict) -> tuple[float, list[str]]:
    score     = nlp_result.get("top_scam_score", 0.0)
    top_label = nlp_result.get("top_label", "")
    reasons: list[str] = []

    if score > 0.5:
        tier1_note = " (Tier-1 confirmed)" if nlp_result.get("tier1_triggered") else ""
        reasons.append(
            f'NLP model flagged: "{top_label}" — {score:.0%} confidence{tier1_note}'
        )

    for label, conf in nlp_result.get("scores", {}).items():
        if conf > 0.60 and "legitimate" not in label.lower():
            reasons.append(f'  • {label.capitalize()} — {conf:.0%}')

    return score, reasons


def _pattern_component(
    transcript_text: str,
    ocr_text: str,
    kw_result: dict,
) -> tuple[float, list[str]]:
    import re
    reasons: list[str] = []
    signals   = 0.0
    max_sigs  = 5.0
    combined  = (transcript_text + " " + ocr_text).lower()

    if re.search(r"(bit\.ly|t\.me|telegram|whatsapp\.com/invite)", combined):
        signals += 1.5
        reasons.append("Suspicious link / invite (Telegram / BitLy) detected")

    if re.search(r"\b(whatsapp|call us|contact us|واتساب|تواصل)\b", combined):
        signals += 1.0
        reasons.append("Direct contact CTA (WhatsApp / phone) detected")

    money_matches = re.findall(r"\$[\d,]+|\d+\s*(?:usd|dollars|ريال|دولار)", combined)
    if len(money_matches) >= 2:
        signals += 1.0
        reasons.append(f"Multiple monetary amounts: {money_matches[:4]}")

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
            f"Multiple scam categories active simultaneously: {', '.join(active_cats)}"
        )

    return min(signals / max_sigs, 1.0), reasons


def _visual_component(visual_result: dict | None) -> tuple[float, list[str]]:
    if visual_result is None:
        return 0.0, []

    score   = min(visual_result.get("visual_score", 0) / 100.0, 1.0)
    reasons = list(visual_result.get("reasons", []))

    for qr in visual_result.get("qr_codes_found", [])[:3]:
        reasons.append(f'QR code decoded: "{qr[:80]}"')

    return score, reasons


# ─── Adaptive weight redistribution ──────────────────────────────────────────

def _adaptive_weights(
    has_speech: bool,
    has_ocr: bool,
    has_visual: bool,
) -> dict[str, float]:
    """
    Redistribute the base weights when a modality is unavailable.
    The goal is that absent evidence does NOT pull the score toward zero —
    instead the available evidence carries the full inferential burden.
    """
    w = {
        "keyword": KEYWORD_WEIGHT,
        "nlp":     NLP_WEIGHT,
        "pattern": PATTERN_WEIGHT,
        "visual":  VISUAL_WEIGHT,
    }

    has_text = has_speech or has_ocr

    if not has_visual and has_text:
        # Redistribute visual weight to NLP (primary) and keyword (secondary)
        extra = w["visual"]
        w["nlp"]     += extra * 0.55
        w["keyword"] += extra * 0.30
        w["pattern"] += extra * 0.15
        w["visual"]   = 0.0

    elif not has_text and has_visual:
        # No text at all — push everything to visual
        extra = w["nlp"] + w["keyword"] + w["pattern"]
        w["visual"]  += extra
        w["nlp"]      = 0.0
        w["keyword"]  = 0.0
        w["pattern"]  = 0.0

    # Round to avoid floating-point drift
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    return w


# ─── Dominant-signal floor ────────────────────────────────────────────────────

def _apply_floors(
    score: float,
    kw_score: float,
    nlp_score: float,
    visual_score: float,
) -> float:
    """
    Prevent near-certain single signals from being dragged below a minimum
    by the absence of supporting evidence from other modalities.
    """
    if nlp_score > 0.92:
        score = max(score, NLP_FLOOR_92)
    elif nlp_score > 0.85:
        score = max(score, NLP_FLOOR_85)

    if kw_score > 0.80:
        score = max(score, KW_FLOOR_80)

    if visual_score > 0.85:
        score = max(score, VIS_FLOOR_85)

    return score


# ─── Convergence boost ────────────────────────────────────────────────────────

def _apply_convergence_boost(score: float, component_scores: dict) -> float:
    """
    When multiple independent modalities agree (each scoring above the
    CONVERGENCE_STRONG_THRESHOLD), reward convergence with a multiplier.
    This reflects real-world fraud analysis where corroborating evidence
    across different channels greatly increases certainty.
    """
    strong = sum(
        1 for v in component_scores.values()
        if v > CONVERGENCE_STRONG_THRESHOLD
    )
    multiplier = {0: 1.0, 1: 1.0, 2: CONVERGENCE_BOOST_2,
                  3: CONVERGENCE_BOOST_3}.get(strong, CONVERGENCE_BOOST_4)
    return min(score * multiplier, 100.0)


# ─── LLM reasoning boost ─────────────────────────────────────────────────────

def _apply_llm_boost(score: float, llm_result: dict | None) -> tuple[float, list[str]]:
    """
    Add raw points for each indirect scam pattern type detected by the LLM
    reasoning layer, scaled by LLM confidence.
    """
    if llm_result is None or not llm_result.get("ran", False):
        return score, []

    conf    = llm_result.get("confidence", 0.0)
    reasons: list[str] = []
    boost   = 0.0

    if llm_result.get("euphemism_detected"):
        boost += LLM_EUPHEMISM_BOOST * conf
        reasons.append(f"LLM: Euphemistic language detected ({conf:.0%} confidence)")

    if llm_result.get("coded_language_detected"):
        boost += LLM_CODED_LANG_BOOST * conf
        reasons.append(f"LLM: Coded financial language detected ({conf:.0%} confidence)")

    if llm_result.get("implied_promises_detected"):
        boost += LLM_IMPLIED_BOOST * conf
        reasons.append(f"LLM: Implied wealth promises detected ({conf:.0%} confidence)")

    if llm_result.get("social_proof_manipulation"):
        boost += LLM_SOCIAL_PROOF_BOOST * conf
        reasons.append(f"LLM: Social-proof manipulation detected ({conf:.0%} confidence)")

    if llm_result.get("flagged_phrases"):
        phrases = ", ".join(f'"{p}"' for p in llm_result["flagged_phrases"][:4])
        reasons.append(f"LLM flagged phrases: {phrases}")

    if llm_result.get("explanation"):
        reasons.append(f"LLM analysis: {llm_result['explanation']}")

    return min(score + boost, 100.0), reasons


# ─── Public interface ─────────────────────────────────────────────────────────

def compute_risk(
    transcript_text: str,
    ocr_text: str,
    analysis: dict,
    visual_result: dict | None = None,
    llm_result: dict | None = None,
) -> dict:
    """
    Compute the final risk score from all available evidence.

    Args:
        transcript_text : Whisper transcript (empty string if no audio)
        ocr_text        : Combined EasyOCR text (empty if no overlays)
        analysis        : Output from nlp_analysis.analyze()
        visual_result   : Output from vision_analysis.analyze_frames() or None
        llm_result      : Output from llm_reasoning.reason() or None

    Returns a complete risk assessment dict consumed by app.py.
    """
    kw_result  = analysis["keyword"]
    nlp_result = analysis["nlp"]

    # ── Score each component (0–1) ────────────────────────────────────────────
    kw_score,      kw_reasons      = _keyword_component(kw_result)
    nlp_score,     nlp_reasons     = _nlp_component(nlp_result)
    pattern_score, pattern_reasons = _pattern_component(transcript_text, ocr_text, kw_result)
    visual_score,  visual_reasons  = _visual_component(visual_result)

    # ── Adaptive weights ──────────────────────────────────────────────────────
    has_speech = len(transcript_text.strip()) > 20
    has_ocr    = len(ocr_text.strip())        > 10
    has_visual = visual_result is not None and visual_result.get("visual_score", 0) > 0

    w = _adaptive_weights(has_speech, has_ocr, has_visual)

    # ── Weighted raw score (0–100) ────────────────────────────────────────────
    raw = (
        w["keyword"] * kw_score
        + w["nlp"]     * nlp_score
        + w["pattern"] * pattern_score
        + w["visual"]  * visual_score
    )
    score = min(raw * 100, 100.0)

    # ── Compile component scores for UI and for the boost functions ───────────
    component_scores = {
        "keyword": round(kw_score     * 100),
        "nlp":     round(nlp_score    * 100),
        "pattern": round(pattern_score * 100),
        "visual":  round(visual_score  * 100),
    }

    # ── Dominant-signal floors ────────────────────────────────────────────────
    score = _apply_floors(score, kw_score, nlp_score, visual_score)

    # ── Convergence boost ─────────────────────────────────────────────────────
    score = _apply_convergence_boost(score, component_scores)

    # ── LLM reasoning boost ───────────────────────────────────────────────────
    score, llm_reasons = _apply_llm_boost(score, llm_result)

    risk_score = round(min(score, 100))

    # ── Risk level ────────────────────────────────────────────────────────────
    risk_level, color, icon = "Unknown", "grey", "❓"
    for (lo, hi), (level, clr, icn) in RISK_LEVELS.items():
        if lo <= risk_score <= hi:
            risk_level, color, icon = level, clr, icn
            break

    # ── Reasons (deduplicated) ────────────────────────────────────────────────
    all_reasons = (
        kw_reasons + nlp_reasons + pattern_reasons
        + visual_reasons + llm_reasons
        + analysis.get("extra_reasons", [])
    )
    seen: set[str] = set()
    deduped: list[str] = []
    for r in all_reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)

    # ── Top categories for bar chart ──────────────────────────────────────────
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
        "component_scores": component_scores,
        "weights_used":     {k: round(v, 3) for k, v in w.items()},
        "llm_ran":          llm_result.get("ran", False) if llm_result else False,
    }


def apply_llm_boost(preliminary: dict, llm_result: dict) -> dict:
    """
    Lightweight helper used by app.py when the LLM result arrives after the
    initial compute_risk() call (two-pass pattern).
    Mutates a copy of the preliminary dict.
    """
    result = dict(preliminary)
    boosted_score, llm_reasons = _apply_llm_boost(
        float(preliminary["risk_score"]), llm_result
    )
    final_score = round(min(boosted_score, 100))

    risk_level, color, icon = preliminary["risk_level"], preliminary["risk_color"], preliminary["risk_icon"]
    for (lo, hi), (level, clr, icn) in RISK_LEVELS.items():
        if lo <= final_score <= hi:
            risk_level, color, icon = level, clr, icn
            break

    result["risk_score"]  = final_score
    result["risk_level"]  = risk_level
    result["risk_color"]  = color
    result["risk_icon"]   = icon
    result["reasons"]     = preliminary["reasons"] + llm_reasons
    result["llm_ran"]     = True
    return result
