"""
nlp_analysis.py
Three-layer text-based scam analysis:

  1. Keyword engine      — fast, deterministic, language-agnostic
  2. Two-pass BART-MNLI  — Tier-1 triage → Tier-2 fine-grained 15-label
                           classification on chunked text input
  3. Ollama backend      — optional replacement (NLP_BACKEND="ollama")

Key improvements over v1:
  - 15 specific financial-crime labels (up from 9 broad ones)
  - Hierarchical two-pass approach prevents probability dilution
  - Chunked text input: entire transcript is scored, not just first 1000 chars
  - Tier-1 confirmation boosts Tier-2 scam scores by a calibrated factor
"""

from __future__ import annotations
import re
import json
import requests
from config import (
    NLP_BACKEND, NLP_MODEL,
    OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SEC,
    NLP_CHUNK_SIZE, NLP_CHUNK_OVERLAP, NLP_MAX_CHUNKS,
    NLP_TIER1_TRIGGER_CONF, NLP_TIER2_BOOST_FACTOR,
)


# ─── Keyword dictionary ───────────────────────────────────────────────────────
# (keyword, weight) tuples.  Weight > 1 = high-signal phrase.

SCAM_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "gambling": [
        ("casino", 1.0), ("online casino", 1.5), ("betting", 1.0),
        ("jackpot", 1.0), ("slot machine", 1.2), ("poker", 0.8),
        ("roulette", 1.0), ("sports betting", 1.2), ("bet now", 1.5),
        ("win big", 1.3), ("free spins", 1.2), ("gambling", 1.2),
        ("sportsbook", 1.3), ("fixed match", 2.0), ("odds guaranteed", 2.0),
        ("كازينو", 1.5), ("مراهنات", 1.5), ("يانصيب", 1.2), ("رهان", 1.3),
    ],
    "crypto_scam": [
        ("send btc", 2.0), ("send bitcoin", 2.0), ("send eth", 2.0),
        ("100x", 1.5), ("1000x", 2.0), ("to the moon", 1.3),
        ("guaranteed profit", 2.0), ("free crypto", 1.8),
        ("airdrop", 1.3), ("crypto signals", 1.5),
        ("nft drop", 1.2), ("defi yield", 1.2), ("invest in crypto", 1.3),
        ("double your bitcoin", 2.5), ("crypto giveaway", 2.0),
        ("blockchain investment", 1.3), ("mining profit", 1.2),
        ("wallet address", 1.0), ("private sale", 1.5), ("presale", 1.3),
        ("token launch", 1.2), ("10x guaranteed", 2.5),
    ],
    "investment_fraud": [
        ("guaranteed returns", 2.5), ("risk-free", 2.0), ("risk free", 2.0),
        ("double your money", 2.5), ("triple your money", 2.5),
        ("passive income", 1.2), ("financial freedom", 1.3),
        ("forex signals", 1.5), ("trading bot", 1.3), ("100% profit", 2.5),
        ("no risk", 2.0), ("profit guaranteed", 2.5), ("high returns", 1.5),
        ("exclusive opportunity", 1.5), ("insider tip", 1.8),
        ("invest now", 1.2), ("limited time offer", 1.3),
        ("unlicensed broker", 2.0), ("unregulated", 1.5),
        ("daily profit", 1.8), ("monthly return", 1.5),
        ("مضاعفة أموالك", 2.5), ("ربح مضمون", 2.5), ("استثمار مضمون", 2.0),
        ("عائد يومي", 2.0), ("ربح يومي", 2.0),
    ],
    "ponzi_pyramid": [
        ("referral bonus", 1.8), ("recruit friends", 1.8), ("downline", 2.0),
        ("upline", 2.0), ("mlm", 1.8), ("network marketing", 1.5),
        ("multi-level", 1.8), ("join our team", 1.2), ("earn from referrals", 2.0),
        ("pyramid scheme", 3.0), ("ponzi", 3.0), ("chain letter", 2.5),
        ("make money from home", 1.5), ("unlimited earnings", 1.8),
        ("passive referral income", 2.0), ("team building bonus", 1.8),
        ("matrix plan", 2.0), ("binary plan", 1.8),
    ],
    "fake_giveaway": [
        ("giveaway", 1.2), ("free money", 2.0), ("you won", 2.0),
        ("congratulations you have been selected", 2.5),
        ("claim your prize", 2.0), ("processing fee", 2.5),
        ("advance fee", 3.0), ("send fee", 2.5), ("free gift", 1.5),
        ("double giveaway", 2.5), ("elon musk giveaway", 3.0),
        ("celebrity giveaway", 2.5), ("tesla giveaway", 2.5),
        ("send 0.5 btc get 1 btc", 3.0), ("we are giving back", 2.0),
    ],
    "urgency_pressure": [
        ("act now", 1.5), ("limited time", 1.2), ("expires today", 1.8),
        ("last chance", 1.5), ("hurry up", 1.3), ("only today", 1.8),
        ("join now before", 1.5), ("register today", 1.2), ("deadline", 1.0),
        ("spots are filling", 1.5), ("don't miss out", 1.2),
        ("offer ends", 1.5), ("closing soon", 1.5), ("seats limited", 1.5),
    ],
    "referral_fraud": [
        ("referral link", 1.8), ("my referral code", 1.5),
        ("sign up using my link", 1.8), ("use my code", 1.5),
        ("referral commission", 2.0), ("earn per referral", 2.0),
        ("affiliate commission", 1.5), ("refer and earn", 1.8),
        ("invite friends earn", 1.8),
    ],
}


# ─── BART-MNLI label sets ─────────────────────────────────────────────────────

# Tier 1 — fast triage (3 broad labels).  Run first to decide if Tier 2 is
# needed and to provide a confirmation signal for boosting Tier-2 scores.
TIER1_LABELS = [
    "financial scam, fraud, or illegal investment scheme",
    "gambling, casino, or betting promotion",
    "legitimate everyday content with no scam indicators",
]

# Tier 2 — 15 specific financial-crime sub-labels.
# The first 13 are scam labels; the last 2 are control/legitimate labels.
TIER2_LABELS = [
    "illegal online gambling operation",
    "casino promotion or gambling advertisement",
    "sports betting promotion or fixed match offer",
    "cryptocurrency scam or fake crypto investment",
    "fraudulent investment promising guaranteed profit",
    "Ponzi scheme or chain investment fraud",
    "pyramid scheme or MLM recruitment fraud",
    "referral fraud or affiliate recruitment scam",
    "high-pressure financial marketing with false urgency",
    "get-rich-quick scheme or passive income fraud",
    "financial market manipulation or pump and dump",
    "unlicensed financial service or unregulated broker",
    "fraudulent wealth display or fake success lifestyle",
    "legitimate financial education or investment news",
    "legitimate product or service advertisement",
]

TIER2_SCAM_COUNT = 13   # first N labels in TIER2_LABELS are scam


# ─── Text chunking ────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks so the NLP model sees the whole
    transcript.  Returns a list of chunks capped at NLP_MAX_CHUNKS.
    The chunk with the highest scam score will drive the final result.
    """
    if len(text) <= NLP_CHUNK_SIZE:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < NLP_MAX_CHUNKS:
        end = min(start + NLP_CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += NLP_CHUNK_SIZE - NLP_CHUNK_OVERLAP

    return chunks


# ─── Keyword analysis ─────────────────────────────────────────────────────────

def keyword_analysis(text: str) -> dict:
    """
    Scan combined text for scam keywords.
    Returns per-category scores, hits, and an aggregate 0–1 score.
    """
    text_lower = text.lower()
    results: dict[str, dict] = {}

    for category, keywords in SCAM_KEYWORDS.items():
        hits: list[str] = []
        score = 0.0
        for kw, weight in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            matches = re.findall(pattern, text_lower)
            if matches:
                hits.extend(matches)
                score += weight * len(matches)
        results[category] = {
            "hits":             list(set(hits)),
            "raw_score":        score,
            "normalized_score": min(score / 5.0, 1.0),
        }

    all_scores  = [v["normalized_score"] for v in results.values()]
    aggregate   = min(sum(all_scores) / len(all_scores) * 2.5, 1.0)
    all_hits    = [h for v in results.values() for h in v["hits"]]

    return {
        "per_category":         results,
        "aggregate_score":      aggregate,
        "all_matched_keywords": list(set(all_hits)),
    }


# ─── Transformers two-pass zero-shot classification ───────────────────────────

def load_zs_classifier():
    from transformers import pipeline
    clf = pipeline(
        "zero-shot-classification",
        model=NLP_MODEL,
        device=-1,
    )
    return clf


def classify_with_transformers(text: str, classifier=None) -> dict:
    """
    Two-pass hierarchical zero-shot classification.

    Pass 1: 3-label triage to determine if the text is broadly scam-related.
    Pass 2: 15-label fine-grained classification on each text chunk.
             If Pass 1 flagged the text, Tier-2 scam scores are boosted.

    Returns the result from the chunk with the highest scam confidence.
    """
    if not text.strip():
        return {
            "scores": {}, "top_scam_score": 0.0,
            "top_label": "none", "tier1_triggered": False,
        }

    if classifier is None:
        classifier = load_zs_classifier()

    # ── Pass 1: triage ────────────────────────────────────────────────────────
    # Use only the first 600 chars — we just want a fast signal
    t1 = classifier(text[:600], TIER1_LABELS, multi_label=False)
    t1_scores = dict(zip(t1["labels"], t1["scores"]))

    tier1_scam_conf = (
        t1_scores.get(TIER1_LABELS[0], 0.0) +   # financial fraud
        t1_scores.get(TIER1_LABELS[1], 0.0)      # gambling
    )
    tier1_triggered = tier1_scam_conf >= NLP_TIER1_TRIGGER_CONF

    # ── Pass 2: fine-grained on all chunks ────────────────────────────────────
    chunks = _chunk_text(text)
    best_scores: dict[str, float] = {}
    best_top_scam = 0.0

    for chunk in chunks:
        r = classifier(chunk, TIER2_LABELS, multi_label=True)
        chunk_label_scores = dict(zip(r["labels"], r["scores"]))

        # Apply Tier-1 boost to scam labels when Tier-1 confirmed
        if tier1_triggered:
            for lbl in TIER2_LABELS[:TIER2_SCAM_COUNT]:
                if lbl in chunk_label_scores:
                    chunk_label_scores[lbl] = min(
                        chunk_label_scores[lbl] * NLP_TIER2_BOOST_FACTOR, 1.0
                    )

        chunk_top_scam = max(
            chunk_label_scores.get(lbl, 0.0)
            for lbl in TIER2_LABELS[:TIER2_SCAM_COUNT]
        )

        # Keep the chunk that produced the highest scam confidence
        if chunk_top_scam > best_top_scam:
            best_top_scam = chunk_top_scam
            best_scores   = chunk_label_scores

    if not best_scores:
        return {
            "scores": {}, "top_scam_score": 0.0,
            "top_label": "none", "tier1_triggered": tier1_triggered,
        }

    top_label = max(best_scores, key=best_scores.get)

    return {
        "scores":               best_scores,
        "top_scam_score":       best_top_scam,
        "top_label":            top_label,
        "tier1_triggered":      tier1_triggered,
        "tier1_scam_confidence": round(tier1_scam_conf, 3),
    }


# ─── Ollama backend (replacement mode) ───────────────────────────────────────
# This is the NLP_BACKEND="ollama" path — it replaces BART entirely.
# For the additive LLM reasoning layer (euphemism detection), see llm_reasoning.py.

def classify_with_ollama(text: str) -> dict:
    """
    Ask a locally running Ollama model to evaluate text for scam patterns.
    Returns the same interface shape as classify_with_transformers.
    """
    prompt = f"""Analyze the following text extracted from a social media video for scam or fraud patterns.

TEXT:
\"\"\"
{text[:2000]}
\"\"\"

Respond ONLY with a JSON object (no markdown, no explanation):
{{
  "is_scam": true or false,
  "confidence": 0.0 to 1.0,
  "categories": ["gambling", "crypto_scam", "investment_fraud", "ponzi_pyramid",
                 "fake_giveaway", "referral_fraud", "urgency_pressure"],
  "reasons": ["reason 1", "reason 2"],
  "top_label": "short label"
}}

Only include categories actually present. If no scam found, return empty list."""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT_SEC,
        )
        response.raise_for_status()
        raw  = response.json().get("response", "{}")
        data = json.loads(raw)

        confidence = float(data.get("confidence", 0.0))
        categories = data.get("categories", [])

        scores = {lbl: 0.1 for lbl in TIER2_LABELS}
        for cat in categories:
            for lbl in TIER2_LABELS:
                if cat.replace("_", " ") in lbl:
                    scores[lbl] = confidence

        return {
            "scores":          scores,
            "top_scam_score":  confidence if data.get("is_scam") else 0.0,
            "top_label":       data.get("top_label", "unknown"),
            "tier1_triggered": data.get("is_scam", False),
            "ollama_reasons":  data.get("reasons", []),
        }
    except Exception as e:
        return {
            "scores": {}, "top_scam_score": 0.0,
            "top_label": "error", "tier1_triggered": False,
            "error": str(e),
        }


# ─── Public interface ─────────────────────────────────────────────────────────

def analyze(text: str, classifier=None) -> dict:
    """
    Full NLP analysis: keyword engine + zero-shot classification.
    Always runs keyword analysis.
    Zero-shot uses BART (transformers) or Ollama depending on NLP_BACKEND.
    """
    kw_result = keyword_analysis(text)

    if NLP_BACKEND == "ollama":
        nlp_result = classify_with_ollama(text)
    else:
        nlp_result = classify_with_transformers(text, classifier)

    extra_reasons = nlp_result.pop("ollama_reasons", [])

    return {
        "keyword":       kw_result,
        "nlp":           nlp_result,
        "extra_reasons": extra_reasons,
    }
