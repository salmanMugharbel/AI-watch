"""
nlp_analysis.py
Two-layer scam detection:
  1. Keyword matching – fast, deterministic, language-agnostic rules
  2. Zero-shot NLP classification – semantic understanding via a pretrained LLM
     (or Ollama if configured)
"""

from __future__ import annotations
import re
import json
import requests
from config import NLP_BACKEND, NLP_MODEL, OLLAMA_MODEL, OLLAMA_BASE_URL


# ─── Scam keyword dictionary ──────────────────────────────────────────────────
# Each category maps to a list of (keyword, weight) tuples.
# Weight > 1 marks a high-signal phrase that strongly indicates a scam.

SCAM_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "gambling": [
        ("casino", 1.0), ("online casino", 1.5), ("betting", 1.0),
        ("jackpot", 1.0), ("slot machine", 1.2), ("poker", 0.8),
        ("roulette", 1.0), ("sports betting", 1.2), ("bet now", 1.5),
        ("win big", 1.3), ("free spins", 1.2), ("gambling", 1.2),
        ("كازينو", 1.5), ("مراهنات", 1.5), ("يانصيب", 1.2),
    ],
    "crypto_scam": [
        ("send btc", 2.0), ("send bitcoin", 2.0), ("send eth", 2.0),
        ("100x", 1.5), ("1000x", 2.0), ("to the moon", 1.3),
        ("guaranteed profit", 2.0), ("free crypto", 1.8),
        ("airdrop", 1.3), ("crypto signals", 1.5), ("pump", 1.0),
        ("nft drop", 1.2), ("defi yield", 1.2), ("invest in crypto", 1.3),
        ("double your bitcoin", 2.5), ("crypto giveaway", 2.0),
        ("blockchain investment", 1.3), ("mining profit", 1.2),
    ],
    "investment_fraud": [
        ("guaranteed returns", 2.5), ("risk-free", 2.0), ("risk free", 2.0),
        ("double your money", 2.5), ("triple your money", 2.5),
        ("passive income", 1.2), ("financial freedom", 1.3),
        ("forex signals", 1.5), ("trading bot", 1.3), ("100% profit", 2.5),
        ("no risk", 2.0), ("profit guaranteed", 2.5), ("high returns", 1.5),
        ("exclusive opportunity", 1.5), ("insider tip", 1.8),
        ("invest now", 1.2), ("limited time offer", 1.3),
        ("مضاعفة أموالك", 2.5), ("ربح مضمون", 2.5), ("استثمار مضمون", 2.0),
    ],
    "ponzi_pyramid": [
        ("referral bonus", 1.8), ("recruit friends", 1.8), ("downline", 2.0),
        ("upline", 2.0), ("mlm", 1.8), ("network marketing", 1.5),
        ("multi-level", 1.8), ("join our team", 1.2), ("earn from referrals", 2.0),
        ("pyramid scheme", 3.0), ("ponzi", 3.0), ("chain letter", 2.5),
        ("make money from home", 1.5), ("work from home", 0.8),
        ("unlimited earnings", 1.8), ("passive referral income", 2.0),
    ],
    "fake_giveaway": [
        ("giveaway", 1.2), ("free money", 2.0), ("you won", 2.0),
        ("congratulations you have been selected", 2.5),
        ("claim your prize", 2.0), ("processing fee", 2.5),
        ("advance fee", 3.0), ("send fee", 2.5), ("free gift", 1.5),
        ("double giveaway", 2.5), ("elon musk giveaway", 3.0),
        ("celebrity giveaway", 2.5), ("tesla giveaway", 2.5),
        ("send 0.5 btc get 1 btc", 3.0),
    ],
    "urgency_pressure": [
        ("act now", 1.5), ("limited time", 1.2), ("expires today", 1.8),
        ("last chance", 1.5), ("hurry up", 1.3), ("only today", 1.8),
        ("join now before", 1.5), ("register today", 1.2), ("deadline", 1.0),
        ("spots are filling", 1.5), ("don't miss out", 1.2),
    ],
}

# Labels passed to the zero-shot classifier
ZS_LABELS = [
    "gambling advertisement or betting promotion",
    "cryptocurrency scam or fraudulent crypto investment",
    "investment fraud with guaranteed or unrealistic returns",
    "pyramid scheme or multi-level marketing fraud",
    "fake giveaway or prize scam",
    "get rich quick scheme",
    "financial fraud or money scam",
    "legitimate educational or news content",
    "legitimate product advertisement",
]

# Labels considered "scam" (all except the last two)
SCAM_LABEL_INDICES = list(range(len(ZS_LABELS) - 2))


# ─── Keyword analysis ─────────────────────────────────────────────────────────

def keyword_analysis(text: str) -> dict:
    """
    Scan combined text for scam keywords.
    Returns per-category hit counts, weighted scores, and matched phrases.
    """
    text_lower = text.lower()
    results: dict[str, dict] = {}

    for category, keywords in SCAM_KEYWORDS.items():
        hits: list[str] = []
        score = 0.0

        for kw, weight in keywords:
            # Word-boundary aware search
            pattern = r"\b" + re.escape(kw) + r"\b"
            matches = re.findall(pattern, text_lower)
            if matches:
                hits.extend(matches)
                score += weight * len(matches)

        results[category] = {
            "hits": list(set(hits)),
            "raw_score": score,
            # Normalise to 0–1 using a soft cap at score=5
            "normalized_score": min(score / 5.0, 1.0),
        }

    # Aggregate keyword score across all categories
    all_scores = [v["normalized_score"] for v in results.values()]
    aggregate = min(sum(all_scores) / len(all_scores) * 2.5, 1.0)  # boost & cap

    all_hits = []
    for v in results.values():
        all_hits.extend(v["hits"])

    return {
        "per_category": results,
        "aggregate_score": aggregate,
        "all_matched_keywords": list(set(all_hits)),
    }


# ─── Transformers zero-shot classification ────────────────────────────────────

def load_zs_classifier():
    from transformers import pipeline
    clf = pipeline(
        "zero-shot-classification",
        model=NLP_MODEL,
        device=-1,  # -1 = CPU; set to 0 for GPU
    )
    return clf


def classify_with_transformers(text: str, classifier=None) -> dict:
    """
    Run multi-label zero-shot classification.
    Scores all ZS_LABELS and returns per-label confidence scores.
    """
    if not text.strip():
        return {"scores": {}, "top_scam_score": 0.0, "top_label": "none"}

    if classifier is None:
        classifier = load_zs_classifier()

    # Truncate to 1000 chars – the model has a token limit
    truncated = text[:1000]

    result = classifier(truncated, ZS_LABELS, multi_label=True)
    label_scores = dict(zip(result["labels"], result["scores"]))

    # Highest scam label score
    scam_scores = [
        result["scores"][i] for i in SCAM_LABEL_INDICES if i < len(result["scores"])
    ]
    top_scam_score = max(scam_scores) if scam_scores else 0.0
    top_label = result["labels"][result["scores"].index(max(result["scores"]))]

    return {
        "scores": label_scores,
        "top_scam_score": top_scam_score,
        "top_label": top_label,
    }


# ─── Ollama LLM classification (optional) ────────────────────────────────────

def classify_with_ollama(text: str) -> dict:
    """
    Ask a locally running Ollama model to evaluate the text for scams.
    Returns the same shape as classify_with_transformers.
    """
    prompt = f"""Analyze the following text extracted from a social media video for scam or fraud patterns.

TEXT:
\"\"\"
{text[:1500]}
\"\"\"

Respond ONLY with a JSON object like this (no markdown, no explanation):
{{
  "is_scam": true or false,
  "confidence": 0.0 to 1.0,
  "categories": ["gambling", "crypto_scam", "investment_fraud", "ponzi_pyramid", "fake_giveaway"],
  "reasons": ["reason 1", "reason 2"],
  "top_label": "short label"
}}

Categories detected should only include ones actually present. If no scam, return empty list."""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json().get("response", "{}")
        data = json.loads(raw)

        confidence = float(data.get("confidence", 0.0))
        categories = data.get("categories", [])

        # Build scores dict compatible with transformer output
        scores = {label: 0.1 for label in ZS_LABELS}
        for cat in categories:
            for label in ZS_LABELS:
                if cat.replace("_", " ") in label:
                    scores[label] = confidence

        return {
            "scores": scores,
            "top_scam_score": confidence if data.get("is_scam") else 0.0,
            "top_label": data.get("top_label", "unknown"),
            "ollama_reasons": data.get("reasons", []),
        }
    except Exception as e:
        # Ollama unavailable – return neutral result
        return {"scores": {}, "top_scam_score": 0.0, "top_label": "error", "error": str(e)}


# ─── Public interface ─────────────────────────────────────────────────────────

def analyze(text: str, classifier=None) -> dict:
    """
    Full NLP analysis pipeline.
    Always runs keyword analysis; additionally runs zero-shot or Ollama
    depending on NLP_BACKEND config.

    Returns a unified analysis dict consumed by risk_engine.py.
    """
    kw_result = keyword_analysis(text)

    if NLP_BACKEND == "ollama":
        nlp_result = classify_with_ollama(text)
    else:
        nlp_result = classify_with_transformers(text, classifier)

    # Merge Ollama reasons (if any) into the output
    extra_reasons = nlp_result.pop("ollama_reasons", [])

    return {
        "keyword": kw_result,
        "nlp": nlp_result,
        "extra_reasons": extra_reasons,
    }
