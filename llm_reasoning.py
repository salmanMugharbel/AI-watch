"""
llm_reasoning.py
Additive LLM reasoning layer using Ollama.

This module is NOT a replacement for the BART classifier.
Its purpose is to detect what keyword engines and zero-shot models miss:

  • Euphemisms        "join our community", "the system", "this method"
  • Coded language    "passive streams", "leveraging", "compound growth"
  • Implied promises  "this changed my life", "quit your job"
  • Social proof      "thousands already joined", "my team of 500"
  • Soft urgency      "seats are filling up", "before the algorithm changes"

It runs AFTER the initial risk score is computed.  It only makes an Ollama
API call when:
  1. ENABLE_LLM_REASONING = True in config.py, AND
  2. The preliminary risk score is within [LLM_TRIGGER_MIN_SCORE, LLM_TRIGGER_MAX_SCORE]
     (borderline scores benefit most — very-low and very-high scores don't need it)

If Ollama is not running, this module returns a safe stub silently.
It never raises; errors are captured in the returned dict.
"""

from __future__ import annotations
import json
import requests
from config import (
    ENABLE_LLM_REASONING,
    LLM_TRIGGER_MIN_SCORE, LLM_TRIGGER_MAX_SCORE,
    OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SEC,
)


# ─── Prompt ───────────────────────────────────────────────────────────────────

_REASONING_PROMPT = """\
You are a financial fraud analyst specialising in detecting disguised and indirect scam language.

Standard keyword systems already handle obvious terms like "guaranteed profit" or "send BTC".
Your task is to find what those systems MISS.

Look for ONLY these five patterns:

1. EUPHEMISMS — indirect phrases that replace direct scam terms
   Examples: "join our community" (pyramid recruitment), "the opportunity" (scheme),
   "the system" (fraud method), "this platform" (unlicensed service)

2. CODED FINANCIAL LANGUAGE — jargon misused to imply illegal promises
   Examples: "passive streams" (illegal scheme), "leveraging capital" (Ponzi),
   "compounding daily" (fake return), "scaling your portfolio" (scam investment)

3. IMPLIED PROMISES — wealth suggestions without an explicit guarantee
   Examples: "imagine waking up to notifications" (passive income fraud),
   "this completely changed my life" (false testimonial),
   "I was able to quit my 9-to-5" (unrealistic income claim)

4. SOCIAL PROOF MANIPULATION — fake community size or success claims
   Examples: "over 50,000 members already", "my team of 800 people",
   "members from 60 countries", "joined by doctors and engineers"

5. SOFT URGENCY — pressure without explicit deadline language
   Examples: "before the algorithm changes", "while seats last",
   "early adopters are getting the most", "first movers always win"

TEXT TO ANALYSE:
\"\"\"
{text}
\"\"\"

Respond ONLY with valid JSON. No markdown. No explanation outside the JSON:
{{
  "euphemism_detected":         true or false,
  "coded_language_detected":    true or false,
  "implied_promises_detected":  true or false,
  "social_proof_manipulation":  true or false,
  "soft_urgency_detected":      true or false,
  "confidence":                 0.0 to 1.0,
  "flagged_phrases":            ["exact short phrase from the text", ...],
  "explanation":                "one sentence summary of findings",
  "scam_type_hint":             "e.g. MLM recruitment / investment fraud / none"
}}"""


# ─── Empty stub ───────────────────────────────────────────────────────────────

def _empty_result(reason: str = "") -> dict:
    return {
        "euphemism_detected":        False,
        "coded_language_detected":   False,
        "implied_promises_detected": False,
        "social_proof_manipulation": False,
        "soft_urgency_detected":     False,
        "confidence":                0.0,
        "flagged_phrases":           [],
        "explanation":               "",
        "scam_type_hint":            "none",
        "ran":                       False,
        "skip_reason":               reason,
    }


# ─── Public entry point ───────────────────────────────────────────────────────

def reason(text: str, preliminary_score: int) -> dict:
    """
    Run the additive LLM reasoning pass.

    Args:
        text              : combined transcript + OCR text
        preliminary_score : risk score computed WITHOUT LLM boost (0–100)

    Returns a reasoning dict.  Always returns a valid dict — never raises.
    The caller (risk_engine.py) reads the boolean flags and confidence to
    decide how many points to add to the preliminary score.
    """
    # ── Guard 1: feature disabled in config ───────────────────────────────────
    if not ENABLE_LLM_REASONING:
        return _empty_result("LLM reasoning disabled in config")

    # ── Guard 2: score outside the useful trigger window ─────────────────────
    if preliminary_score < LLM_TRIGGER_MIN_SCORE:
        return _empty_result(f"score {preliminary_score} below trigger minimum {LLM_TRIGGER_MIN_SCORE}")
    if preliminary_score > LLM_TRIGGER_MAX_SCORE:
        return _empty_result(f"score {preliminary_score} above trigger maximum {LLM_TRIGGER_MAX_SCORE}")

    # ── Guard 3: no text to analyse ───────────────────────────────────────────
    if not text.strip():
        return _empty_result("no text available for LLM analysis")

    # ── Build prompt ──────────────────────────────────────────────────────────
    # Use first 2500 chars — enough to capture the core pitch without overwhelming
    prompt = _REASONING_PROMPT.format(text=text[:2500])

    # ── Call Ollama ───────────────────────────────────────────────────────────
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},  # low temperature for consistent JSON
            },
            timeout=OLLAMA_TIMEOUT_SEC,
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()

        # Strip markdown code fences if the model added them
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        data = json.loads(raw_text)

    except requests.exceptions.ConnectionError:
        return _empty_result("Ollama not reachable — is `ollama serve` running?")
    except requests.exceptions.Timeout:
        return _empty_result("Ollama request timed out")
    except json.JSONDecodeError:
        return _empty_result("Ollama returned invalid JSON")
    except Exception as e:
        return _empty_result(f"unexpected error: {e}")

    # ── Validate and return ───────────────────────────────────────────────────
    result = {
        "euphemism_detected":        bool(data.get("euphemism_detected",        False)),
        "coded_language_detected":   bool(data.get("coded_language_detected",   False)),
        "implied_promises_detected": bool(data.get("implied_promises_detected", False)),
        "social_proof_manipulation": bool(data.get("social_proof_manipulation", False)),
        "soft_urgency_detected":     bool(data.get("soft_urgency_detected",     False)),
        "confidence":                float(min(max(data.get("confidence", 0.0), 0.0), 1.0)),
        "flagged_phrases":           data.get("flagged_phrases", [])[:10],   # cap at 10
        "explanation":               str(data.get("explanation", ""))[:400],
        "scam_type_hint":            str(data.get("scam_type_hint", "none")),
        "ran":                       True,
        "skip_reason":               "",
    }

    return result
