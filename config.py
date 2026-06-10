"""
config.py — Central configuration for AI Media Watch.
All tunable parameters in one place. No magic numbers in module code.
"""

# ─── Whisper (speech-to-text) ──────────────────────────────────────────────────
WHISPER_MODEL        = "base"   # tiny | base | small | medium | large
WHISPER_DEVICE       = "cpu"    # cpu  | cuda
WHISPER_COMPUTE_TYPE = "int8"   # int8 (fastest CPU) | float16 | float32

# ─── OCR (EasyOCR) ────────────────────────────────────────────────────────────
OCR_LANGUAGES          = ["en", "ar"]   # add "fr", "es", "zh" etc. as needed
OCR_GPU                = False
OCR_PREPROCESSING      = True   # apply CLAHE + sharpening before OCR
# Dual confidence thresholds — short strings (URLs, phone numbers, handles)
# are often rendered in small / stylised fonts and score below the general
# threshold even when correct.  We use a lower bar so we never drop "t.me/"
# or "@username" purely because the font is unusual.
OCR_SHORT_STRING_CONF  = 0.25   # strings ≤ OCR_SHORT_STRING_MAX_LEN chars
OCR_LONG_STRING_CONF   = 0.40   # strings  > OCR_SHORT_STRING_MAX_LEN chars
OCR_SHORT_STRING_MAX_LEN = 20   # chars — below this, use the lower threshold

# ─── NLP (BART-MNLI zero-shot) ────────────────────────────────────────────────
NLP_BACKEND              = "transformers"  # "transformers" | "ollama"
NLP_MODEL                = "facebook/bart-large-mnli"
NLP_MODEL_MULTILINGUAL   = "joeddav/xlm-roberta-large-xnli"

# Two-pass classification settings
NLP_TIER1_TRIGGER_CONF   = 0.40   # Tier-1 confidence above this → boost Tier-2 scores
NLP_TIER2_BOOST_FACTOR   = 1.15   # multiplier on Tier-2 scam scores when Tier-1 fired

# Text chunking — ensures the NLP model sees the whole transcript, not just
# the first 1000 chars.  Each chunk is classified independently; the chunk
# with the highest scam score drives the final result.
NLP_CHUNK_SIZE    = 800    # characters per chunk
NLP_CHUNK_OVERLAP = 150    # overlap between consecutive chunks
NLP_MAX_CHUNKS    = 4      # cap to keep inference time reasonable on CPU

# ─── LLM reasoning layer (Ollama — additive, not a replacement) ───────────────
ENABLE_LLM_REASONING  = True    # set False to skip (Ollama not required)
LLM_TRIGGER_MIN_SCORE = 25      # preliminary score must be >= this to trigger
LLM_TRIGGER_MAX_SCORE = 80      # preliminary score must be <= this to trigger
                                  # (very-high scores don't need extra evidence)
OLLAMA_MODEL          = "llama3.2"
OLLAMA_BASE_URL       = "http://localhost:11434"
OLLAMA_TIMEOUT_SEC    = 45

# ─── YOLOv8 (visual object detection) ────────────────────────────────────────
YOLO_MODEL_PATH         = "yolov8n.pt"   # nano model ~6 MB, fast on CPU
YOLO_CONFIDENCE         = 0.35
YOLO_MAX_ANNOTATED_FRAMES = 12
# A YOLO detection only contributes to the score when the SAME frame also
# has a CLIP cosine similarity score above this gate threshold for at least
# one scam-oriented prompt.  Eliminates false positives from cars in travel
# vlogs and laptops in tech tutorials.
YOLO_CLIP_GATE_THRESHOLD = 0.18  # CLIP cosine score required to credit YOLO

# ─── CLIP (zero-shot image classification) ────────────────────────────────────
CLIP_MODEL              = "openai/clip-vit-base-patch32"
CLIP_COSINE_THRESHOLD   = 0.22   # independent per-prompt cosine similarity
ENABLE_CLIP_ANALYSIS    = True

# ─── Adaptive frame sampling ──────────────────────────────────────────────────
ADAPTIVE_FRAME_SAMPLING = True   # if False, falls back to FRAME_EXTRACTION_FPS
FRAME_EXTRACTION_FPS    = 0.5    # used only when ADAPTIVE_FRAME_SAMPLING=False
FRAME_FPS_SHORT         = 2.0   # video duration < 30 s
FRAME_FPS_MEDIUM        = 1.0   # 30 s – 120 s
FRAME_FPS_LONG          = 0.5   # 120 s – 600 s
FRAME_FPS_XLONG         = 0.25  # > 600 s
MAX_FRAMES              = 48    # hard cap (up from 40)

# ─── Risk engine weights (must sum to 1.0) ────────────────────────────────────
KEYWORD_WEIGHT = 0.25
NLP_WEIGHT     = 0.35
PATTERN_WEIGHT = 0.15
VISUAL_WEIGHT  = 0.25   # increased — CLIP cosine is more reliable than before

# ─── Risk engine floors ───────────────────────────────────────────────────────
# If a single modality is near-certain, the final score cannot fall below
# these floors regardless of what other (absent) modalities scored.
NLP_FLOOR_85 = 60   # NLP confidence > 85 % → score ≥ 60
NLP_FLOOR_92 = 78   # NLP confidence > 92 % → score ≥ 78
KW_FLOOR_80  = 55   # keyword score   > 80 % → score ≥ 55
VIS_FLOOR_85 = 65   # visual score    > 85 % → score ≥ 65

# ─── Risk engine convergence boosts ──────────────────────────────────────────
# When multiple independent modalities agree, multiply the raw score.
CONVERGENCE_STRONG_THRESHOLD = 40   # component must score > 40/100 to count
CONVERGENCE_BOOST_2          = 1.10  # 2 strong signals → ×1.10
CONVERGENCE_BOOST_3          = 1.20  # 3 strong signals → ×1.20
CONVERGENCE_BOOST_4          = 1.30  # 4 strong signals → ×1.30

# ─── LLM boost values (applied after initial score is known) ─────────────────
LLM_EUPHEMISM_BOOST    = 15   # raw points added when euphemisms detected
LLM_CODED_LANG_BOOST   = 10   # raw points added when coded language detected
LLM_IMPLIED_BOOST      =  8   # raw points added when implied promises detected
LLM_SOCIAL_PROOF_BOOST =  5   # raw points added when social-proof manipulation seen

# ─── Risk level thresholds ───────────────────────────────────────────────────
RISK_LEVELS = {
    (0,  25):  ("Low",      "green",   "✅"),
    (26, 50):  ("Moderate", "orange",  "⚠️"),
    (51, 75):  ("High",     "red",     "🚨"),
    (76, 100): ("Critical", "darkred", "🛑"),
}
