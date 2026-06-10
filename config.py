"""
Central configuration for AI Media Watch.
Tune these settings to balance speed vs. accuracy.
"""

# ─── Whisper (speech-to-text) ─────────────────────────────────────────────────
WHISPER_MODEL = "base"          # tiny | base | small | medium | large
WHISPER_DEVICE = "cpu"          # cpu | cuda
WHISPER_COMPUTE_TYPE = "int8"   # int8 (fast) | float16 | float32

# ─── EasyOCR ──────────────────────────────────────────────────────────────────
OCR_LANGUAGES = ["en", "ar"]    # add "fr", "es", "zh" etc. as needed
OCR_GPU = False

# ─── NLP backend ──────────────────────────────────────────────────────────────
# "transformers" uses a local zero-shot model (downloads ~1.5 GB once)
# "ollama"       uses a locally running Ollama server (must be running)
NLP_BACKEND = "transformers"
NLP_MODEL = "facebook/bart-large-mnli"
NLP_MODEL_MULTILINGUAL = "joeddav/xlm-roberta-large-xnli"

OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# ─── Video processing ─────────────────────────────────────────────────────────
FRAME_EXTRACTION_FPS = 0.5   # 1 frame every 2 seconds
MAX_FRAMES = 40               # cap to keep inference fast

# ─── YOLOv8 (visual object detection) ────────────────────────────────────────
# "yolov8n.pt" is the nano model — ~6 MB, fast on CPU.
# Replace with a path to custom weights once you train them.
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE = 0.35          # detections below this confidence are ignored
YOLO_MAX_ANNOTATED_FRAMES = 12  # how many annotated frames to keep for the UI

# ─── CLIP (zero-shot image classification) ────────────────────────────────────
# CLIP lets us classify frames against text prompts with no custom training.
# Downloads ~340 MB on first run, cached afterwards.
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_CONFIDENCE_THRESHOLD = 0.20  # minimum softmax score to report a CLIP hit
ENABLE_CLIP_ANALYSIS = True        # set False to skip CLIP and save ~340 MB

# ─── Risk scoring weights ─────────────────────────────────────────────────────
# Must sum to 1.0.
# Visual gets 15 %; the other three are scaled down proportionally from before.
KEYWORD_WEIGHT = 0.30
NLP_WEIGHT     = 0.38
PATTERN_WEIGHT = 0.17
VISUAL_WEIGHT  = 0.15

# ─── Risk thresholds ──────────────────────────────────────────────────────────
RISK_LEVELS = {
    (0,  25):  ("Low",      "green",   "✅"),
    (26, 50):  ("Moderate", "orange",  "⚠️"),
    (51, 75):  ("High",     "red",     "🚨"),
    (76, 100): ("Critical", "darkred", "🛑"),
}
