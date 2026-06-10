# 🔍 AI Media Watch — Scam Detection in Social Media Videos

A hackathon-ready Streamlit app that analyzes social media videos end-to-end and scores them for scam / fraud risk (0–100).

---

## Project Structure

```
AI watch/
├── app.py               # Streamlit dashboard (entry point)
├── video_processing.py  # FFmpeg audio extraction + OpenCV frame sampling
├── speech_to_text.py    # Faster-Whisper transcription
├── ocr.py               # EasyOCR text extraction from frames
├── nlp_analysis.py      # Zero-shot NLP + keyword scam analysis
├── risk_engine.py       # Weighted risk score aggregation (0–100)
├── config.py            # All tunable parameters in one place
├── requirements.txt
└── .streamlit/
    └── config.toml      # Dark theme + upload size config
```

---

## Prerequisites

### 1. FFmpeg (system binary)

FFmpeg is used to extract audio from video files. Install it before running.

**Windows**
```powershell
winget install --id=Gyan.FFmpeg -e
# or download from https://ffmpeg.org/download.html and add to PATH
```

**macOS**
```bash
brew install ffmpeg
```

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install ffmpeg -y
```

Verify: `ffmpeg -version`

### 2. Python 3.10+

Check: `python --version`

---

## Installation

```bash
# 1. Clone / navigate to the project
cd "AI watch"

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **First run note:** Three things download automatically on first launch:
> - Whisper `base` model (~150 MB)
> - EasyOCR English+Arabic models (~500 MB combined)
> - `facebook/bart-large-mnli` zero-shot model (~1.6 GB)
>
> Subsequent runs use cached versions.

### GPU acceleration (optional)

If you have an NVIDIA GPU with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Then set in `config.py`:
```python
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
OCR_GPU = True
```

### Ollama backend (optional alternative to transformers)

Install [Ollama](https://ollama.com) and pull a model:
```bash
ollama pull llama3.2
```
Then in `config.py`:
```python
NLP_BACKEND = "ollama"
```

---

## Running the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## How to Use

1. Open the app in your browser
2. Drag-and-drop (or click to upload) a video — MP4, MOV, AVI, MKV, WebM, FLV
3. Click **"🔍 Analyze Video"**
4. Watch the pipeline run step-by-step in the status panel
5. Review results:

---

## Example Output

```
Video: suspicious_ad.mp4 | 45.3s | 1280×720 @ 30fps | Frames sampled: 22

Risk Score: 87 / 100
Risk Level: 🛑 CRITICAL

Detected Patterns:
  Cryptocurrency Scam          92%  ████████████████████
  Investment Fraud             78%  ████████████████
  High-Pressure Urgency        61%  ████████████

Score Breakdown:
  Keyword Match     84/100
  NLP Classifier    91/100
  Pattern Analysis  72/100

⚠️ Flagged Keywords:
  "send btc"  "guaranteed profit"  "100x"  "free crypto"
  "act now"   "double your money"  "airdrop"

📋 Detection Reasons:
  - Cryptocurrency Scam keywords: "send btc", "guaranteed profit", "100x"
  - NLP model flagged: "cryptocurrency scam or fraudulent crypto investment" (91%)
  - Multiple monetary amounts mentioned: ['$500', '$50000']
  - Suspicious link detected (Telegram / BitLy)
  - Multiple scam categories triggered simultaneously: crypto_scam, investment_fraud, urgency_pressure

📝 Transcript (EN):
  "Send 0.5 BTC to this wallet and receive 1 BTC back guaranteed.
   This is a limited time offer — act now before it expires..."

📷 OCR from Frames:
  SEND 0.5 BTC | GET 1 BTC BACK GUARANTEED | t.me/cryptoprofit
```

---

## Configuration Reference (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WHISPER_MODEL` | `base` | Whisper model size: `tiny/base/small/medium/large` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `NLP_BACKEND` | `transformers` | `transformers` or `ollama` |
| `NLP_MODEL` | `facebook/bart-large-mnli` | Zero-shot model (HuggingFace hub ID) |
| `OCR_LANGUAGES` | `["en", "ar"]` | EasyOCR language codes |
| `FRAME_EXTRACTION_FPS` | `0.5` | Frames per second to sample |
| `MAX_FRAMES` | `40` | Max frames processed per video |
| `KEYWORD_WEIGHT` | `0.35` | Weight of keyword score in final risk |
| `NLP_WEIGHT` | `0.45` | Weight of NLP score in final risk |
| `PATTERN_WEIGHT` | `0.20` | Weight of structural patterns in final risk |

---

## Scam Categories Detected

| Category | Examples |
|----------|---------|
| 🎰 Gambling Ads | "casino", "jackpot", "sports betting", "bet now" |
| ₿ Crypto Scams | "send BTC", "100x", "free crypto", "airdrop", "double your Bitcoin" |
| 📈 Investment Fraud | "guaranteed returns", "risk-free", "100% profit", "forex signals" |
| 🔺 Pyramid / MLM | "referral bonus", "downline", "recruit friends", "Ponzi" |
| 🎁 Fake Giveaways | "you won", "claim your prize", "Elon Musk giveaway", "advance fee" |
| ⚡ Urgency Tactics | "act now", "limited time", "expires today", "last chance" |

---

## Troubleshooting

**`ffmpeg: command not found`**
→ Install FFmpeg and ensure it's in your PATH.

**CUDA out of memory**
→ Set `WHISPER_DEVICE = "cpu"` and `OCR_GPU = False` in `config.py`.

**Slow first run**
→ Models download once to `~/.cache/huggingface`. Subsequent runs are fast.

**EasyOCR taking too long**
→ Reduce `MAX_FRAMES` in `config.py` (e.g. `10`).

**Ollama error / timeout**
→ Make sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3.2`).

---

## Pipeline Diagram

```
Video File
    │
    ├─► FFmpeg ──────────► Audio (WAV 16kHz)
    │                           │
    │                           ▼
    │                    Faster-Whisper
    │                           │
    │                    Transcript Text
    │                           │
    ├─► OpenCV ──────────► Sampled Frames
    │                           │
    │                           ▼
    │                       EasyOCR
    │                           │
    │                      OCR Text
    │                           │
    └───────────────────► Combined Text
                                │
                    ┌───────────┴───────────┐
                    │                       │
             Keyword Analysis      Zero-Shot NLP
             (deterministic)    (facebook/bart-large-mnli)
                    │                       │
                    └───────────┬───────────┘
                                │
                         Pattern Analysis
                         (URLs, CAPS, etc.)
                                │
                          Risk Engine
                         (weighted sum)
                                │
                    ┌───────────▼───────────┐
                    │  Risk Score 0–100     │
                    │  Reasons              │
                    │  Keywords             │
                    │  Category Breakdown   │
                    └───────────────────────┘
                                │
                       Streamlit Dashboard
```
