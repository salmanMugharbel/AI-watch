"""
app.py – AI Media Watch: Scam Detection in Social Media Videos
Streamlit dashboard that ties the full pipeline together.

Pipeline order:
  1. Video → audio + frames
  2. Faster-Whisper  → transcript
  3. EasyOCR         → frame text
  4. YOLOv8 + pyzbar + CLIP → visual detections   ← NEW
  5. NLP zero-shot   → semantic classification
  6. Risk Engine     → 0–100 score (4 components)
  7. Dashboard       → results + visual section    ← UPDATED
"""

import os
import shutil
import tempfile
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Media Watch",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }

    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #2d3250;
        text-align: center;
    }
    .risk-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 6px;
    }
    .kw-chip {
        display: inline-block;
        background: #2d1f1f;
        color: #ff6b6b;
        border: 1px solid #ff6b6b55;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 12px;
        margin: 3px;
    }
    .vis-chip {
        display: inline-block;
        background: #1f2a3d;
        color: #60a5fa;
        border: 1px solid #60a5fa55;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 12px;
        margin: 3px;
    }
    .qr-chip {
        display: inline-block;
        background: #2a1f1f;
        color: #f87171;
        border: 1px solid #f8717155;
        padding: 4px 12px;
        border-radius: 14px;
        font-size: 12px;
        margin: 3px;
        font-family: monospace;
    }
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        border-bottom: 1px solid #2d3250;
        padding-bottom: 6px;
    }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Cached model loaders ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading speech model…")
def get_whisper_model():
    from speech_to_text import load_whisper_model
    return load_whisper_model()


@st.cache_resource(show_spinner="Loading OCR engine…")
def get_ocr_reader():
    from ocr import load_ocr_reader
    return load_ocr_reader()


@st.cache_resource(show_spinner="Loading NLP classifier…")
def get_nlp_classifier():
    from config import NLP_BACKEND
    if NLP_BACKEND == "transformers":
        from nlp_analysis import load_zs_classifier
        return load_zs_classifier()
    return None


@st.cache_resource(show_spinner="Loading YOLO model…")
def get_yolo_model():
    """YOLOv8n — downloads ~6 MB on first run."""
    from vision_analysis import load_yolo_model
    return load_yolo_model()


@st.cache_resource(show_spinner="Loading CLIP vision model…")
def get_clip_models():
    """CLIP vit-base-patch32 — downloads ~340 MB on first run."""
    from config import ENABLE_CLIP_ANALYSIS
    if not ENABLE_CLIP_ANALYSIS:
        return None
    from vision_analysis import load_clip_model
    return load_clip_model()   # returns (processor, model) tuple


# ─── Gauge chart ──────────────────────────────────────────────────────────────

def make_gauge(score: int, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"size": 48, "color": "white"}, "suffix": ""},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#444",
                "tickfont": {"color": "#888", "size": 11},
            },
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1e2130",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  25], "color": "#1a3a1a"},
                {"range": [25, 50], "color": "#3a3010"},
                {"range": [50, 75], "color": "#3a1a10"},
                {"range": [75, 100], "color": "#3a0a0a"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin={"t": 30, "b": 10, "l": 20, "r": 20},
        height=220,
    )
    return fig


# ─── Visual contribution mini-bar chart ───────────────────────────────────────

def make_component_bar(component_scores: dict) -> go.Figure:
    """Horizontal bar chart comparing the four scoring components."""
    labels = ["Keyword Match", "NLP Classifier", "Pattern Analysis", "Visual (YOLO/CLIP)"]
    keys   = ["keyword", "nlp", "pattern", "visual"]
    values = [component_scores.get(k, 0) for k in keys]
    colors = ["#818cf8", "#34d399", "#fbbf24", "#60a5fa"]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v}/100" for v in values],
        textposition="outside",
        textfont={"color": "white", "size": 11},
    ))
    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1e2130",
        margin={"t": 10, "b": 10, "l": 10, "r": 60},
        height=160,
        xaxis={"range": [0, 115], "showgrid": False, "color": "#444", "tickfont": {"color": "#888"}},
        yaxis={"showgrid": False, "color": "white",  "tickfont": {"color": "#e2e8f0", "size": 12}},
        showlegend=False,
    )
    return fig


# ─── Helper renderers ─────────────────────────────────────────────────────────

def render_keyword_chips(keywords: list[str]) -> None:
    if not keywords:
        st.caption("No scam keywords matched.")
        return
    chips = " ".join(f'<span class="kw-chip">{kw}</span>' for kw in sorted(keywords))
    st.markdown(chips, unsafe_allow_html=True)


def render_visual_chips(detections: list[dict]) -> None:
    if not detections:
        st.caption("No visual scam indicators detected.")
        return
    seen: set[str] = set()
    chips = []
    for d in detections:
        lbl = d["label"]
        if lbl not in seen:
            seen.add(lbl)
            chips.append(f'<span class="vis-chip">{lbl}</span>')
    st.markdown(" ".join(chips), unsafe_allow_html=True)


def render_risk_badge(level: str, color: str, icon: str) -> str:
    badge_colors = {
        "green":   ("#1a3a1a", "#4ade80"),
        "orange":  ("#3a2a00", "#fbbf24"),
        "red":     ("#3a0a0a", "#f87171"),
        "darkred": ("#2a0000", "#ef4444"),
    }
    bg, fg = badge_colors.get(color, ("#1e2130", "#a0aec0"))
    return (
        f'<span class="risk-badge" style="background:{bg};color:{fg};border:1px solid {fg}55">'
        f'{icon} {level.upper()}'
        f'</span>'
    )


def color_for_score(score: int) -> str:
    if score <= 25:  return "#4ade80"
    if score <= 50:  return "#fbbf24"
    if score <= 75:  return "#f87171"
    return "#ef4444"


# ─── Visual Scam Indicators section ──────────────────────────────────────────

def render_visual_section(visual_data: dict, component_scores: dict) -> None:
    """
    Renders the full 'Visual Scam Indicators' section including:
      - Visual risk score + contribution
      - Annotated frame gallery (YOLO bounding boxes)
      - CLIP scene classifications
      - QR codes decoded
      - Detection detail table
    """
    st.markdown("---")
    st.markdown(
        '<div class="section-header">👁️ Visual Scam Indicators (YOLOv8 + CLIP)</div>',
        unsafe_allow_html=True,
    )

    vis_score   = visual_data.get("visual_score", 0)
    detections  = visual_data.get("detections", [])
    ann_frames  = visual_data.get("annotated_frames", [])
    qr_codes    = visual_data.get("qr_codes_found", [])
    clip_hits   = visual_data.get("clip_classifications", [])

    # ── Summary row ───────────────────────────────────────────────────────────
    col_vscore, col_vchips, col_vcontrib = st.columns([1, 1.5, 1.5])

    with col_vscore:
        clr = color_for_score(vis_score)
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="color:#a0aec0;font-size:12px;text-transform:uppercase;letter-spacing:1px">Visual Risk Score</div>'
            f'<div style="font-size:52px;font-weight:800;color:{clr};line-height:1.1">{vis_score}</div>'
            f'<div style="color:#718096;font-size:12px">out of 100</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_vchips:
        st.markdown('<div style="margin-bottom:6px;font-size:13px;color:#a0aec0">Detected Visual Indicators</div>',
                    unsafe_allow_html=True)
        render_visual_chips(detections)

    with col_vcontrib:
        st.markdown('<div style="margin-bottom:6px;font-size:13px;color:#a0aec0">Score Contribution (15% weight)</div>',
                    unsafe_allow_html=True)
        vis_contribution = round(visual_data.get("visual_score", 0) * 0.15)
        st.markdown(
            f'<div style="font-size:28px;font-weight:700;color:#60a5fa">+{vis_contribution} pts</div>'
            f'<div style="color:#718096;font-size:12px">contributed to final risk score</div>',
            unsafe_allow_html=True,
        )

    # ── QR codes ──────────────────────────────────────────────────────────────
    if qr_codes:
        st.markdown("**🔲 QR Codes Decoded:**")
        chips = " ".join(f'<span class="qr-chip">📎 {qr[:80]}</span>' for qr in qr_codes)
        st.markdown(chips, unsafe_allow_html=True)
        st.warning(
            f"⚠️ {len(qr_codes)} QR code(s) found. "
            "Scam videos use QR codes to route victims to phishing pages or Telegram channels.",
            icon="🔲",
        )

    # ── Annotated frame gallery ───────────────────────────────────────────────
    if ann_frames:
        with st.expander(
            f"📸 Annotated Frames ({len(ann_frames)} frames with detections)",
            expanded=True,
        ):
            cols_per_row = 3
            for row_start in range(0, len(ann_frames), cols_per_row):
                row_frames = ann_frames[row_start : row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, frame_rgb in zip(cols, row_frames):
                    with col:
                        st.image(
                            frame_rgb,
                            use_container_width=True,
                            caption=f"Frame {row_start + list(zip(cols, row_frames)).index((col, frame_rgb)) + 1}",
                        )
    else:
        st.info("No frames with YOLO detections to display.", icon="ℹ️")

    # ── CLIP scene classifications ─────────────────────────────────────────────
    if clip_hits:
        with st.expander(f"🎬 CLIP Scene Classifications ({len(clip_hits)} results)", expanded=False):
            # Deduplicate: show best confidence per label
            best: dict[str, dict] = {}
            for d in clip_hits:
                if d["label"] not in best or d["confidence"] > best[d["label"]]["confidence"]:
                    best[d["label"]] = d

            for lbl, d in sorted(best.items(), key=lambda x: x[1]["confidence"], reverse=True):
                conf  = d["confidence"]
                score = d["score"]
                clr   = color_for_score(score * 4)
                st.markdown(
                    f'<div style="margin-bottom:10px">'
                    f'<div style="display:flex;justify-content:space-between;color:#e2e8f0;font-size:13px">'
                    f'<span>🎬 {lbl}</span>'
                    f'<span style="color:{clr}">{conf:.0%} confidence → +{score} pts</span>'
                    f'</div>'
                    f'<div style="background:#2d3250;border-radius:4px;height:5px;margin-top:4px">'
                    f'<div style="background:{clr};width:{conf*100:.0f}%;height:5px;border-radius:4px"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    # ── Detection detail table ─────────────────────────────────────────────────
    non_clip = [d for d in detections if d["source"] != "clip"]
    if non_clip:
        with st.expander(f"🔍 YOLO + QR Detection Log ({len(non_clip)} detections)", expanded=False):
            import pandas as pd
            rows = [
                {
                    "Frame":      d["frame_idx"] + 1,
                    "Object":     d["label"],
                    "Category":   d["category"],
                    "Confidence": f"{d['confidence']:.0%}",
                    "Risk +pts":  d["score"],
                    "Source":     d["source"].upper(),
                }
                for d in non_clip
            ]
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

    if not detections and not qr_codes and not clip_hits:
        st.success("No visual scam indicators detected in this video.", icon="✅")


# ─── Main app ─────────────────────────────────────────────────────────────────

def main() -> None:

    # Header
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0">
        <h1 style="font-size:2.4rem; font-weight:800; color:white; margin:0">
            🔍 AI Media Watch
        </h1>
        <p style="color:#718096; font-size:1rem; margin-top:6px">
            Scam & Fraud Detection · Speech · OCR · YOLOv8 · CLIP
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload section ────────────────────────────────────────────────────────
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload a video file",
            type=["mp4", "mov", "avi", "mkv", "webm", "flv"],
            help="Supports MP4, MOV, AVI, MKV, WebM, FLV",
        )

    with col_info:
        st.markdown("""
        <div class="metric-card" style="margin-top:8px">
            <div style="color:#a0aec0; font-size:13px">
                <b style="color:white">What we detect</b><br><br>
                🎰 Gambling & Betting Ads<br>
                ₿ Crypto Scams<br>
                📈 Investment Fraud<br>
                🔺 Pyramid / MLM Schemes<br>
                🎁 Fake Giveaways<br>
                👁️ Visual Scam Indicators<br>
                🔲 QR Code Routing<br>
                🎬 Scene Classification (CLIP)
            </div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded is None:
        st.info("⬆️ Upload a video above to begin analysis.")
        return

    # Save upload to a temp file
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    st.video(video_path)

    if not st.button("🔍 Analyze Video", type="primary", use_container_width=False):
        return

    # ── Pipeline ──────────────────────────────────────────────────────────────
    result_data: dict = {}

    with st.status("Running analysis pipeline…", expanded=True) as status:

        # Step 1 – audio + frames
        st.write("📹 Extracting audio and frames…")
        from video_processing import process_video
        video_data = process_video(video_path)
        meta = video_data["metadata"]
        st.write(
            f"   ✔ {len(video_data['frame_paths'])} frames | "
            f"{meta['duration_sec']:.1f}s | "
            f"{meta['width']}×{meta['height']}"
        )

        # Step 2 – speech-to-text
        transcript_text = ""
        language = "unknown"
        if video_data["audio_path"]:
            st.write("🎙️ Transcribing audio (Faster-Whisper)…")
            from speech_to_text import transcribe
            stt = transcribe(video_data["audio_path"], model=get_whisper_model())
            transcript_text = stt["full_text"]
            language = stt["language"]
            st.write(f"   ✔ {len(transcript_text)} chars | Language: {language}")
        else:
            st.write("   ℹ️ No audio track — skipping transcription.")

        # Step 3 – OCR
        st.write("👁️ Extracting text from frames (EasyOCR)…")
        from ocr import extract_text_from_frames
        ocr_data = extract_text_from_frames(video_data["frame_paths"], reader=get_ocr_reader())
        ocr_text = ocr_data["combined_text"]
        st.write(f"   ✔ {len(ocr_data['unique_lines'])} unique text lines")

        # Step 4 – Visual analysis (YOLOv8 + pyzbar + CLIP)
        st.write("👁️‍🗨️ Running visual analysis (YOLOv8 + CLIP)…")
        from vision_analysis import analyze_frames
        clip_models = get_clip_models()   # None if ENABLE_CLIP_ANALYSIS=False
        visual_data = analyze_frames(
            video_data["frame_paths"],
            yolo_model=get_yolo_model(),
            clip_models=clip_models,
        )
        st.write(
            f"   ✔ {len(visual_data['detections'])} visual detections | "
            f"Visual score: {visual_data['visual_score']}/100"
        )
        if visual_data["qr_codes_found"]:
            st.write(f"   🔲 QR codes found: {len(visual_data['qr_codes_found'])}")

        # Step 5 – NLP
        st.write("🧠 Running NLP scam classification…")
        combined_text = f"{transcript_text} {ocr_text}".strip()
        from nlp_analysis import analyze
        analysis = analyze(combined_text, classifier=get_nlp_classifier())
        st.write("   ✔ NLP analysis complete")

        # Step 6 – Risk engine (4-component)
        st.write("⚖️ Computing risk score…")
        from risk_engine import compute_risk
        risk = compute_risk(transcript_text, ocr_text, analysis, visual_result=visual_data)
        st.write(f"   ✔ Final risk: {risk['risk_score']}/100 — {risk['risk_icon']} {risk['risk_level']}")

        result_data = {
            "risk":       risk,
            "transcript": transcript_text,
            "ocr":        ocr_data,
            "visual":     visual_data,
            "language":   language,
            "meta":       meta,
        }

        shutil.rmtree(video_data["temp_dir"], ignore_errors=True)
        status.update(label="Analysis complete!", state="complete")

    if not result_data:
        return

    # ── Results dashboard ─────────────────────────────────────────────────────
    risk  = result_data["risk"]
    score = risk["risk_score"]
    clr   = color_for_score(score)

    st.markdown("---")
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

    # Row 1: gauge + categories + component bar
    col_gauge, col_cats, col_comp = st.columns([1, 1.4, 1.4])

    with col_gauge:
        st.plotly_chart(make_gauge(score, clr), use_container_width=True)
        badge_html = render_risk_badge(risk["risk_level"], risk["risk_color"], risk["risk_icon"])
        st.markdown(f'<div style="text-align:center">{badge_html}</div>', unsafe_allow_html=True)

    with col_cats:
        st.markdown('<div class="section-header" style="margin-top:10px">Detected Patterns</div>',
                    unsafe_allow_html=True)
        if risk["top_categories"]:
            for label, pct in risk["top_categories"]:
                bar_color = color_for_score(pct)
                st.markdown(
                    f'<div style="margin-bottom:10px">'
                    f'<div style="display:flex;justify-content:space-between;color:#e2e8f0;font-size:13px">'
                    f'<span>{label}</span><span style="color:{bar_color}">{pct}%</span></div>'
                    f'<div style="background:#2d3250;border-radius:4px;height:6px;margin-top:4px">'
                    f'<div style="background:{bar_color};width:{pct}%;height:6px;border-radius:4px"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No significant text-based scam categories detected.")

    with col_comp:
        st.markdown('<div class="section-header" style="margin-top:10px">Score Breakdown</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_component_bar(risk["component_scores"]),
            use_container_width=True,
        )

    # Row 2: keywords + reasons
    st.markdown("---")
    col_kw, col_reasons = st.columns([1, 1])

    with col_kw:
        st.markdown('<div class="section-header">⚠️ Flagged Keywords</div>', unsafe_allow_html=True)
        render_keyword_chips(risk["matched_keywords"])

    with col_reasons:
        st.markdown('<div class="section-header">📋 Detection Reasons</div>', unsafe_allow_html=True)
        if risk["reasons"]:
            for r in risk["reasons"]:
                st.markdown(f"- {r}")
        else:
            st.caption("No specific reasons generated.")

    # ── Visual Scam Indicators section (new) ──────────────────────────────────
    render_visual_section(result_data["visual"], risk["component_scores"])

    # Row 3: transcript + OCR (expandable)
    st.markdown("---")
    col_tr, col_ocr = st.columns(2)

    with col_tr:
        with st.expander(f"📝 Speech Transcript ({result_data['language'].upper()})", expanded=False):
            if result_data["transcript"].strip():
                st.text_area("", value=result_data["transcript"],
                             height=200, label_visibility="collapsed")
            else:
                st.caption("No speech detected.")

    with col_ocr:
        with st.expander(
            f"📷 OCR Text from Frames ({len(result_data['ocr']['unique_lines'])} lines)",
            expanded=False,
        ):
            if result_data["ocr"]["unique_lines"]:
                st.text_area("", value="\n".join(result_data["ocr"]["unique_lines"]),
                             height=200, label_visibility="collapsed")
            else:
                st.caption("No text detected in frames.")

    # Footer metadata
    m = result_data["meta"]
    st.caption(
        f"Video: {uploaded.name} | "
        f"{m['duration_sec']:.1f}s | "
        f"{m['width']}×{m['height']} @ {m['fps']:.1f}fps | "
        f"Frames sampled: {len(result_data['ocr']['per_frame'])} | "
        f"Visual detections: {len(result_data['visual']['detections'])}"
    )

    try:
        os.unlink(video_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
