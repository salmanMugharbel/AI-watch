"""
app.py – AI Media Watch: Scam Detection in Social Media Videos

Pipeline:
  1. Video → audio + adaptive frames
  2. Faster-Whisper  → transcript
  3. EasyOCR         → frame text  (with preprocessing)
  4. YOLOv8 + CLIP + pyzbar → visual detections  (CLIP-gated YOLO)
  5. BART-MNLI       → two-pass NLP classification
  6. compute_risk()  → preliminary score  (dynamic weights + floors + boost)
  7. llm_reasoning   → Ollama additive pass  (euphemisms, coded language)
  8. apply_llm_boost → final score
  9. Dashboard       → full results + LLM reasoning panel
"""

import os
import shutil
import tempfile
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Media Watch",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: #1e2130; border-radius: 12px;
        padding: 20px 24px; border: 1px solid #2d3250; text-align: center;
    }
    .risk-badge {
        display: inline-block; padding: 4px 16px; border-radius: 20px;
        font-size: 14px; font-weight: 600; letter-spacing: 1px;
        text-transform: uppercase; margin-top: 6px;
    }
    .kw-chip {
        display: inline-block; background: #2d1f1f; color: #ff6b6b;
        border: 1px solid #ff6b6b55; padding: 3px 10px;
        border-radius: 14px; font-size: 12px; margin: 3px;
    }
    .vis-chip {
        display: inline-block; background: #1f2a3d; color: #60a5fa;
        border: 1px solid #60a5fa55; padding: 3px 10px;
        border-radius: 14px; font-size: 12px; margin: 3px;
    }
    .qr-chip {
        display: inline-block; background: #2a1f1f; color: #f87171;
        border: 1px solid #f8717155; padding: 4px 12px;
        border-radius: 14px; font-size: 12px; margin: 3px; font-family: monospace;
    }
    .llm-phrase {
        display: inline-block; background: #1f2d1f; color: #86efac;
        border: 1px solid #86efac55; padding: 3px 10px;
        border-radius: 14px; font-size: 12px; margin: 3px; font-style: italic;
    }
    .weight-pill {
        display: inline-block; background: #252840; color: #a5b4fc;
        border: 1px solid #a5b4fc44; padding: 2px 8px;
        border-radius: 10px; font-size: 11px; margin: 2px;
    }
    .section-header {
        font-size: 16px; font-weight: 600; color: #a0aec0;
        text-transform: uppercase; letter-spacing: 1.5px;
        margin-bottom: 12px; border-bottom: 1px solid #2d3250; padding-bottom: 6px;
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
    from vision_analysis import load_yolo_model
    return load_yolo_model()

@st.cache_resource(show_spinner="Loading CLIP vision model…")
def get_clip_models():
    from config import ENABLE_CLIP_ANALYSIS
    if not ENABLE_CLIP_ANALYSIS:
        return None
    from vision_analysis import load_clip_model
    return load_clip_model()


# ─── Chart helpers ────────────────────────────────────────────────────────────

def make_gauge(score: int, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"size": 48, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#444",
                     "tickfont": {"color": "#888", "size": 11}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1e2130", "borderwidth": 0,
            "steps": [
                {"range": [0,  25], "color": "#1a3a1a"},
                {"range": [25, 50], "color": "#3a3010"},
                {"range": [50, 75], "color": "#3a1a10"},
                {"range": [75, 100], "color": "#3a0a0a"},
            ],
            "threshold": {"line": {"color": "white", "width": 3},
                          "thickness": 0.75, "value": score},
        },
    ))
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      margin={"t": 30, "b": 10, "l": 20, "r": 20}, height=220)
    return fig


def make_component_bar(component_scores: dict, weights_used: dict) -> go.Figure:
    labels = ["Keyword", "NLP", "Pattern", "Visual"]
    keys   = ["keyword", "nlp", "pattern", "visual"]
    values = [component_scores.get(k, 0) for k in keys]
    wts    = [f"{weights_used.get(k, 0):.0%}" for k in keys]
    colors = ["#818cf8", "#34d399", "#fbbf24", "#60a5fa"]

    fig = go.Figure(go.Bar(
        x=values, y=[f"{l} (w={w})" for l, w in zip(labels, wts)],
        orientation="h", marker_color=colors,
        text=[f"{v}/100" for v in values], textposition="outside",
        textfont={"color": "white", "size": 11},
    ))
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1e2130",
        margin={"t": 10, "b": 10, "l": 10, "r": 60}, height=170,
        xaxis={"range": [0, 120], "showgrid": False,
               "color": "#444", "tickfont": {"color": "#888"}},
        yaxis={"showgrid": False, "tickfont": {"color": "#e2e8f0", "size": 11}},
        showlegend=False,
    )
    return fig


# ─── Render helpers ───────────────────────────────────────────────────────────

def color_for_score(score: int) -> str:
    if score <= 25:  return "#4ade80"
    if score <= 50:  return "#fbbf24"
    if score <= 75:  return "#f87171"
    return "#ef4444"


def render_risk_badge(level: str, color: str, icon: str) -> str:
    badge_map = {
        "green":   ("#1a3a1a", "#4ade80"),
        "orange":  ("#3a2a00", "#fbbf24"),
        "red":     ("#3a0a0a", "#f87171"),
        "darkred": ("#2a0000", "#ef4444"),
    }
    bg, fg = badge_map.get(color, ("#1e2130", "#a0aec0"))
    return (f'<span class="risk-badge" style="background:{bg};color:{fg};'
            f'border:1px solid {fg}55">{icon} {level.upper()}</span>')


def render_keyword_chips(keywords: list[str]) -> None:
    if not keywords:
        st.caption("No scam keywords matched.")
        return
    st.markdown(
        " ".join(f'<span class="kw-chip">{kw}</span>' for kw in sorted(keywords)),
        unsafe_allow_html=True,
    )


def render_visual_chips(detections: list[dict]) -> None:
    if not detections:
        st.caption("No visual indicators detected.")
        return
    seen: set[str] = set()
    chips = []
    for d in detections:
        lbl = d["label"]
        if lbl not in seen:
            seen.add(lbl)
            chips.append(f'<span class="vis-chip">{lbl}</span>')
    st.markdown(" ".join(chips), unsafe_allow_html=True)


# ─── Visual section ───────────────────────────────────────────────────────────

def render_visual_section(visual_data: dict, component_scores: dict) -> None:
    st.markdown("---")
    st.markdown(
        '<div class="section-header">👁️ Visual Scam Indicators (YOLOv8 + CLIP)</div>',
        unsafe_allow_html=True,
    )

    vis_score  = visual_data.get("visual_score", 0)
    detections = visual_data.get("detections", [])
    ann_frames = visual_data.get("annotated_frames", [])
    qr_codes   = visual_data.get("qr_codes_found", [])
    clip_hits  = visual_data.get("clip_classifications", [])

    col_vs, col_vc, col_vw = st.columns([1, 1.5, 1.5])

    with col_vs:
        clr = color_for_score(vis_score)
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="color:#a0aec0;font-size:12px;text-transform:uppercase;letter-spacing:1px">Visual Score</div>'
            f'<div style="font-size:52px;font-weight:800;color:{clr};line-height:1.1">{vis_score}</div>'
            f'<div style="color:#718096;font-size:12px">out of 100</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_vc:
        st.markdown('<div style="margin-bottom:6px;font-size:13px;color:#a0aec0">Detected Indicators</div>',
                    unsafe_allow_html=True)
        render_visual_chips(detections)

    with col_vw:
        vis_contrib = round(vis_score * 0.25)
        st.markdown(
            f'<div style="font-size:28px;font-weight:700;color:#60a5fa">+{vis_contrib} pts</div>'
            f'<div style="color:#718096;font-size:12px">contributed to final score (25% weight)</div>',
            unsafe_allow_html=True,
        )

    if qr_codes:
        chips = " ".join(f'<span class="qr-chip">📎 {qr[:80]}</span>' for qr in qr_codes)
        st.markdown(f"**🔲 QR Codes:** {chips}", unsafe_allow_html=True)
        st.warning(f"{len(qr_codes)} QR code(s) detected — scam videos use QR codes to route victims to phishing pages.", icon="🔲")

    if ann_frames:
        with st.expander(f"📸 Annotated Frames ({len(ann_frames)} with detections)", expanded=True):
            cols_per_row = 3
            for i in range(0, len(ann_frames), cols_per_row):
                row = ann_frames[i: i + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, frm in zip(cols, row):
                    col.image(frm, use_container_width=True)

    if clip_hits:
        with st.expander(f"🎬 CLIP Scene Scores ({len(clip_hits)} results)", expanded=False):
            best: dict[str, dict] = {}
            for d in clip_hits:
                if d["label"] not in best or d["confidence"] > best[d["label"]]["confidence"]:
                    best[d["label"]] = d
            for lbl, d in sorted(best.items(), key=lambda x: x[1]["confidence"], reverse=True):
                clr = color_for_score(d["score"] * 4)
                st.markdown(
                    f'<div style="margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;color:#e2e8f0;font-size:13px">'
                    f'<span>🎬 {lbl}</span>'
                    f'<span style="color:{clr}">cos={d["confidence"]:.3f} → +{d["score"]} pts</span></div>'
                    f'<div style="background:#2d3250;border-radius:4px;height:5px;margin-top:3px">'
                    f'<div style="background:{clr};width:{min(d["confidence"]*100/0.4*100, 100):.0f}%;'
                    f'height:5px;border-radius:4px"></div></div></div>',
                    unsafe_allow_html=True,
                )

    non_clip = [d for d in detections if d["source"] != "clip"]
    if non_clip:
        with st.expander(f"🔍 YOLO + QR Log ({len(non_clip)} detections)", expanded=False):
            import pandas as pd
            df = pd.DataFrame([{
                "Frame": d["frame_idx"] + 1,
                "Object": d["label"],
                "Category": d["category"],
                "Confidence": f"{d['confidence']:.0%}",
                "Points": d["score"],
                "Gate": "✅" if d.get("gate_passed") else "⬜",
                "Source": d["source"].upper(),
            } for d in non_clip])
            st.dataframe(df, use_container_width=True, hide_index=True)

    if not detections and not qr_codes and not clip_hits:
        st.success("No visual scam indicators detected.", icon="✅")


# ─── LLM reasoning section ────────────────────────────────────────────────────

def render_llm_section(llm_result: dict) -> None:
    st.markdown("---")
    st.markdown(
        '<div class="section-header">🧠 LLM Reasoning Layer (Ollama)</div>',
        unsafe_allow_html=True,
    )

    if not llm_result or not llm_result.get("ran"):
        skip = llm_result.get("skip_reason", "not run") if llm_result else "not run"
        st.info(f"LLM reasoning was not invoked — {skip}", icon="ℹ️")
        return

    conf = llm_result.get("confidence", 0.0)
    col_a, col_b = st.columns([1, 1.5])

    with col_a:
        flags = {
            "euphemism_detected":        ("Euphemistic Language",    "🎭"),
            "coded_language_detected":   ("Coded Financial Language","🔐"),
            "implied_promises_detected": ("Implied Wealth Promises", "💸"),
            "social_proof_manipulation": ("Social Proof Manipulation","👥"),
            "soft_urgency_detected":     ("Soft Urgency Tactics",    "⏳"),
        }
        st.markdown(f"**LLM confidence: {conf:.0%}**")
        for key, (label, icon) in flags.items():
            detected = llm_result.get(key, False)
            color    = "#86efac" if detected else "#4b5563"
            status   = "Detected" if detected else "Not found"
            st.markdown(
                f'<div style="padding:4px 0;color:{color};font-size:13px">'
                f'{icon} {label}: <b>{status}</b></div>',
                unsafe_allow_html=True,
            )

    with col_b:
        phrases = llm_result.get("flagged_phrases", [])
        if phrases:
            st.markdown("**Flagged phrases:**")
            chips = " ".join(f'<span class="llm-phrase">"{p}"</span>' for p in phrases[:8])
            st.markdown(chips, unsafe_allow_html=True)

        hint = llm_result.get("scam_type_hint", "")
        if hint and hint.lower() != "none":
            st.markdown(f"**Scam type hint:** `{hint}`")

        explanation = llm_result.get("explanation", "")
        if explanation:
            st.markdown(f"**Summary:** {explanation}")


# ─── Main app ─────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 20px 0">
        <h1 style="font-size:2.4rem;font-weight:800;color:white;margin:0">🔍 AI Media Watch</h1>
        <p style="color:#718096;font-size:1rem;margin-top:6px">
            Scam & Fraud Detection · Speech · OCR · YOLOv8 · CLIP · LLM Reasoning
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_info = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload a video file",
            type=["mp4", "mov", "avi", "mkv", "webm", "flv"],
        )
    with col_info:
        st.markdown("""
        <div class="metric-card" style="margin-top:8px">
            <div style="color:#a0aec0;font-size:13px">
                <b style="color:white">Detection layers</b><br><br>
                🎙️ Faster-Whisper transcript<br>
                👁️ EasyOCR + preprocessing<br>
                🧠 BART-MNLI two-pass NLP<br>
                👁️‍🗨️ YOLOv8 (CLIP-gated)<br>
                🎬 CLIP cosine similarity<br>
                🔲 pyzbar QR decoding<br>
                🤖 Ollama euphemism layer<br>
                ⚖️ Dynamic risk engine
            </div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded is None:
        st.info("⬆️ Upload a video to begin analysis.")
        return

    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    st.video(video_path)

    if not st.button("🔍 Analyze Video", type="primary"):
        return

    result_data: dict = {}

    with st.status("Running analysis pipeline…", expanded=True) as status:

        # Step 1 – video
        st.write("📹 Extracting audio and frames (adaptive sampling)…")
        from video_processing import process_video
        video_data = process_video(video_path)
        meta = video_data["metadata"]
        st.write(
            f"   ✔ {len(video_data['frame_paths'])} frames sampled | "
            f"{meta['duration_sec']:.1f}s | {meta['width']}×{meta['height']}"
        )

        # Step 2 – speech
        transcript_text = ""
        language = "unknown"
        if video_data["audio_path"]:
            st.write("🎙️ Transcribing audio…")
            from speech_to_text import transcribe
            stt = transcribe(video_data["audio_path"], model=get_whisper_model())
            transcript_text = stt["full_text"]
            language = stt["language"]
            st.write(f"   ✔ {len(transcript_text)} chars | Language: {language}")
        else:
            st.write("   ℹ️ No audio — skipping transcription.")

        # Step 3 – OCR
        st.write("👁️ Extracting text from frames (EasyOCR + preprocessing)…")
        from ocr import extract_text_from_frames
        ocr_data = extract_text_from_frames(video_data["frame_paths"], reader=get_ocr_reader())
        ocr_text = ocr_data["combined_text"]
        st.write(f"   ✔ {len(ocr_data['unique_lines'])} unique text lines")

        # Step 4 – visual
        st.write("👁️‍🗨️ Running visual analysis (CLIP cosine → YOLO gate → QR)…")
        from vision_analysis import analyze_frames
        visual_data = analyze_frames(
            video_data["frame_paths"],
            yolo_model=get_yolo_model(),
            clip_models=get_clip_models(),
        )
        st.write(
            f"   ✔ {len(visual_data['detections'])} detections | "
            f"Visual score: {visual_data['visual_score']}/100"
        )
        if visual_data["qr_codes_found"]:
            st.write(f"   🔲 QR codes: {len(visual_data['qr_codes_found'])}")

        # Step 5 – NLP (two-pass)
        st.write("🧠 Running two-pass NLP classification…")
        combined_text = f"{transcript_text} {ocr_text}".strip()
        from nlp_analysis import analyze
        analysis = analyze(combined_text, classifier=get_nlp_classifier())
        tier1 = analysis["nlp"].get("tier1_triggered", False)
        st.write(
            f"   ✔ NLP complete | "
            f"Top scam score: {analysis['nlp'].get('top_scam_score', 0):.0%} | "
            f"Tier-1 triggered: {'yes' if tier1 else 'no'}"
        )

        # Step 6 – preliminary risk (no LLM yet)
        st.write("⚖️ Computing preliminary risk score…")
        from risk_engine import compute_risk, apply_llm_boost
        prelim_risk = compute_risk(transcript_text, ocr_text, analysis, visual_data)
        prelim_score = prelim_risk["risk_score"]
        st.write(
            f"   ✔ Preliminary: {prelim_score}/100 | "
            f"Weights: " +
            " ".join(
                f'{k}={v:.0%}'
                for k, v in prelim_risk["weights_used"].items()
                if v > 0
            )
        )

        # Step 7 – LLM reasoning (additive, optional)
        from config import ENABLE_LLM_REASONING, LLM_TRIGGER_MIN_SCORE, LLM_TRIGGER_MAX_SCORE
        llm_result = None
        if (ENABLE_LLM_REASONING
                and LLM_TRIGGER_MIN_SCORE <= prelim_score <= LLM_TRIGGER_MAX_SCORE):
            st.write(f"🤖 Running LLM reasoning (preliminary score {prelim_score} in trigger range)…")
            from llm_reasoning import reason
            llm_result = reason(combined_text, prelim_score)
            if llm_result.get("ran"):
                st.write(
                    f"   ✔ LLM confidence: {llm_result['confidence']:.0%} | "
                    f"Hint: {llm_result.get('scam_type_hint', 'none')}"
                )
            else:
                st.write(f"   ℹ️ LLM skipped — {llm_result.get('skip_reason', '')}")
        else:
            st.write("   ℹ️ LLM reasoning skipped (score outside trigger range or disabled)")

        # Step 8 – final risk (with LLM boost)
        if llm_result and llm_result.get("ran"):
            final_risk = apply_llm_boost(prelim_risk, llm_result)
        else:
            final_risk = prelim_risk
            llm_result = llm_result or {}

        st.write(
            f"   ✔ Final risk: {final_risk['risk_score']}/100 — "
            f"{final_risk['risk_icon']} {final_risk['risk_level']}"
        )

        result_data = {
            "risk":       final_risk,
            "transcript": transcript_text,
            "ocr":        ocr_data,
            "visual":     visual_data,
            "llm":        llm_result,
            "language":   language,
            "meta":       meta,
        }

        shutil.rmtree(video_data["temp_dir"], ignore_errors=True)
        status.update(label="Analysis complete!", state="complete")

    if not result_data:
        return

    # ── Dashboard ─────────────────────────────────────────────────────────────
    risk  = result_data["risk"]
    score = risk["risk_score"]
    clr   = color_for_score(score)

    st.markdown("---")
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

    col_g, col_c, col_b = st.columns([1, 1.4, 1.4])

    with col_g:
        st.plotly_chart(make_gauge(score, clr), use_container_width=True)
        badge = render_risk_badge(risk["risk_level"], risk["risk_color"], risk["risk_icon"])
        st.markdown(f'<div style="text-align:center">{badge}</div>', unsafe_allow_html=True)
        # Show whether LLM ran
        if risk.get("llm_ran"):
            st.markdown(
                '<div style="text-align:center;margin-top:6px;font-size:11px;color:#86efac">🤖 LLM boost applied</div>',
                unsafe_allow_html=True,
            )

    with col_c:
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
            st.caption("No text-based scam patterns detected.")

    with col_b:
        st.markdown('<div class="section-header" style="margin-top:10px">Score Breakdown</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_component_bar(risk["component_scores"], risk.get("weights_used", {})),
            use_container_width=True,
        )

    # Keywords + reasons
    st.markdown("---")
    col_kw, col_r = st.columns(2)
    with col_kw:
        st.markdown('<div class="section-header">⚠️ Flagged Keywords</div>', unsafe_allow_html=True)
        render_keyword_chips(risk["matched_keywords"])
    with col_r:
        st.markdown('<div class="section-header">📋 Detection Reasons</div>', unsafe_allow_html=True)
        for r in risk["reasons"]:
            st.markdown(f"- {r}")
        if not risk["reasons"]:
            st.caption("No reasons generated.")

    # Visual section
    render_visual_section(result_data["visual"], risk["component_scores"])

    # LLM reasoning section
    render_llm_section(result_data["llm"])

    # Transcript + OCR
    st.markdown("---")
    col_tr, col_ocr = st.columns(2)
    with col_tr:
        with st.expander(f"📝 Transcript ({result_data['language'].upper()})", expanded=False):
            if result_data["transcript"].strip():
                st.text_area("", value=result_data["transcript"], height=200,
                             label_visibility="collapsed")
            else:
                st.caption("No speech detected.")
    with col_ocr:
        with st.expander(
            f"📷 OCR Text ({len(result_data['ocr']['unique_lines'])} lines)", expanded=False,
        ):
            if result_data["ocr"]["unique_lines"]:
                st.text_area("", value="\n".join(result_data["ocr"]["unique_lines"]),
                             height=200, label_visibility="collapsed")
            else:
                st.caption("No text detected in frames.")

    m = result_data["meta"]
    st.caption(
        f"Video: {uploaded.name} | {m['duration_sec']:.1f}s | "
        f"{m['width']}×{m['height']} @ {m['fps']:.1f}fps | "
        f"Frames: {len(result_data['ocr']['per_frame'])} | "
        f"Visual detections: {len(result_data['visual']['detections'])}"
    )

    try:
        os.unlink(video_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
