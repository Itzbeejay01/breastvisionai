"""Streamlit implementation of the BreastVisionAI React interface."""

import base64
import io

import streamlit as st
from PIL import Image

from prediction.services.model_registry import PSOModelRegistry
from prediction.services.prediction import predict_single


st.set_page_config(page_title="BreastVisionAI", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner=False)
def load_ensemble():
    return PSOModelRegistry.initialize()


def add_react_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --surface:#fbfcfa; --primary:#005f6c; --secondary:#087c72; --ink:#153e42; --muted:#627779; }
        .stApp { background:#fbfcfa; color:var(--ink); font-family:'Plus Jakarta Sans',sans-serif; background-image:radial-gradient(circle at 8% 0%,rgba(0,95,108,.07),transparent 28rem),radial-gradient(circle at 92% 18%,rgba(0,108,68,.045),transparent 24rem); }
        .block-container { max-width:1280px; padding:1.4rem 3rem 4rem; }
        h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink) !important; letter-spacing:-.04em; }
        .topbar { display:flex; justify-content:space-between; align-items:center; padding:8px 0 26px; border-bottom:1px solid rgba(25,28,29,.08); margin-bottom:46px; }
        .brand { display:flex; align-items:center; gap:12px; color:var(--primary); font:700 21px 'Space Grotesk'; }.brand-mark { width:42px; height:42px; display:grid; place-items:center; border-radius:14px; background:linear-gradient(135deg,#c9f8ff,#d9ffe6); box-shadow:0 8px 20px rgba(0,95,108,.12); font-size:22px; }.nav-pill { color:var(--primary); font-weight:700; font-size:13px; padding:11px 17px; border-radius:999px; background:#e5faff; }
        .eyebrow,.kicker { color:var(--primary); font-size:11px; font-weight:800; letter-spacing:.16em; }.eyebrow { display:inline-flex; gap:10px; align-items:center; padding:8px 13px; border:1px solid #b9e8ed; border-radius:999px; background:rgba(229,250,255,.7); }.eyebrow-dot { width:7px; height:7px; border-radius:50%; background:#f07b75; box-shadow:0 0 0 5px rgba(240,123,117,.14); }
        .hero-title { font:600 clamp(42px,6vw,76px)/.98 'Space Grotesk'; letter-spacing:-.06em; color:var(--ink); margin:25px 0; }.gradient { background:linear-gradient(105deg,#006c78 5%,#168c79 46%,#d46271 95%); -webkit-background-clip:text; background-clip:text; color:transparent; }.hero-copy { max-width:610px; color:var(--muted); font-size:18px; line-height:1.75; }
        .scan-card { margin:12px auto; max-width:470px; padding:14px; border-radius:26px; background:rgba(255,255,255,.68); border:1px solid rgba(255,255,255,.9); box-shadow:0 30px 80px rgba(43,72,75,.19); transform:rotate(2deg); }.scan-top,.scan-bottom { display:flex; justify-content:space-between; padding:9px 5px 12px; color:#537072; font-size:10px; font-weight:800; letter-spacing:.12em; }.scan-image { height:315px; display:grid; place-items:center; position:relative; overflow:hidden; border-radius:17px; background:radial-gradient(circle at 52% 45%,#8dd4d1 0 4%,#1d686d 15%,#153e42 53%,#0e2b31 100%); color:#d6fff4; }.scan-image:after { content:''; position:absolute; left:5%; right:5%; top:48%; height:2px; background:linear-gradient(90deg,transparent,#d6fff4,transparent); box-shadow:0 0 16px #d6fff4; }.crosshair { width:82px; height:82px; border:1px solid rgba(209,255,242,.9); border-radius:50%; box-shadow:0 0 24px rgba(141,248,189,.5); font-size:34px; display:grid; place-items:center; }.scan-score { color:var(--primary); font:700 35px 'Space Grotesk'; letter-spacing:-.06em; }.risk { color:#08724d; padding:8px 11px; border-radius:999px; background:#e0f9eb; font-size:9px; }
        .feature { min-height:225px; padding:25px; border-radius:23px; border:1px solid rgba(25,28,29,.06); box-shadow:0 4px 20px rgba(0,0,0,.04); }.feature.teal { background:linear-gradient(145deg,#d8f8fa,#effefd); color:#064b55; }.feature.lilac { background:linear-gradient(145deg,#eee5ff,#fbf8ff); color:#563c78; }.feature.coral { background:linear-gradient(145deg,#ffe0dc,#fff5ef); color:#7a4544; }.feature-icon { font-size:28px; margin-bottom:28px; }.feature-tag { float:right; font-size:10px; font-weight:800; letter-spacing:.12em; opacity:.65; }.feature h3 { font-size:22px !important; margin:0 0 12px; }.feature p { line-height:1.65; opacity:.8; }
        .workflow { margin:35px 0; padding:40px; border-radius:28px; background:linear-gradient(115deg,#e5fbfc,#f8f0ff 55%,#fff0ea); border:1px solid rgba(0,95,108,.08); }.step { padding-top:8px; border-top:1px solid rgba(0,95,108,.17); }.step-number { color:#a35267; font:700 12px 'Space Grotesk'; }.step h3 { font-size:19px !important; margin:17px 0 5px; }.step p { color:var(--muted); font-size:13px; line-height:1.6; }
        .panel { padding:28px; border:1px solid rgba(120,160,163,.22); border-radius:24px; background:rgba(255,255,255,.82); box-shadow:0 14px 40px rgba(31,68,72,.07); }.insight { min-height:300px; color:white; background:linear-gradient(150deg,#064b55,#087c72 55%,#4c6c9a); }.insight h2 { color:white !important; font-size:32px !important; }.insight p { color:rgba(255,255,255,.7); line-height:1.7; }.model-row { display:flex; justify-content:space-between; padding:11px 14px; margin-top:7px; border-radius:10px; background:rgba(255,255,255,.12); }
        .verdict { padding:25px; border-radius:18px; background:#fff; border:1px solid rgba(120,160,163,.22); box-shadow:0 8px 30px rgba(0,0,0,.08); }.verdict-safe { border-top:7px solid #087c72; }.verdict-danger { border-top:7px solid #ba3d55; }.verdict h2 { font-size:23px !important; }.confidence { font:600 47px 'Space Grotesk'; color:var(--ink); }.bar { height:12px; background:#edf2f0; border-radius:99px; overflow:hidden; }.bar > div { height:100%; border-radius:99px; background:#087c72; }.dangerbar > div { background:#ba3d55; }
        .stButton > button { border-radius:13px; min-height:44px; border:1px solid #b7d5d6; color:var(--primary); background:rgba(255,255,255,.7); font-weight:700; }.stButton > button:hover { border-color:var(--primary); color:var(--primary); }.primary-btn button { color:#fff !important; background:linear-gradient(110deg,#005f6c,#137d73) !important; border:0 !important; }.stDownloadButton > button { border-radius:13px; }
        [data-testid='stFileUploader'] { border:1px solid #9ed8d7; border-radius:18px; padding:20px; background:rgba(255,255,255,.62); }.disclaimer { color:var(--muted); font-size:12px; text-align:center; margin-top:35px; }
        @media (max-width:800px) { .block-container { padding:1rem 1.2rem 3rem; }.topbar { margin-bottom:25px; }.scan-card { transform:none; }.workflow { padding:25px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header():
    st.markdown('<div class="topbar"><div class="brand"><span class="brand-mark">🩺</span>BreastVisionAI</div><div class="nav-pill">Clinical AI / Imaging Intelligence</div></div>', unsafe_allow_html=True)


def landing():
    header()
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="eyebrow"><span class="eyebrow-dot"></span> CLINICAL AI / IMAGING INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">See the signal<br><span class="gradient">behind every scan.</span></div>', unsafe_allow_html=True)
        st.markdown('<p class="hero-copy">BreastVisionAI turns complex breast imaging into a clearer, explainable starting point for clinical research and review.</p>', unsafe_allow_html=True)
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Open the workspace  →", key="open_workspace"):
            st.session_state.page = "workspace"
            st.rerun()
        st.markdown('</div><p style="color:#627779;margin-top:24px">✓ Explainable results &nbsp;&nbsp; 🔒 Secure workspace</p>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="scan-card"><div class="scan-top"><span>🔴 LIVE MODEL VIEW</span><span>SCAN 0248</span></div><div class="scan-image"><div class="crosshair">✦</div></div><div class="scan-bottom"><div>ENSEMBLE CONFIDENCE<br><span class="scan-score">94.8%</span></div><span class="risk">● LOW RISK</span></div></div>', unsafe_allow_html=True)

    st.markdown('<br><p class="kicker">WHY BREASTVISIONAI</p><h2>Intelligence you can<br><span class="gradient">understand and act on.</span></h2><p style="color:#627779;line-height:1.7">A focused toolkit for imaging teams who need speed, transparency, and a better view of the evidence.</p>', unsafe_allow_html=True)
    cards = [("teal", "◈", "01 / ENSEMBLE", "Three models. One clearer signal.", "EfficientNet, ResNet, and VGG16 work together through PSO-optimized fusion for dependable diagnostic confidence."), ("lilac", "◉", "02 / EXPLAINABILITY", "See what the model sees.", "Grad-CAM heatmaps reveal the tissue regions influencing each verdict, helping review every result with context."), ("coral", "✦", "03 / SCALE", "Built for real workflows.", "Move from one scan to a full triage queue with image upload and batch-ready processing.")]
    cols = st.columns(3, gap="medium")
    for col, (tone, icon, tag, title, desc) in zip(cols, cards):
        with col:
            st.markdown(f'<article class="feature {tone}"><span class="feature-icon">{icon}</span><span class="feature-tag">{tag}</span><h3>{title}</h3><p>{desc}</p></article>', unsafe_allow_html=True)
    st.markdown('<section class="workflow"><p class="kicker">A SIMPLE PATH TO CLARITY</p><h2>From image to insight.</h2><p style="color:#627779;line-height:1.7">A calm, repeatable workflow that keeps the clinical team in control at every step.</p></section>', unsafe_allow_html=True)
    steps = [("01", "Upload", "Add a raw or pre-processed scan."), ("02", "Analyze", "Three models evaluate it in parallel."), ("03", "Review", "Receive a calibrated clinical verdict."), ("04", "Explain", "Inspect the diagnostic heatmap.")]
    cols = st.columns(4)
    for col, (number, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="step"><span class="step-number">{number}</span><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.markdown('<p class="disclaimer">For research support only — not a substitute for professional medical diagnosis.</p>', unsafe_allow_html=True)


def workspace():
    header()
    left, right = st.columns([3, 1])
    with left:
        st.markdown('<p class="kicker">CLINICAL RESEARCH WORKSPACE</p><h1>Good to see you,<br><span class="gradient">researcher.</span></h1><p style="color:#627779;font-size:17px">Your diagnostic workspace is ready when you are.</p>', unsafe_allow_html=True)
    with right:
        if st.button("← Landing page"):
            st.session_state.page = "landing"
            st.rerun()

    upload, insight = st.columns([1.35, .65], gap="medium")
    with upload:
        st.markdown('<div class="panel"><p class="kicker">START HERE</p><h2>Quick diagnosis</h2><p style="color:#627779">Upload one scan for an immediate ensemble verdict with explainable AI.</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Add a DICOM, PNG, or JPEG scan", type=["jpg", "jpeg", "png", "bmp", "webp"])
        image_type = st.radio("Input type", ["raw", "processed"], format_func=str.title, horizontal=True)
        explain = st.checkbox("Generate Grad-CAM explanation", True)
        if uploaded:
            uploaded.seek(0)
            st.image(Image.open(uploaded), caption="Scan ready", width=340)
            uploaded.seek(0)
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            analyse = st.button("Start analysis  →")
            st.markdown('</div></div>', unsafe_allow_html=True)
            if analyse:
                with st.spinner("Analysing scan with the active ensemble…"):
                    try:
                        load_ensemble()
                        uploaded.seek(0)
                        st.session_state.result = predict_single(uploaded, image_type=image_type, generate_heatmap=explain)
                    except Exception as error:
                        st.error("The scan could not be analysed.")
                        st.exception(error)
        else:
            st.markdown('</div>', unsafe_allow_html=True)
    with insight:
        st.markdown('<div class="panel insight"><div style="font-size:28px">✦</div><p class="kicker" style="color:#b9fff2">THE BREASTVISION ENGINE</p><h2>Clarity at<br>clinical speed.</h2><p>A PSO-optimized ensemble designed to make complex imaging easier to review.</p><p style="color:#fff;font-weight:700">Active model ensemble</p><div class="model-row"><span>EfficientNet</span><b>01</b></div><div class="model-row"><span>ResNet</span><b>02</b></div><div class="model-row"><span>VGG16</span><b>03</b></div></div>', unsafe_allow_html=True)

    result = st.session_state.get("result")
    if result:
        malignant = result["prediction"] == "malignant"
        tone = "danger" if malignant else "safe"
        label = "HIGH RISK" if malignant else "LOW RISK"
        color = "#ba3d55" if malignant else "#087c72"
        icon = "⚠" if malignant else "✓"
        st.markdown(f'<br><div class="verdict verdict-{tone}"><h2>AI Diagnostic Verdict <span style="float:right;color:{color}">{icon} {label}</span></h2><div class="confidence">{result["confidence"]:.1%} <small style="font:500 13px Plus Jakarta Sans;color:#627779;text-transform:uppercase">confidence</small></div><div class="bar {"dangerbar" if malignant else ""}"><div style="width:{result["confidence"]:.1%}"></div></div><p style="color:#627779;line-height:1.7;margin-top:20px">The ensemble indicates a {"high" if malignant else "low"} probability of malignancy based on fused predictions from {len(result["selected_models"])} PSO-selected models, with a fused probability of {result["fused_probability"]:.1%}.</p></div>', unsafe_allow_html=True)
        st.markdown('<br><h2>Model breakdown</h2>', unsafe_allow_html=True)
        cols = st.columns(len(result["model_breakdown"]))
        for col, (name, values) in zip(cols, result["model_breakdown"].items()):
            with col:
                st.markdown(f'<div class="panel"><p class="kicker">{name}</p><div class="confidence" style="font-size:30px">{values["probability"]:.1%}</div><p style="color:#627779">Fusion weight: {values["weight"]:.1%}</p></div>', unsafe_allow_html=True)
        if result.get("heatmap_base64"):
            _, encoded = result["heatmap_base64"].split(",", 1)
            heatmap = Image.open(io.BytesIO(base64.b64decode(encoded)))
            st.markdown('<br><h2>Grad-CAM explanation</h2>', unsafe_allow_html=True)
            st.image(heatmap, caption="Highlighted areas show regions used by the primary model.", use_container_width=True)


def main():
    add_react_theme()
    if "page" not in st.session_state:
        st.session_state.page = "landing"
    landing() if st.session_state.page == "landing" else workspace()


if __name__ == "__main__":
    main()
