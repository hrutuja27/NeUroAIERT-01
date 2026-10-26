import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
import base64
import time
import hashlib
from groq import Groq



import json

MODELS = ["qwen/qwen3.8-27b"]


def _prepare_image(pil_image):
    img = pil_image.copy().convert("RGB")
    img.thumbnail((1024, 768))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def ai_report(pil_image):
    b64 = _prepare_image(pil_image)
    client = Groq(api_key=st.secrets["GROQ_API_KEY"], timeout=60)
    prompt = """You are assisting a student research prototype that drafts a preliminary, radiology-style observation report for REVIEW BY A RADIOLOGIST. You are NOT diagnosing. Use cautious wording such as 'appears', 'possible', 'cannot be excluded'. Never invent findings; if something is unclear, say 'not assessable'.

First state the image type (CT or MRI, plane, brain or other). If it is not a brain CT, say so and stop the detailed review.

Write the report in bullet points. Use standard medical terminology and name any possible condition, always as a possibility for the radiologist to confirm.

Use exactly these headings:
1. Image type and quality
2. Skull and scalp
3. Brain parenchyma (any abnormally bright or dark areas, with location)
4. Extra-axial spaces (around the brain surface)
5. Ventricles and basal cisterns
6. Midline shift
7. Features that could suggest hemorrhage (location, size, appearance), or 'none visually evident'
8. Confidence (low / medium / high) and why
9. Limitations of this AI review
10. Suggested next step for the clinician: urgent or routine radiologist review, based only on what is visible

End with: 'AI-generated draft. Not a diagnosis. Must be verified by a qualified radiologist.'"""
    last_error = None
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                max_tokens=1500,
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
    raise last_error


def ai_findings(pil_image):
    b64 = _prepare_image(pil_image)
    client = Groq(api_key=st.secrets["GROQ_API_KEY"], timeout=60)
    prompt = """Look at this medical scan. Return ONLY valid JSON, no other text, in this form: {"findings":[{"region":"Skull","status":"no abnormality evident","concern":0}]}. Include exactly these regions: Skull, Brain tissue, Ventricles, Extra-axial spaces, Midline, Possible hemorrhage. status must be one of: no abnormality evident, possible abnormality, not assessable. concern is an integer 0 (none) to 3 (high). Use 0 for not assessable. Be cautious and never invent findings."""
    try:
        response = client.chat.completions.create(
            model=MODELS[0],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            response_format={"type": "json_object"},
            max_tokens=800,
        )
        return json.loads(response.choices[0].message.content)["findings"]
    except Exception:
        return []
# ------------------------------------------------------------
# NeuroAlert AI — educational research dashboard
# IMPORTANT: Demo metrics below are simulated and are NOT
# calculated from the uploaded image. This is not a diagnostic tool.
# ------------------------------------------------------------

st.set_page_config(
    page_title="NeuroAlert AI | Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A compact custom theme. Streamlit's own controls remain accessible.
st.markdown(
    """
    <style>
    .stApp {
        background:
          radial-gradient(circle at 85% 0%, rgba(32, 115, 145, .16), transparent 30%),
          linear-gradient(180deg, #0b1220 0%, #101a2b 100%);
        color: #eef4ff;
    }
    [data-testid="stSidebar"] {
        background: #0a1120;
        border-right: 1px solid rgba(148, 163, 184, .18);
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .hero {
        padding: 1.45rem 1.6rem;
        border: 1px solid rgba(125, 211, 252, .22);
        border-radius: 20px;
        background: linear-gradient(120deg, rgba(15, 37, 61, .96), rgba(18, 58, 78, .72));
        margin-bottom: 1.15rem;
    }
    .eyebrow {
        color: #67e8f9; font-size: .78rem; font-weight: 800;
        letter-spacing: .16em; text-transform: uppercase;
    }
    .hero h1 { margin: .25rem 0 .35rem 0; font-size: 2.35rem; color: #f8fbff; }
    .hero p { margin: 0; color: #c3d2e7; font-size: 1.02rem; }
    .pill {
        display: inline-block; padding: .32rem .65rem; border-radius: 999px;
        font-size: .78rem; font-weight: 700; margin: .35rem .3rem 0 0;
        background: rgba(251, 191, 36, .13); color: #fcd34d;
        border: 1px solid rgba(251, 191, 36, .28);
    }
    .metric-card {
        background: rgba(20, 32, 51, .88);
        border: 1px solid rgba(148, 163, 184, .18);
        border-radius: 16px; padding: 1rem 1.05rem; min-height: 112px;
    }
    .metric-label { color: #9fb2cb; font-size: .82rem; margin-bottom: .35rem; }
    .metric-value { color: #f8fbff; font-size: 1.7rem; font-weight: 800; line-height: 1.2; }
    .metric-note { color: #fbbf24; font-size: .75rem; margin-top: .35rem; }
    .section-title { font-size: 1.3rem; font-weight: 800; margin: .45rem 0 .15rem 0; color: #f3f7ff; }
    .muted { color: #a9bad0; }
    .panel {
        padding: 1rem 1.1rem; border-radius: 15px;
        background: rgba(20, 32, 51, .72);
        border: 1px solid rgba(148, 163, 184, .17);
    }
    .step {
        padding: .85rem 1rem; margin: .45rem 0; border-radius: 12px;
        background: rgba(20, 32, 51, .8);
        border: 1px solid rgba(148, 163, 184, .16);
    }
    .small-note { color: #a9bad0; font-size: .82rem; }
    div[data-testid="stFileUploader"] {
        background: rgba(20, 32, 51, .55); border-radius: 14px; padding: .4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🧠 NeuroAlert AI")
    st.caption("Brain CT research prototype")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Image Workspace", "Demo Analytics", "AI Integration Roadmap"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("### Prototype status")
    st.warning("Research interface only")
    st.caption(
        "No trained medical model is connected. Any demo metrics shown here are illustrative."
    )
    st.markdown("---")
    st.caption("Version 1.0 · Educational project")

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI · Medical Imaging Research</div>
      <h1>🧠 NeuroAlert AI</h1>
      <p>Exploring how artificial intelligence could support review of brain CT images.</p>
      <span class="pill">SIMULATED DEMO DATA</span>
      <span class="pill">NOT FOR CLINICAL USE</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Safety notice: This prototype does not diagnose, rule out, or detect a medical condition. "
    "Uploaded images are displayed for interface testing only; they are not analyzed by AI."
)

# ---------- Shared image uploader ----------
with st.expander("Upload a brain CT image (optional)", expanded=(page == "Image Workspace")):
    uploaded_file = st.file_uploader(
        "Choose a PNG or JPG/JPEG image",
        type=["png", "jpg", "jpeg"],
        help="For interface testing, use a public educational image with no patient-identifying information.",
        key="ct_image_upload",
    )
    if uploaded_file is not None:
        try:
            raw = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(raw))
            image = ImageOps.exif_transpose(image).convert("RGB")
            st.session_state["uploaded_image_bytes"] = raw
            st.session_state["uploaded_image_name"] = uploaded_file.name
            st.session_state["uploaded_image_size"] = image.size
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("This file could not be opened as an image. Please choose a valid PNG or JPG file.")
            st.session_state.pop("uploaded_image_bytes", None)
            st.session_state.pop("uploaded_image_name", None)

# ---------- Pages ----------
if page == "Overview":
    st.markdown('<div class="section-title">Project overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="muted">A student research interface demonstrating image upload, transparent demo analytics, '
        'and a planned path toward a properly evaluated model.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Interface", "READY", "Upload and preview"),
        ("AI model", "NOT CONNECTED", "No real predictions"),
        ("Demo metrics", "SIMULATED", "For UI demonstration"),
        ("Clinical use", "NOT APPROVED", "Expert review required"),
    ]
    for col, (label, value, note) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="font-size:1.25rem">{value}</div>'
                f'<div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown('<div class="section-title">How the proposed workflow would work</div>', unsafe_allow_html=True)
        for number, title, desc in [
            ("01", "Upload", "Select a permitted educational CT image."),
            ("02", "Pre-process", "A future model pipeline would validate and prepare the image."),
            ("03", "Model inference", "A trained model would produce an output only after integration."),
            ("04", "Review", "A qualified clinician would assess findings; the tool would not replace clinical judgment."),
        ]:
            st.markdown(
                f'<div class="step"><b style="color:#67e8f9">{number}</b> &nbsp; '
                f'<b>{title}</b><br><span class="muted">{desc}</span></div>',
                unsafe_allow_html=True,
            )
    with right:
        st.markdown('<div class="section-title">Current build</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="panel">
              <p><b>Working now</b></p>
              <p>✓ Streamlit dashboard<br>✓ PNG/JPG upload<br>✓ Image preview<br>✓ Demo charts and project roadmap</p>
              <hr>
              <p><b>Not implemented yet</b></p>
              <p>○ Trained model inference<br>○ Validated clinical performance<br>○ Explainability overlays<br>○ Clinical workflow integration</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown('<div class="section-title">Project objective</div>', unsafe_allow_html=True)
    st.write(
        "NeuroAlert AI is a student research prototype exploring whether computer vision could assist "
        "research into brain CT image review. Any future model must be trained and evaluated on a suitable, "
        "permitted dataset and must be assessed for limitations and generalization before any real-world use."
    )


elif page == "Image Workspace":
    st.markdown('<div class="section-title">Image workspace</div>', unsafe_allow_html=True)
    st.caption("Preview an uploaded image and generate an AI-drafted observation report for expert review.")
    if "uploaded_image_bytes" not in st.session_state:
        st.markdown(
            '<div class="panel">No image selected yet. Use the uploader above to choose a PNG or JPG image.</div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            img = Image.open(io.BytesIO(st.session_state["uploaded_image_bytes"]))
            img = ImageOps.exif_transpose(img).convert("RGB")
            left, right = st.columns([1.1, 0.9], gap="large")
            with left:
                st.image(img, caption="Uploaded image", use_container_width=True)
            with right:
                st.markdown('<div class="section-title">File details</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.write(f"**Filename:** {st.session_state.get('uploaded_image_name', 'Uploaded image')}")
                st.write(f"**Dimensions:** {img.width} × {img.height} pixels")
                st.write("**Color mode:** RGB preview")
                st.write(f"**Loaded at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown("</div>", unsafe_allow_html=True)
                st.warning("AI observations are descriptive only and are not a medical assessment.")
                st.download_button(
                    "Download the uploaded image",
                    data=st.session_state["uploaded_image_bytes"],
                    file_name=st.session_state.get("uploaded_image_name", "uploaded_ct_image.png"),
                    mime="application/octet-stream",
                )

            st.markdown("---")
            img_id = hashlib.md5(st.session_state["uploaded_image_bytes"]).hexdigest()
            if st.button("Generate AI report"):
                with st.spinner("Analyzing image (up to 1-2 minutes)..."):
                    try:
                        st.session_state["ai_report"] = ai_report(img)
                        st.session_state["ai_findings"] = ai_findings(img)
                        st.session_state["ai_report_for"] = img_id
                    except Exception as e:
                        st.error(f"Could not generate the report. Please try again in a few minutes. ({e})")
            if st.session_state.get("ai_report") and st.session_state.get("ai_report_for") == img_id:
                st.subheader("AI preliminary observations")
                st.write(st.session_state["ai_report"])
                findings = st.session_state.get("ai_findings")
                if findings:
                    st.markdown("#### AI-estimated concern level by region")
                    fdf = pd.DataFrame(findings)
                    fig = px.bar(
                        fdf, x="concern", y="region", orientation="h",
                        color="status", range_x=[0, 3],
                        labels={"concern": "Concern level (0 none - 3 high)", "region": ""},
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Visual estimates by an AI model. Not measured values.")
                st.markdown("#### Image brightness histogram")
                hist = pd.DataFrame({
                    "Brightness": list(range(256)),
                    "Pixels": img.convert("L").histogram(),
                })
                st.plotly_chart(px.area(hist, x="Brightness", y="Pixels"), use_container_width=True)
                st.caption("AI-generated description. Not a diagnosis or medical assessment. Verify with a qualified radiologist.")
        except Exception:
            st.error("The image preview could not be displayed. Please upload another PNG or JPG file.")

elif page == "AI Integration Roadmap":
    st.markdown('<div class="section-title">Roadmap to a real model</div>', unsafe_allow_html=True)
    st.write(
        "The dashboard is currently a front-end prototype. Connecting a model requires a separate "
        "machine-learning workflow and evidence that the model performs reliably."
    )
    roadmap = [
        ("1", "Choose a permitted dataset", "Use a dataset with clear research permissions, documented labels, and suitable data governance."),
        ("2", "Define the task", "Specify whether the model classifies a whole scan, a slice, or localizes a finding. These are different tasks."),
        ("3", "Build preprocessing", "Match the model's expected input size, intensity/windowing, color channels, and normalization."),
        ("4", "Train and validate", "Use patient-level train/validation/test separation to reduce data leakage."),
        ("5", "Evaluate carefully", "Report sensitivity, specificity, AUROC, calibration, confidence intervals, and subgroup performance."),
        ("6", "Integrate inference", "Load the exact trained model and preprocessing pipeline; validate file format and input assumptions."),
        ("7", "Add safeguards", "Display limitations, model version, uncertainty, and clinician-review messaging; protect patient data."),
    ]
    for number, title, desc in roadmap:
        st.markdown(
            f'<div class="step"><b style="color:#67e8f9">{number}. {title}</b><br>'
            f'<span class="muted">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Suggested project structure")
    st.code(
        """neuroalert-ai/
├── app.py                # Streamlit interface
├── requirements.txt      # Python dependencies
├── models/
│   └── model_card.md     # Model details and limitations
├── src/
│   ├── preprocessing.py  # Exact model preprocessing
│   └── inference.py      # Load model and run inference
└── tests/
    └── test_preprocessing.py""",
        language="text",
    )
    st.info(
        "When a model is ready, add inference behind a dedicated function such as "
        "`predict(image)`. It should return real model outputs only after input preprocessing, "
        "model loading, and evaluation are documented."
    )

st.markdown("---")
st.caption(
    "NeuroAlert AI · Student research prototype · Not a medical device · "
    "Not for diagnosis, treatment decisions, or emergency use."
)
