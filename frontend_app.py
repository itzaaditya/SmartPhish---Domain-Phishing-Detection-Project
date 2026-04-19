import os
import sys
import pickle
import requests

import streamlit as st
import pandas as pd

#  Ensure feature_extractor is importable 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


# Config
 
SERVER_URL  = os.environ.get("SERVER_URL", "").strip()
MODEL_PATH  = os.path.join(BASE_DIR, "model_final.pkl")

st.set_page_config(
    page_title="SmartPhish — URL Phishing Detector",
    page_icon="🔐",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;400;600;700;800&display=swap');

/* ── Base Reset ─────────────────────────────────────── */
:root {
    --bg:          #070b14;
    --bg-card:     #0d1424;
    --bg-card2:    #111827;
    --border:      #1e2d45;
    --border-glow: #1e3a5f;
    --accent:      #00c9ff;
    --accent2:     #7b61ff;
    --green:       #00e5a0;
    --red:         #ff4560;
    --yellow:      #ffc94d;
    --text:        #c8d8f0;
    --text-muted:  #4a6080;
    --text-dim:    #6b85a8;
    --mono:        'JetBrains Mono', monospace;
    --sans:        'Outfit', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Typography ─────────────────────────────────────── */
h1, h2, h3 {
    font-family: var(--sans) !important;
    font-weight: 700 !important;
    color: #e8f0ff !important;
}

/* ── Inputs ─────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,201,255,0.08) !important;
}
.stTextInput > label,
.stTextArea > label {
    color: var(--text-dim) !important;
    font-size: 0.78rem !important;
    font-family: var(--mono) !important;
    letter-spacing: 0.04em !important;
}

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.2rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(0,201,255,0.25) !important;
}

/* ── Expander ────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-dim) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}

/* ── Metric cards (used in model metrics grid) ───────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 1.2rem 0;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glow);
    border-radius: 10px;
    padding: 1rem 0.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: var(--mono);
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-sub {
    font-size: 0.65rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
    font-family: var(--mono);
}

/* ── Tech badge pills ────────────────────────────────── */
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0.8rem 0;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(0,201,255,0.07);
    border: 1px solid rgba(0,201,255,0.2);
    color: var(--accent);
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.28rem 0.75rem;
    border-radius: 20px;
    letter-spacing: 0.04em;
}
.badge.purple {
    background: rgba(123,97,255,0.08);
    border-color: rgba(123,97,255,0.25);
    color: var(--accent2);
}
.badge.green {
    background: rgba(0,229,160,0.07);
    border-color: rgba(0,229,160,0.22);
    color: var(--green);
}

/* ── Pipeline diagram ────────────────────────────────── */
.pipeline {
    display: flex;
    flex-direction: column;
    gap: 0;
    margin: 1.2rem 0;
}
.pipeline-step {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    position: relative;
}
.pipeline-step.highlight {
    border-color: var(--border-glow);
}
.pipeline-arrow {
    text-align: center;
    color: var(--text-muted);
    font-size: 1.1rem;
    line-height: 1.4;
    margin: 2px 0;
}
.step-title {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.step-body {
    font-size: 0.82rem;
    color: var(--text-dim);
    line-height: 1.5;
}
.step-tag {
    display: inline-block;
    background: rgba(0,229,160,0.08);
    border: 1px solid rgba(0,229,160,0.2);
    color: var(--green);
    font-family: var(--mono);
    font-size: 0.62rem;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    margin: 2px 3px 2px 0;
}
.step-tag.red {
    background: rgba(255,69,96,0.08);
    border-color: rgba(255,69,96,0.2);
    color: var(--red);
}

/* ── Section divider ─────────────────────────────────── */
.section-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2rem 0 1rem;
}
.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}
.section-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* ── Team card ───────────────────────────────────────── */
.team-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 1rem 0;
}
.team-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-family: var(--sans);
}
.team-name {
    font-weight: 600;
    font-size: 0.88rem;
    color: #dce8ff;
}

/* ── Info banner ─────────────────────────────────────── */
.info-banner {
    background: rgba(0,201,255,0.04);
    border: 1px solid rgba(0,201,255,0.15);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-size: 0.85rem;
    color: var(--text-dim);
    line-height: 1.7;
}
.info-banner strong { color: var(--text); }

/* ── Footer ──────────────────────────────────────────── */
.footer-bar {
    margin-top: 2.5rem;
    padding: 1rem 0;
    text-align: center;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    border-top: 1px solid var(--border);
    letter-spacing: 0.08em;
}

/* ── Mobile Responsive ───────────────────────────────── */
@media (max-width: 768px) {
    h1 { font-size: 22px !important; }
    h2 { font-size: 18px !important; }

    .stTextInput > div > div > input { font-size: 13px !important; }
    .stButton > button { font-size: 13px !important; width: 100% !important; }
    textarea { width: 100% !important; }

    .metric-grid { grid-template-columns: repeat(2, 1fr); }
    .team-grid   { grid-template-columns: 1fr; }
    .badge-row   { gap: 6px; }
    .badge       { font-size: 0.65rem; }
}
@media (max-width: 420px) {
    .metric-grid  { grid-template-columns: 1fr 1fr; }
    .metric-value { font-size: 1.1rem; }
}
</style>
""", unsafe_allow_html=True)

# Load model (direct mode)
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as fh:
        bundle = pickle.load(fh)
    return bundle

def local_predict(url: str, bundle: dict) -> dict:
    from feature_extractor import extract_features_batch
    import pandas as pd

    # Saari features extract karo
    all_feats_df = extract_features_batch([url])
    all_feats    = all_feats_df.iloc[0].to_dict()

    trust = all_feats.get("is_trusted_domain", 0)
    ip    = all_feats.get("has_ip_address", 0)

    # Sirf 2 hard rules — baaki sab model decide karega
    if trust == 1:
        return {
            "prediction": "LEGITIMATE", "label": 0,
            "confidence": 1.0, "confidence_pct": "100.0% (rule-based)",
            "decided_by": "rule: trusted_domain", "features": all_feats,
        }
    if ip == 1:
        return {
            "prediction": "PHISHING", "label": 1,
            "confidence": 1.0, "confidence_pct": "100.0% (rule-based)",
            "decided_by": "rule: ip_address", "features": all_feats,
        }

    # Baaki sab → model decide karega
    feat_df = pd.DataFrame([all_feats])[bundle["feature_names"]]
    feat_sc = bundle["scaler"].transform(feat_df)
    model   = bundle["model"]
    label   = int(model.predict(feat_sc)[0])
    proba   = model.predict_proba(feat_sc)[0] if hasattr(model, "predict_proba") else None
    conf    = max(proba) if proba is not None else None
    return {
        "prediction":     "PHISHING" if label == 1 else "LEGITIMATE",
        "label":          label,
        "confidence":     round(conf, 4) if conf else None,
        "confidence_pct": f"{conf*100:.1f}%" if conf else "N/A",
        "decided_by":     f"model: {bundle.get('model_name', 'Unknown')}",
        "features":       all_feats,
    }

def api_predict(url: str) -> dict:
    resp = requests.post(
        f"{SERVER_URL}/predict",
        json={"url": url},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# UI — Header

st.title("🔐 SmartPhish")
st.subheader("ML-powered URL Phishing Detection")
st.markdown("---")

# ── Mode indicator ────────────────────────────────────────────────────────
use_api = bool(SERVER_URL)
if use_api:
    st.info(f"🌐 **API mode** — sending requests to `{SERVER_URL}`")
else:
    try:
        bundle = load_model()
        model_info = (
            f"**{bundle.get('model_name','Model')}** — "
            f"Accuracy: {bundle.get('test_accuracy',0)*100:.2f}%  |  "
            f"F1: {bundle.get('test_f1',0):.4f}"
        )
        st.success(f"✅ **Local mode** — {model_info}")
    except FileNotFoundError:
        st.error(
            " `model_final.pkl` not found. "
            "Run `model.ipynb` to train the model first, "
            "or set `SERVER_URL` to use API mode."
        )
        st.stop()


# Single URL check
st.markdown("###  Check a URL")
url_input = st.text_input(
    "Enter URL to analyse:",
    placeholder="https://example.com/path?query=value",
    help="Include the full URL with http:// or https://",
)

col1, col2 = st.columns(2)
check_btn  = col1.button(" Analyse", use_container_width=True)
clear_btn  = col2.button("🗑️ Clear", use_container_width=True)
if clear_btn:
    st.rerun()

if check_btn and url_input.strip():
    url = url_input.strip()

    with st.spinner("Analysing URL …"):
        try:
            if use_api:
                result = api_predict(url)
            else:
                result = local_predict(url, bundle)

            is_phishing = result["prediction"] == "PHISHING"

            #  Verdict banner 
            if is_phishing:
                st.error(
                    f"🚨 **PHISHING DETECTED**  "
                    f"(confidence: {result.get('confidence_pct','N/A')})"
                )
            else:
                st.success(
                    f"✅ **LEGITIMATE URL**  "
                    f"(confidence: {result.get('confidence_pct','N/A')})"
                )

            #  Confidence bar 
            conf_val = result.get("confidence")
            if conf_val is not None:
                bar_color = "🟥" if is_phishing else "🟩"
                st.markdown(f"**Confidence:** {conf_val*100:.1f}%")
                st.progress(float(conf_val))

            #  Feature breakdown 
            with st.expander("📊 Feature Breakdown", expanded=False):
                feats = result.get("features", {})
                if feats:
                    df_feats = pd.DataFrame(
                        {"Feature": list(feats.keys()), "Value": list(feats.values())}
                    ).set_index("Feature")
                    st.dataframe(df_feats, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error(
                f"Cannot connect to server at `{SERVER_URL}`. "
                "Check that `server.py` is running."
            )
        except Exception as exc:
            st.error(f"Error: {exc}")

elif check_btn:
    st.warning("Please enter a URL first.")

# Batch check section
st.markdown("---")
st.markdown("###  Batch Check (paste multiple URLs)")

batch_text = st.text_area(
    "One URL per line:",
    placeholder="https://google.com\nhttp://192.168.0.1/login\nhttp://free-gift.tk/verify",
)

if st.button(" Analyse All", use_container_width=True):
    urls = [u.strip() for u in batch_text.strip().splitlines() if u.strip()]
    if not urls:
        st.warning("No URLs found.")
    else:
        rows = []
        progress = st.progress(0)
        for i, url in enumerate(urls):
            try:
                if use_api:
                    r = api_predict(url)
                else:
                    r = local_predict(url, bundle)
                rows.append({
                    "URL":        url,
                    "Verdict":    r["prediction"],
                    "Confidence": r.get("confidence_pct", "N/A"),
                })
            except Exception as exc:
                rows.append({"URL": url, "Verdict": "ERROR", "Confidence": str(exc)})
            progress.progress((i + 1) / len(urls))

        df_batch = pd.DataFrame(rows)

        # Colour rows
        def colour_row(row):
            if row["Verdict"] == "PHISHING":
                return ["background-color: #ffcccc"] * len(row)
            elif row["Verdict"] == "LEGITIMATE":
                return ["background-color: #ccffcc"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_batch.style.apply(colour_row, axis=1),
            use_container_width=True,
        )

        phish_count = sum(1 for r in rows if r["Verdict"] == "PHISHING")
        st.markdown(
            f"**Summary:** {len(urls)} URLs checked — "
            f"🚨 {phish_count} phishing  |  ✅ {len(urls)-phish_count} legitimate"
        )

# ABOUT PROJECT
st.markdown("---")
st.markdown("""
<div class='section-head'>
  <span class='section-label'>01 &nbsp;/&nbsp; About the Project</span>
  <div class='section-line'></div>
</div>
""", unsafe_allow_html=True)

st.markdown("##  About SmartPhish")

st.markdown("""
<div class='info-banner'>
<strong>SmartPhish</strong> is an AI-powered phishing URL detection system that protects users from
malicious websites in real time. It combines a <strong>HistGradientBoosting ML model</strong> with a
hard rule-based override engine to classify any URL as <strong>Phishing</strong> or <strong>Legitimate</strong>
within milliseconds — fully offline, no third-party APIs required.<br><br>
The system analyses <strong>22 unbiased structural features</strong> extracted directly from the URL —
including domain entropy, TLD risk level, subdomain patterns, brand impersonation signals,
and a composite phishing risk score — to deliver accurate, explainable predictions.
</div>
""", unsafe_allow_html=True)

#  Tech Stack 
st.markdown("####  Tech Stack")

st.markdown("""
<div class='badge-row'>
  <span class='badge'>Python 3.10+</span>
  <span class='badge purple'> scikit-learn 1.5.2</span>
  <span class='badge purple'> pandas 2.2.3</span>
  <span class='badge purple'> numpy 1.26.4</span>
</div>
<div class='badge-row'>
  <span class='badge green'> Flask 3.0.3</span>
  <span class='badge green'> Streamlit 1.39.0</span>
  <span class='badge green'> requests 2.32.3</span>
  <span class='badge green'> tldextract 5.1.2</span>
</div>
<div class='badge-row'>
  <span class='badge'> HistGradientBoostingClassifier</span>
  <span class='badge'> RandomForestClassifier</span>
  <span class='badge purple'> BeautifulSoup4</span>
  <span class='badge purple'> joblib 1.4.2</span>
</div>
""", unsafe_allow_html=True)

#  Key highlights 
st.markdown("####  Key Highlights")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
-  Instant predictions — runs fully offline
-  22 unbiased URL features per request
-  HistGradientBoosting — F1: **0.9405**
""")
with col_b:
    st.markdown("""
-  Hard override rules for trusted domains
-  Batch analysis for up to 500 URLs
-  Biased features removed to cut false positives
""")

# MODEL METRICS
st.markdown("""
<div class='section-head' style='margin-top:2rem'>
  <span class='section-label'>02 &nbsp;/&nbsp; Model Performance</span>
  <div class='section-line'></div>
</div>
""", unsafe_allow_html=True)

st.markdown("##  Model Metrics")
st.markdown("<p style='color:var(--text-dim); font-size:0.85rem; margin-bottom:1rem;'>Evaluated on held-out test set &nbsp;·&nbsp; HistGradientBoostingClassifier &nbsp;·&nbsp; 22 features</p>", unsafe_allow_html=True)

st.markdown("""
<div class='metric-grid'>
  <div class='metric-card'>
    <div class='metric-label'>Test Accuracy</div>
    <div class='metric-value'>94.04%</div>
    <div class='metric-sub'>Overall correctness</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>Precision</div>
    <div class='metric-value'>0.9827</div>
    <div class='metric-sub'>Low false positives</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>Recall</div>
    <div class='metric-value'>0.9017</div>
    <div class='metric-sub'>Phishing catch rate</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>F1-Score</div>
    <div class='metric-value'>0.9405</div>
    <div class='metric-sub'>Precision–Recall balance</div>
  </div>
</div>
""", unsafe_allow_html=True)

# PROJECT WORKFLOW
st.markdown("""
<div class='section-head' style='margin-top:2rem'>
  <span class='section-label'>03 &nbsp;/&nbsp; Project Workflow</span>
  <div class='section-line'></div>
</div>
""", unsafe_allow_html=True)

st.markdown("##  How It Works")
st.markdown("<p style='color:var(--text-dim); font-size:0.85rem;'>SmartPhish uses a two-stage pipeline — fast rule-based filtering followed by ML inference for ambiguous cases.</p>", unsafe_allow_html=True)

st.markdown("""
<div class='pipeline'>

  <div class='pipeline-step highlight'>
    <div class='step-title'>① URL Submitted</div>
    <div class='step-body'>User pastes a URL into the single-check or batch interface.</div>
  </div>

  <div class='pipeline-arrow'>↓</div>

  <div class='pipeline-step highlight'>
    <div class='step-title'>② Feature Extraction &nbsp;·&nbsp; 22 signals computed</div>
    <div class='step-body'>
      URL is parsed without visiting the page. Features computed instantly:
      <br><br>
      <span class='step-tag'>domain entropy</span>
      <span class='step-tag'>TLD classification</span>
      <span class='step-tag'>subdomain count</span>
      <span class='step-tag'>HTTPS usage</span>
      <span class='step-tag'>brand-in-subdomain</span>
      <span class='step-tag'>gibberish detection</span>
      <span class='step-tag'>keyword signals</span>
      <span class='step-tag'>phishing risk score</span>
    </div>
  </div>

  <div class='pipeline-arrow'>↓</div>

  <div class='pipeline-step highlight'>
    <div class='step-title'>③ Hard Override Rules &nbsp;·&nbsp; runs first, always</div>
    <div class='step-body'>
      Clear-cut cases are resolved instantly without the model:<br><br>
      <span class='step-tag'>Trusted domain → LEGIT</span>
      <span class='step-tag'>Risk score = 0 → LEGIT</span>
      <span class='step-tag'>HTTPS + legit TLD + low risk → LEGIT</span>
      <span class='step-tag red'>IP address URL → PHISHING</span>
      <span class='step-tag red'>Risk score ≥ 6 → PHISHING</span>
    </div>
  </div>

  <div class='pipeline-arrow'>↓ &nbsp;<span style='color:var(--text-muted); font-size:0.75rem;'>ambiguous cases only</span></div>

  <div class='pipeline-step highlight'>
    <div class='step-title'>④ ML Model &nbsp;·&nbsp; HistGradientBoostingClassifier</div>
    <div class='step-body'>
      22 unbiased features scaled and passed to the trained model.
      Returns <strong>PHISHING</strong> or <strong>LEGITIMATE</strong> with a confidence probability.
      Biased length/path features are excluded to prevent false positives on legitimate e-commerce URLs.
    </div>
  </div>

  <div class='pipeline-arrow'>↓</div>

  <div class='pipeline-step'>
    <div class='step-title'>⑤ Final Verdict</div>
    <div class='step-body'>Prediction + Confidence % + full Feature Breakdown returned to the UI.</div>
  </div>

</div>
""", unsafe_allow_html=True)

# DEVELOPED BY
st.markdown("""
<div class='section-head' style='margin-top:2rem'>
  <span class='section-label'>04 &nbsp;/&nbsp; Team</span>
  <div class='section-line'></div>
</div>
""", unsafe_allow_html=True)

st.markdown("##  Developed By")

st.markdown("""
<div class='team-grid'>
  <div class='team-card'><div class='team-name'>Priyal Toshniwal</div></div>
  <div class='team-card'><div class='team-name'>Ashish Vats</div></div>
  <div class='team-card'><div class='team-name'>Aaditya Pareek</div></div>
  <div class='team-card'><div class='team-name'>Vinay Soni</div></div>
</div>
<div style='
    background: rgba(0,201,255,0.03);
    border: 1px solid rgba(0,201,255,0.12);
    border-radius: 10px;
    padding: 0.9rem 1.4rem;
    margin-top: 0.5rem;
    text-align: center;
    color: #4a6080;
    font-size: 12.5px;
    font-family: var(--mono);
'>
    Built as part of a cybersecurity research project on ML-based phishing detection.<br>
    Trained on real-world phishing datasets &nbsp;·&nbsp; Rule-based override engine &nbsp;·&nbsp; 22-feature URL analysis
</div>
""", unsafe_allow_html=True)

#  Footer 
st.markdown("""
<div class='footer-bar'>
  SMARTPHISH &nbsp;·&nbsp; Powered by scikit-learn &nbsp;·&nbsp; 22-feature URL analysis &nbsp;·&nbsp; HistGradientBoosting
</div>
""", unsafe_allow_html=True)
