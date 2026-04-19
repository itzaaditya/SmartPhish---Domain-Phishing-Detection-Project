# 🔐 SmartPhish — AI-Powered Phishing URL Detector

SmartPhish is a machine learning system that classifies any URL as **Phishing** or **Legitimate** in milliseconds — fully offline, no third-party APIs required. It combines a trained **HistGradientBoostingClassifier** with a hard rule-based override engine for high accuracy and low false positives.

> Built as part of a cybersecurity research project on ML-based phishing detection.  
> **Team:** Priyal Toshniwal · Ashish Vats · Aaditya Pareek · Vinay Soni

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | **94.04%** |
| Precision | **0.9827** |
| Recall | **0.9017** |
| F1-Score | **0.9405** |

Evaluated on a held-out test set · HistGradientBoostingClassifier · 22 unbiased features

---

## 🗂️ Project Structure

```
SmartPhish/
│
├── feature_extractor.py    # URL parsing & 30 feature functions
├── train_model.py          # Model training pipeline
├── compare_models.py       # Model comparison utilities
├── server.py               # Flask REST API (port 8080)
├── frontend_app.py         # Streamlit web UI
│
├── StealthPhisher2025.csv  # Training dataset
├── model_final.pkl         # Trained model bundle (generated)
└── requirements.txt        # Python dependencies
```

---

## ⚙️ How It Works

SmartPhish uses a **two-stage pipeline** — fast rule-based filtering first, ML inference for ambiguous cases.

```
URL Input
    ↓
Feature Extraction  (22 signals computed from URL structure — no page visit needed)
    ↓
Hard Override Rules (clear-cut cases resolved instantly)
    ↓  (ambiguous cases only)
ML Model            (HistGradientBoostingClassifier)
    ↓
Final Verdict       (PHISHING / LEGITIMATE + confidence %)
```

### Override Rules (run before the model)

| Rule | Verdict |
|---|---|
| Domain is in the trusted whitelist | ✅ LEGITIMATE |
| Composite risk score = 0 | ✅ LEGITIMATE |
| HTTPS + legit TLD + no IP + risk ≤ 1 | ✅ LEGITIMATE |
| URL contains a bare IP address | 🚨 PHISHING |
| Composite risk score ≥ 6 | 🚨 PHISHING |

---

## 🧠 Features

30 structural features are extracted per URL; **22 unbiased features** are used for training (8 length/path features are excluded to eliminate false positives on legitimate e-commerce URLs).

### Features Used in Training

| Category | Features |
|---|---|
| Domain | `is_trusted_domain`, `domain_has_hyphen`, `domain_has_digits`, `domain_length`, `domain_is_long`, `domain_entropy`, `domain_is_random` |
| Subdomain | `subdomain_count`, `excessive_subdomains`, `brand_in_subdomain`, `keyword_in_domain` |
| TLD | `tld_is_risky`, `tld_is_legit` |
| Protocol | `uses_https`, `has_ip_address`, `has_at_symbol`, `has_double_slash_redirect`, `has_port` |
| Content | `digit_ratio`, `special_char_ratio`, `has_phishing_pattern`, `phishing_risk_score` |

### Removed (Dataset-Biased) Features

`url_length`, `url_is_very_long`, `slash_ratio`, `path_slash_count`, `path_entropy`, `num_query_params`, `excessive_query_params`, `weak_keyword_count`

> These features caused false positives on legitimate e-commerce URLs (e.g. Amazon, Flipkart deep links).

---

## 🚀 Setup & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This reads `StealthPhisher2025.csv`, extracts features, trains both RandomForest and HistGradientBoosting, picks the best model, and saves it as `model_final.pkl`.

### 3. Start the API Server

```bash
python server.py
```

Server runs at `http://localhost:8080`. Set `PORT` environment variable to change the port.

### 4. Launch the Web UI

```bash
streamlit run frontend_app.py
```

The UI auto-connects to the local server. Set the `SERVER_URL` environment variable to point to a remote server.

---

## 🌐 API Reference

### Health Check

```
GET /health
```

```json
{
  "status": "ok",
  "model_name": "HistGradientBoosting",
  "test_accuracy": "94.04%",
  "features": 22,
  "override_rules": "active"
}
```

### Single URL Prediction

```
POST /predict
Content-Type: application/json

{ "url": "http://paypal-secure-login.xyz/verify?account=123" }
```

```json
{
  "url": "http://paypal-secure-login.xyz/verify?account=123",
  "prediction": "PHISHING",
  "label": 1,
  "confidence": 0.9821,
  "confidence_pct": "98.2%",
  "decided_by": "rule: high_risk_override",
  "risk_score": 8,
  "features": { ... }
}
```

### Batch Prediction (up to 500 URLs)

```
POST /batch
Content-Type: application/json

{ "urls": ["https://google.com", "http://free-gift.tk/verify", ...] }
```

---

## 🧪 Quick Feature Test

Run the self-test built into `feature_extractor.py`:

```bash
python feature_extractor.py
```

This checks 10 labelled URLs (both legitimate and phishing) and prints a pass/fail report.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn 1.5.2 (HistGradientBoostingClassifier, RandomForestClassifier) |
| Data | pandas 2.2.3 · numpy 1.26.4 |
| API | Flask 3.0.3 · flask-cors 4.0.1 |
| UI | Streamlit 1.39.0 |
| URL Parsing | tldextract 5.1.2 · urllib3 2.2.2 |
| Other | joblib 1.4.2 · beautifulsoup4 4.12.3 · python-dotenv |

---

## 📌 Notes

- The model runs **fully offline** — no external API calls are made during prediction.
- The **trusted domain whitelist** in `feature_extractor.py` covers major global and Indian sites (Google, Amazon, Flipkart, SBI, IRCTC, etc.). Add domains to `TRUSTED_DOMAINS` as needed.
- For production deployments, use `gunicorn` instead of Flask's built-in server:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:8080 server:app
  ```
- The `.pkl` model bundle stores the model, scaler, feature names, and evaluation metrics together for easy portability.
