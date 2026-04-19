import os, sys, pickle, logging, urllib.parse
import multiprocessing
multiprocessing.freeze_support()

from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from feature_extractor import (
    extract_features_batch, FEATURE_NAMES,
    phishing_risk_score, is_trusted_domain, uses_https,
    tld_is_legit, has_ip_address,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

#  Load Model 
MODEL_PATH = os.path.join(BASE_DIR, "model_final.pkl")
try:
    with open(MODEL_PATH, "rb") as fh:
        _bundle = pickle.load(fh)
    _model      = _bundle["model"]
    _scaler     = _bundle["scaler"]
    _feat_names = _bundle["feature_names"]   # only unbiased features
    _model_name = _bundle.get("model_name", "Unknown")
    _test_acc   = _bundle.get("test_accuracy", 0.0)
    logger.info(f"Model loaded: {_model_name}  accuracy={_test_acc*100:.2f}%")
    logger.info(f"Features used: {len(_feat_names)}")
except FileNotFoundError:
    logger.error("model_final.pkl nahi mila! Pehle train_model.py chalao.")
    sys.exit(1)


# HARD OVERRIDE RULES
# Model se pehle yeh rules check hoti hain
def apply_override_rules(url: str, raw_features: dict):
    """
    Returns:
      "LEGITIMATE"  - agar rules clearly legit bol rahe hain
      "PHISHING"    - agar rules clearly phishing bol rahe hain
      None          - model decide karega
    """
    risk  = raw_features.get("phishing_risk_score", 0)
    trust = raw_features.get("is_trusted_domain", 0)
    https = raw_features.get("uses_https", 0)
    tld_l = raw_features.get("tld_is_legit", 0)
    ip    = raw_features.get("has_ip_address", 0)

    # Rule 1: Trusted domain → hamesha LEGIT
    if trust == 1:
        return "LEGITIMATE", "trusted_domain_override"

    # Rule 2: Risk score zero → definitely LEGIT
    if risk == 0:
        return "LEGITIMATE", "zero_risk_override"

    # Rule 3: HTTPS + legit TLD + no IP + low risk → LEGIT
    if https == 1 and tld_l == 1 and ip == 0 and risk <= 1:
        return "LEGITIMATE", "low_risk_https_override"

    # Rule 4: IP address URL → hamesha PHISHING (except localhost)
    if ip == 1:
        return "PHISHING", "ip_address_override"

    # Rule 5: Very high risk score → hamesha PHISHING
    if risk >= 6:
        return "PHISHING", "high_risk_override"

    # Model decide karega
    return None, None


def _predict_single(url: str) -> dict:
    import pandas as pd

    url = url.strip()

    # Step 1: Extract ALL features (for override rules + display)
    all_feats = extract_features_batch([url]).iloc[0].to_dict()

    # Step 2: Apply override rules FIRST
    override_result, override_reason = apply_override_rules(url, all_feats)

    if override_result is not None:
        label = 1 if override_result == "PHISHING" else 0
        return {
            "prediction":     override_result,
            "label":          label,
            "confidence":     1.0,
            "confidence_pct": "100.0% (rule-based)",
            "decided_by":     f"rule: {override_reason}",
            "risk_score":     int(all_feats.get("phishing_risk_score", 0)),
            "features":       all_feats,
        }

    # Step 3: Use model for ambiguous cases
    feat_df = pd.DataFrame([all_feats])[_feat_names]
    feat_sc = _scaler.transform(feat_df)
    label   = int(_model.predict(feat_sc)[0])
    proba   = (
        _model.predict_proba(feat_sc)[0].tolist()
        if hasattr(_model, "predict_proba") else None
    )
    conf = max(proba) if proba else None

    return {
        "prediction":     "PHISHING" if label == 1 else "LEGITIMATE",
        "label":          label,
        "confidence":     round(conf, 4) if conf else None,
        "confidence_pct": f"{conf*100:.1f}%" if conf else "N/A",
        "decided_by":     f"model: {_model_name}",
        "risk_score":     int(all_feats.get("phishing_risk_score", 0)),
        "features":       all_feats,
    }


#  Flask App 
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "model_name":    _model_name,
        "test_accuracy": f"{_test_acc*100:.2f}%",
        "features":      len(_feat_names),
        "override_rules": "active",
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url  = data.get("url") or request.form.get("url", "").strip()

    if not url:
        return jsonify({"error": "URL do"}), 400

    try:
        result = _predict_single(url)
        logger.info(
            f"[{result['prediction']:10s}] risk={result['risk_score']}  "
            f"by={result['decided_by']}  url={url[:60]}"
        )
        return jsonify({"url": url, **result})
    except Exception as e:
        logger.exception(f"Error: {url}")
        return jsonify({"error": str(e)}), 500


@app.route("/batch", methods=["POST"])
def batch_predict():
    data = request.get_json(silent=True) or {}
    urls = data.get("urls", [])
    if not urls or not isinstance(urls, list):
        return jsonify({"error": "JSON mein 'urls' list do"}), 400
    if len(urls) > 500:
        return jsonify({"error": "Max 500 URLs ek baar"}), 400

    results = []
    for url in urls:
        try:
            r = _predict_single(str(url))
            results.append({"url": url, **r})
        except Exception as e:
            results.append({"url": url, "error": str(e)})

    return jsonify({"count": len(results), "results": results})


if __name__ == "__main__":
    # Windows pe 8080 use karo (5000 pe AirPlay ya conflict hota hai)
    port  = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    logger.info(f"Server chal raha hai: http://localhost:{port}")
    logger.info("Override rules: ACTIVE")
    app.run(host="0.0.0.0", port=port, debug=debug)