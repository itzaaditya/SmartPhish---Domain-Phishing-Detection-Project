# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import pandas as pd
# import numpy as np
# import traceback
# from feature_extractor import URLFeatureExtractor

# app = Flask(__name__)
# CORS(app)

# # Load model
# try:
#     model = joblib.load("model.pkl")
#     print("✅ Model loaded successfully.")
# except Exception as e:
#     print("⚠️ Model not loaded:", e)
#     model = None

# # Initialize feature extractor
# extractor = URLFeatureExtractor()


# def convert_to_python_types(obj):
#     """Convert numpy/pandas types to native Python types for JSON serialization"""
#     if isinstance(obj, (np.integer, np.int64, np.int32)):
#         return int(obj)
#     elif isinstance(obj, (np.floating, np.float64, np.float32)):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, (list, tuple)):
#         return [convert_to_python_types(item) for item in obj]
#     elif isinstance(obj, dict):
#         return {key: convert_to_python_types(value) for key, value in obj.items()}
#     else:
#         return obj


# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()
#         if not data or "url" not in data:
#             return jsonify({"error": "Missing 'url' in request body"}), 400

#         if model is None:
#             return jsonify({"error": "Model not loaded"}), 500

#         url = data["url"]
#         print(f"Analyzing URL: {url}")

#         # Extract features
#         feature_array = extractor.extract_features(url)
#         feature_df = pd.DataFrame([feature_array])

#         # Predict
#         raw_pred = model.predict(feature_df)
#         proba = model.predict_proba(feature_df)[0] if hasattr(model, "predict_proba") else np.array([0.5, 0.5])

#         # Get prediction value (convert from numpy)
#         pred_value = int(raw_pred[0])
        
#         # Map prediction to readable label
#         model_classes = getattr(model, "classes_", None)

#         if model_classes is not None:
#             # Convert model classes to native Python types
#             model_classes = [convert_to_python_types(cls) for cls in model_classes]
            
#             # Map prediction to label
#             if isinstance(pred_value, int) and len(model_classes) == 2:
#                 label = model_classes[pred_value]
#             else:
#                 label = str(pred_value)
#         else:
#             # No classes_ attribute – fallback
#             label = "Phishing" if pred_value == 1 else "Legitimate"

#         # Convert probabilities to native Python floats
#         proba_list = [float(p) for p in proba]
        
#         # Compute confidence (max probability)
#         confidence = float(max(proba_list)) * 100

#         response = {
#             "url": url,
#             "prediction": str(label),
#             "confidence": round(confidence, 2),
#             "probabilities": proba_list,
#             "model_classes": model_classes if model_classes is not None else ["Legitimate", "Phishing"],
#         }

#         print(f"Prediction: {label}, Confidence: {confidence:.2f}%")
#         return jsonify(response), 200

#     except Exception as e:
#         print("Error during prediction:")
#         print(traceback.format_exc())
#         return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# @app.route("/debug_predict", methods=["POST"])
# def debug_predict():
#     try:
#         data = request.get_json()
#         if not data or "url" not in data:
#             return jsonify({"error": "Missing 'url' in request body"}), 400

#         url = data["url"]
#         feature_array = extractor.extract_features(url)
#         feature_df = pd.DataFrame([feature_array])

#         raw_pred = model.predict(feature_df)
#         proba = model.predict_proba(feature_df) if hasattr(model, "predict_proba") else None
#         model_classes = getattr(model, "classes_", None)

#         # Convert all to native Python types
#         response = {
#             "url": url,
#             "features": {f"feature_{i}": convert_to_python_types(v) 
#                         for i, v in enumerate(feature_df.iloc[0].tolist())},
#             "raw_prediction": convert_to_python_types(raw_pred.tolist()),
#             "predict_proba": convert_to_python_types(proba.tolist()) if proba is not None else None,
#             "model_classes": convert_to_python_types(list(model_classes)) if model_classes is not None else None
#         }

#         return jsonify(response)

#     except Exception as e:
#         print(traceback.format_exc())
#         return jsonify({"error": f"Debug failed: {str(e)}"}), 500


# @app.route("/health", methods=["GET"])
# def health():
#     """Health check endpoint"""
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "extractor_loaded": extractor is not None
#     }), 200


# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print("🚀 Starting Phishing Detection API Server")
#     print("="*60)
#     print(f"Model loaded: {'✅' if model else '❌'}")
#     print(f"Feature extractor loaded: {'✅' if extractor else '❌'}")
#     print("="*60 + "\n")
    
#     app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import traceback
from feature_extractor import URLFeatureExtractor

app = Flask(__name__)
CORS(app)

# Load model
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("⚠️ Model not loaded:", e)
    model = None

# Initialize feature extractor with longer timeout
extractor = URLFeatureExtractor(timeout=10)  # Increased from default 5s to 10s


def convert_to_python_types(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    else:
        return obj


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "Missing 'url' in request body"}), 400

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        url = data["url"]
        print(f"Analyzing URL: {url}")

        # Extract features
        feature_array = extractor.extract_features(url)
        feature_df = pd.DataFrame([feature_array])

        # Predict
        raw_pred = model.predict(feature_df)
        proba = model.predict_proba(feature_df)[0] if hasattr(model, "predict_proba") else np.array([0.5, 0.5])

        # Get prediction value (convert from numpy)
        pred_value = int(raw_pred[0])
        
        # Map prediction to readable label
        model_classes = getattr(model, "classes_", None)

        if model_classes is not None:
            # Convert model classes to native Python types
            model_classes = [convert_to_python_types(cls) for cls in model_classes]
            
            # Map prediction to label
            if isinstance(pred_value, int) and len(model_classes) == 2:
                label = model_classes[pred_value]
            else:
                label = str(pred_value)
        else:
            # No classes_ attribute – fallback
            label = "Phishing" if pred_value == 1 else "Legitimate"

        # Convert probabilities to native Python floats
        proba_list = [float(p) for p in proba]
        
        # Compute confidence (max probability)
        confidence = float(max(proba_list)) * 100

        response = {
            "url": url,
            "prediction": str(label),
            "confidence": round(confidence, 2),
            "probabilities": proba_list,
            "model_classes": model_classes if model_classes is not None else ["Legitimate", "Phishing"],
        }

        print(f"Prediction: {label}, Confidence: {confidence:.2f}%")
        return jsonify(response), 200

    except Exception as e:
        print("Error during prediction:")
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/debug_predict", methods=["POST"])
def debug_predict():
    try:
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "Missing 'url' in request body"}), 400

        url = data["url"]
        feature_array = extractor.extract_features(url)
        feature_df = pd.DataFrame([feature_array])

        raw_pred = model.predict(feature_df)
        proba = model.predict_proba(feature_df) if hasattr(model, "predict_proba") else None
        model_classes = getattr(model, "classes_", None)

        # Convert all to native Python types
        response = {
            "url": url,
            "features": {f"feature_{i}": convert_to_python_types(v) 
                        for i, v in enumerate(feature_df.iloc[0].tolist())},
            "raw_prediction": convert_to_python_types(raw_pred.tolist()),
            "predict_proba": convert_to_python_types(proba.tolist()) if proba is not None else None,
            "model_classes": convert_to_python_types(list(model_classes)) if model_classes is not None else None
        }

        return jsonify(response)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": f"Debug failed: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "extractor_loaded": extractor is not None
    }), 200


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Phishing Detection API Server")
    print("="*60)
    print(f"Model loaded: {'✅' if model else '❌'}")
    print(f"Feature extractor loaded: {'✅' if extractor else '❌'}")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)