import joblib
import pandas as pd
import numpy as np
from feature_extractor import URLFeatureExtractor
import warnings
warnings.filterwarnings('ignore')

# Load model
model = joblib.load("model.pkl")
extractor = URLFeatureExtractor(timeout=10)  # Increased timeout

def analyze_url_detailed(url):
    """Analyze URL with detailed feature breakdown"""
    print("\n" + "="*80)
    print(f"Analyzing: {url}")
    print("="*80)
    
    # Extract features
    print("\n⏳ Extracting features...")
    feature_array = extractor.extract_features(url)
    feature_df = pd.DataFrame([feature_array])
    
    # Make prediction
    prediction = model.predict(feature_df)[0]
    proba = model.predict_proba(feature_df)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
    
    # Get class labels
    model_classes = getattr(model, "classes_", ["Legitimate", "Phishing"])
    pred_label = model_classes[int(prediction)] if len(model_classes) > int(prediction) else str(prediction)
    
    print(f"\n🎯 PREDICTION: {pred_label}")
    print(f"📊 Confidence: {max(proba)*100:.2f}%")
    print(f"📈 Probabilities: Legitimate={proba[0]*100:.2f}% | Phishing={proba[1]*100:.2f}%")
    
    # Show feature importance if available
    if hasattr(model, 'feature_importances_'):
        print("\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
        feature_names = [
            "lengthofurl", "urlcomplexity", "charactercomplexity", "domainlengthofurl",
            "isdomainip", "tldlength", "lettercntinurl", "urlletterratio",
            "digitcntinurl", "urldigitratio", "equalcharcntinurl", "quesmarkcntinurl",
            "ampcharcntinurl", "otherspclcharcntinurl", "urlotherspclcharratio",
            "numberofhashtags", "numberofsubdomains", "havingpath", "pathlength",
            "havingquery", "havingfragment", "havinganchor", "hasssl",
            "isunreachable", "lineofcode", "longestlinelength", "hastitle",
            "hasfavicon", "hasrobotsblocked", "isresponsive", "isurlredirects",
            "isselfredirects", "hasdescription", "haspopup", "hasiframe",
            "isformsubmitexternal", "hassocialmediapage", "hassubmitbutton",
            "hashiddenfields", "haspasswordfields", "hasbankingkey", "haspaymentkey",
            "hascryptokey", "hascopyrightinfokey", "cntimages", "cntfilescss",
            "cntfilesjs", "cntselfhref", "cntemptyref", "cntexternalref",
            "cntpopup", "cntiframe", "uniquefeaturecnt", "waplegitimate",
            "wapphishing", "shannonentropy", "fractaldimension", "kolmogorovcomplexity",
            "hexpatterncnt", "base64patterncnt", "likelinessindex"
        ]
        
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        
        for idx in top_indices:
            fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"  {fname:25s}: {importances[idx]:.4f} (value={feature_array[idx]})")
    
    # Show key features
    print("\n📋 KEY FEATURES EXTRACTED:")
    key_features = {
        "URL Length": feature_array[0],
        "Has SSL (HTTPS)": "Yes" if feature_array[22] == 1 else "No",
        "Is Unreachable": "Yes" if feature_array[23] == 1 else "No",
        "Domain is IP": "Yes" if feature_array[4] == 1 else "No",
        "Number of Subdomains": feature_array[16],
        "Has Title": "Yes" if feature_array[26] == 1 else "No",
        "Has Favicon": "Yes" if feature_array[27] == 1 else "No",
        "Is Responsive": "Yes" if feature_array[29] == 1 else "No",
        "Has Description": "Yes" if feature_array[32] == 1 else "No",
        "Lines of Code": feature_array[24],
        "Image Count": feature_array[44],
        "CSS Files": feature_array[45],
        "JS Files": feature_array[46],
        "Shannon Entropy": feature_array[55],
        "WAP Legitimate Prob": feature_array[53],
        "WAP Phishing Prob": feature_array[54],
    }
    
    for key, value in key_features.items():
        print(f"  {key:25s}: {value}")
    
    # Warning flags
    print("\n⚠️  POTENTIAL ISSUES:")
    warnings = []
    
    if feature_array[23] == 1:  # isunreachable
        warnings.append("❌ Website appears unreachable (timeout/error)")
    if feature_array[24] == 0:  # lineofcode
        warnings.append("❌ No HTML content fetched (0 lines of code)")
    if feature_array[26] == 0:  # hastitle
        warnings.append("⚠️  No title tag found")
    if feature_array[27] == 0:  # hasfavicon
        warnings.append("⚠️  No favicon found")
    if feature_array[32] == 0:  # hasdescription
        warnings.append("⚠️  No meta description found")
    
    if warnings:
        for w in warnings:
            print(f"  {w}")
    else:
        print("  ✅ No major issues detected")
    
    print("\n" + "="*80)
    return prediction, proba, feature_array


if __name__ == "__main__":
    print("🔬 URL Feature Debugger")
    print("This tool helps diagnose why legitimate sites are flagged as phishing\n")
    
    # Test known legitimate sites
    test_urls = [
        "https://www.google.com",
        "https://www.myntra.com",
        "https://www.amazon.com",
        "https://github.com",
    ]
    
    print("\n🧪 Testing Known Legitimate Websites:")
    print("If these are flagged as phishing, there's a problem!\n")
    
    results = []
    for url in test_urls:
        try:
            pred, proba, features = analyze_url_detailed(url)
            results.append({
                'url': url,
                'prediction': pred,
                'phishing_prob': proba[1] if len(proba) > 1 else 0
            })
        except Exception as e:
            print(f"\n❌ Error analyzing {url}: {str(e)}\n")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    false_positives = sum(1 for r in results if r['prediction'] == 1)
    
    print(f"\nTotal URLs tested: {len(results)}")
    print(f"Flagged as Phishing: {false_positives}")
    print(f"Flagged as Legitimate: {len(results) - false_positives}")
    
    if false_positives > 0:
        print("\n⚠️  WARNING: Your model has FALSE POSITIVES!")
        print("\nPossible causes:")
        print("1. Feature extraction is timing out (increase timeout)")
        print("2. Model was trained on imbalanced data")
        print("3. Model is overfitted to training data")
        print("4. Robots.txt or anti-bot measures blocking content fetch")
        print("\nSuggested fixes:")
        print("- Increase timeout in URLFeatureExtractor (currently 5s)")
        print("- Retrain model with balanced dataset")
        print("- Use better User-Agent headers")
        print("- Check if training data had similar issues")
    else:
        print("\n✅ All legitimate sites correctly identified!")
    
    print("\n" + "="*80)