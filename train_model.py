import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

    import os, sys, time, pickle, warnings
    warnings.filterwarnings("ignore")

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import resample
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report
    )
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, BASE_DIR)
    from feature_extractor import extract_features_batch, FEATURE_NAMES

    # CRITICAL: Yeh features training mein USE NAHI honge
    # Kyunki ye dataset-biased hain — model galat pattern seekhta tha
    BIASED_FEATURES_TO_REMOVE = [
        "url_length",           
        "url_is_very_long",     # same problem
        "slash_ratio",          
        "path_slash_count",     # same — ecommerce sites mein zyada hote hain
        "path_entropy",         # path structure biased tha dataset mein
        "num_query_params",     # ecommerce mein zyada params hote hain
        "excessive_query_params", # same
        "weak_keyword_count",   # "login" akela kuch nahi batata
    ]

    # Sirf yeh features use honge — domain-based, reliable signals
    TRAINING_FEATURES = [f for f in FEATURE_NAMES if f not in BIASED_FEATURES_TO_REMOVE]

    print("\n" + "="*60)
    print("BIAS-FIXED PHISHING DETECTOR — Training")
    print("="*60)
    print(f"\nTotal features available : {len(FEATURE_NAMES)}")
    print(f"Biased features removed  : {len(BIASED_FEATURES_TO_REMOVE)}")
    print(f"Features used in training: {len(TRAINING_FEATURES)}")
    print("\nFeatures used:")
    for f in TRAINING_FEATURES:
        print(f"  + {f}")
    print("\nFeatures REMOVED (dataset-biased):")
    for f in BIASED_FEATURES_TO_REMOVE:
        print(f"  - {f}")

    # STEP 1 — Load Dataset
    print("\n" + "="*60)
    print("STEP 1: Loading Dataset")
    print("="*60)

    CSV_PATH = os.path.join(BASE_DIR, "StealthPhisher2025.csv")
    if not os.path.exists(CSV_PATH):
        CSV_PATH = "StealthPhisher2025.csv"
    if not os.path.exists(CSV_PATH):
        print("ERROR: StealthPhisher2025.csv nahi mila!")
        input("Enter dabao...")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    URL_COLS   = ["url", "urls", "link", "address"]
    LABEL_COLS = ["label", "type", "class", "target", "phishing", "result", "status"]
    url_col   = next((c for c in URL_COLS   if c in df.columns), df.select_dtypes("object").columns[0])
    label_col = next((c for c in LABEL_COLS if c in df.columns), df.columns[-1])

    print(f"URL column   : '{url_col}'")
    print(f"Label column : '{label_col}'")
    print(f"Label counts : {df[label_col].value_counts().to_dict()}")

    label_series  = df[label_col].astype(str).str.strip().str.lower()
    unique_labels = sorted(label_series.unique())

    if set(unique_labels) <= {"0", "1"}:
        y = label_series.astype(int).values
    elif set(unique_labels) <= {"phishing", "legitimate", "benign", "safe"}:
        y = label_series.isin(["phishing"]).astype(int).values
    else:
        mapping = {v: i for i, v in enumerate(unique_labels)}
        y = label_series.map(mapping).values
        print(f"Mapping: {mapping}")

    print(f"Encoded  ->  0 (legit): {np.sum(y==0):,}  |  1 (phish): {np.sum(y==1):,}")

    MAX_ROWS = 50_000
    if MAX_ROWS and len(df) > MAX_ROWS:
        print(f"\nDataset bada hai ({len(df):,}). Sampling {MAX_ROWS:,}...")
        idx = np.random.RandomState(42).choice(len(df), MAX_ROWS, replace=False)
        df  = df.iloc[idx].reset_index(drop=True)
        y   = y[idx]

    # STEP 2 — Feature Extraction
    print("\n" + "="*60)
    print("STEP 2: Feature Extraction (chunked)")
    print("="*60)

    urls   = df[url_col].astype(str).tolist()
    CHUNK  = 500
    chunks = [urls[i:i+CHUNK] for i in range(0, len(urls), CHUNK)]
    parts  = []
    t0     = time.time()

    for i, chunk in enumerate(chunks):
        parts.append(extract_features_batch(chunk))
        pct     = (i+1) / len(chunks) * 100
        elapsed = time.time() - t0
        eta     = (elapsed / (i+1)) * (len(chunks) - i - 1)
        print(f"  [{i+1}/{len(chunks)}] {pct:5.1f}%  elapsed={elapsed:.0f}s  eta={eta:.0f}s   ",
              end="\r", flush=True)

    X_all = pd.concat(parts, ignore_index=True)
    print(f"\nDone in {time.time()-t0:.1f}s  shape={X_all.shape}")

    # Remove NaN rows
    valid = ~X_all.isnull().any(axis=1)
    X_all = X_all[valid]
    y     = y[valid]
    print(f"After null cleanup: {X_all.shape[0]:,} rows")

    # STEP 3 — Use only unbiased features
    print("\n" + "="*60)
    print("STEP 3: Applying bias filter")
    print("="*60)

    X = X_all[TRAINING_FEATURES]
    print(f"Feature matrix shape: {X.shape}")

    # STEP 4 — Split + Balance + Scale
    print("\n" + "="*60)
    print("STEP 4: Split + Balance + Scale")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    # Oversample minority
    X_tr = pd.DataFrame(X_train, columns=TRAINING_FEATURES)
    X_tr["__lbl__"] = y_train
    maj    = int(pd.Series(y_train).value_counts().idxmax())
    mn     = 1 - maj
    df_maj = X_tr[X_tr["__lbl__"] == maj]
    df_min = X_tr[X_tr["__lbl__"] == mn]
    ratio  = len(df_maj) / max(len(df_min), 1)
    print(f"Imbalance ratio: {ratio:.2f}")

    if ratio > 1.3:
        df_min_up   = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
        df_bal      = pd.concat([df_maj, df_min_up]).sample(frac=1, random_state=42)
        X_train_bal = df_bal.drop("__lbl__", axis=1).values
        y_train_bal = df_bal["__lbl__"].values
        print(f"After oversampling: {len(X_train_bal):,} samples")
    else:
        X_train_bal = X_train.values
        y_train_bal = y_train
        print("Balanced already.")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_bal)
    X_test_sc  = scaler.transform(X_test.values)
    print("Scaling done.")

    # STEP 5 — Train Models
    print("\n" + "="*60)
    print("STEP 5: Training Models")
    print("="*60)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=15,           # shallower = less overfitting
            min_samples_split=10,   # needs more samples to split = generalize better
            min_samples_leaf=5,     # leaf mein minimum 5 samples
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,     # slow learning = less overfitting
            max_depth=5,            # shallow trees
            min_samples_leaf=30,    # conservative
            l2_regularization=2.0,  # strong regularization
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = {}
    CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, clf in models.items():
        print(f"\n  [{name}] Training...")
        t0 = time.time()
        clf.fit(X_train_sc, y_train_bal)
        elapsed = time.time() - t0
        print(f"  [{name}] Done in {elapsed:.1f}s")

        cv_scores = cross_val_score(clf, X_train_sc, y_train_bal,
                                    cv=CV, scoring="f1", n_jobs=1)
        y_pred = clf.predict(X_test_sc)

        results[name] = {
            "model":      clf,
            "y_pred":     y_pred,
            "cv_f1":      cv_scores.mean(),
            "acc":        accuracy_score(y_test, y_pred),
            "precision":  precision_score(y_test, y_pred, zero_division=0),
            "recall":     recall_score(y_test, y_pred, zero_division=0),
            "f1":         f1_score(y_test, y_pred, zero_division=0),
            "train_time": elapsed,
        }
        r = results[name]
        print(f"    CV F1       : {r['cv_f1']:.4f} +/- {cv_scores.std():.4f}")
        print(f"    Test Acc    : {r['acc']*100:.2f}%")
        print(f"    Precision   : {r['precision']:.4f}")
        print(f"    Recall      : {r['recall']:.4f}")
        print(f"    F1-Score    : {r['f1']:.4f}")

    # STEP 6 — Compare + Pick Best
    print("\n" + "="*60)
    print("STEP 6: Comparison")
    print("="*60)

    print(f"\n{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-"*60)
    for name, r in results.items():
        print(f"{name:<25} {r['acc']*100:>7.2f}% {r['precision']:>8.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f}")

    best_name   = max(results, key=lambda n: results[n]["acc"])
    best_result = results[best_name]
    print(f"\nBest: {best_name}  ({best_result['acc']*100:.2f}%)")

    print(f"\nClassification Report -- {best_name}")
    print(classification_report(
        y_test, best_result["y_pred"],
        target_names=["Legitimate", "Phishing"]
    ))

    cm = confusion_matrix(y_test, best_result["y_pred"])
    print("Confusion Matrix:")
    print(f"                 Pred:Legit  Pred:Phish")
    print(f"  Actual:Legit   {cm[0][0]:>8}   {cm[0][1]:>8}   <- False positives (legit→phish)")
    print(f"  Actual:Phish   {cm[1][0]:>8}   {cm[1][1]:>8}")

    fp = cm[0][1]
    total_legit = cm[0][0] + cm[0][1]
    print(f"\n  False Positive Rate: {fp}/{total_legit} = {fp/max(total_legit,1)*100:.1f}%")
    print(f"  (Legitimate sites wrongly called phishing)")

    # STEP 7 — Save Bundle with Override Rules
    print("\n" + "="*60)
    print("STEP 7: Saving model_final.pkl")
    print("="*60)

    bundle = {
        "model":                 best_result["model"],
        "scaler":                scaler,
        "feature_names":         TRAINING_FEATURES,   # only unbiased features
        "all_feature_names":     FEATURE_NAMES,        # for extraction
        "biased_features":       BIASED_FEATURES_TO_REMOVE,
        "model_name":            best_name,
        "test_accuracy":         best_result["acc"],
        "test_f1":               best_result["f1"],
        "test_precision":        best_result["precision"],
        "test_recall":           best_result["recall"],
    }

    save_path = os.path.join(BASE_DIR, "model_final.pkl")
    with open(save_path, "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)

    size_kb = os.path.getsize(save_path) / 1024
    print(f"\nSaved -> model_final.pkl  ({size_kb:.1f} KB)")
    print(f"  Model     : {best_name}")
    print(f"  Accuracy  : {best_result['acc']*100:.2f}%")
    print(f"  Features  : {len(TRAINING_FEATURES)} (biased ones removed)")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nNext:")
    print("  python server.py")
    print("  streamlit run frontend_app.py")
    print()
    input("Enter dabao band karne ke liye...")