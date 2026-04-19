"""
compare_models.py  —  Model Comparison Script
==============================================
Sirf comparison ke liye hai — koi bhi production file modify NAHI hogi.
model_final.pkl ya server.py mein koi changes NAHI hote.

Yeh script sirf dikhata hai:
  - Kaun sa model best perform karta hai
  - Accuracy, Precision, Recall, F1, ROC-AUC, Training Time
  - Confusion Matrix for each model

Models compared:
  1. RandomForest          (current production model)
  2. HistGradientBoosting  (current production model)
  3. LogisticRegression    (baseline — fast, interpretable)
  4. SVC (SVM)             (strong on small-medium data)
  5. ExtraTreesClassifier  (faster than RF, sometimes better)
  6. XGBoost / LightGBM    (if installed — best boosting)
  7. KNeighborsClassifier  (distance-based, simple)
  8. MLPClassifier         (neural net baseline)

Run:
    python compare_models.py
"""

import multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()

    import os, sys, time, warnings
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
        f1_score, roc_auc_score, confusion_matrix, classification_report
    )

    # ── Core Models ──────────────────────────────────────────────────────
    from sklearn.ensemble import (
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
    )
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, BASE_DIR)

    # ── Feature extractor import ─────────────────────────────────────────
    try:
        from feature_extractor import extract_features_batch, FEATURE_NAMES
    except ImportError:
        print("ERROR: feature_extractor.py nahi mila!")
        print("  Yeh script usi folder mein rakhna jahan feature_extractor.py hai.")
        input("Enter dabao...")
        sys.exit(1)

    # ── Biased features (same as train_model.py) ─────────────────────────
    BIASED_FEATURES_TO_REMOVE = [
        "url_length",
        "url_is_very_long",
        "slash_ratio",
        "path_slash_count",
        "path_entropy",
        "num_query_params",
        "excessive_query_params",
        "weak_keyword_count",
    ]
    TRAINING_FEATURES = [f for f in FEATURE_NAMES if f not in BIASED_FEATURES_TO_REMOVE]

    print("\n" + "=" * 65)
    print("  PHISHING DETECTOR — MULTI-MODEL COMPARISON")
    print("=" * 65)
    print(f"  Training features : {len(TRAINING_FEATURES)}")
    print(f"  Biased (removed)  : {len(BIASED_FEATURES_TO_REMOVE)}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1 — Dataset Load (same logic as train_model.py)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  STEP 1: Loading Dataset")
    print("=" * 65)

    CSV_PATH = os.path.join(BASE_DIR, "StealthPhisher2025.csv")
    if not os.path.exists(CSV_PATH):
        CSV_PATH = "StealthPhisher2025.csv"
    if not os.path.exists(CSV_PATH):
        print("ERROR: StealthPhisher2025.csv nahi mila!")
        input("Enter dabao...")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    URL_COLS   = ["url", "urls", "link", "address"]
    LABEL_COLS = ["label", "type", "class", "target", "phishing", "result", "status"]
    url_col   = next((c for c in URL_COLS   if c in df.columns), df.select_dtypes("object").columns[0])
    label_col = next((c for c in LABEL_COLS if c in df.columns), df.columns[-1])
    print(f"  URL col: '{url_col}'  |  Label col: '{label_col}'")

    label_series  = df[label_col].astype(str).str.strip().str.lower()
    unique_labels = sorted(label_series.unique())

    if set(unique_labels) <= {"0", "1"}:
        y = label_series.astype(int).values
    elif set(unique_labels) <= {"phishing", "legitimate", "benign", "safe"}:
        y = label_series.isin(["phishing"]).astype(int).values
    else:
        mapping = {v: i for i, v in enumerate(unique_labels)}
        y = label_series.map(mapping).values

    MAX_ROWS = 50_000
    if len(df) > MAX_ROWS:
        print(f"  Sampling {MAX_ROWS:,} from {len(df):,}...")
        idx = np.random.RandomState(42).choice(len(df), MAX_ROWS, replace=False)
        df  = df.iloc[idx].reset_index(drop=True)
        y   = y[idx]

    print(f"  Legit: {np.sum(y==0):,}  |  Phishing: {np.sum(y==1):,}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2 — Feature Extraction
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  STEP 2: Feature Extraction")
    print("=" * 65)

    urls   = df[url_col].astype(str).tolist()
    CHUNK  = 500
    chunks = [urls[i:i+CHUNK] for i in range(0, len(urls), CHUNK)]
    parts  = []
    t0     = time.time()

    for i, chunk in enumerate(chunks):
        parts.append(extract_features_batch(chunk))
        pct     = (i + 1) / len(chunks) * 100
        elapsed = time.time() - t0
        eta     = (elapsed / (i + 1)) * (len(chunks) - i - 1)
        print(f"  [{i+1}/{len(chunks)}] {pct:5.1f}%  elapsed={elapsed:.0f}s  eta={eta:.0f}s   ",
              end="\r", flush=True)

    X_all = pd.concat(parts, ignore_index=True)
    valid = ~X_all.isnull().any(axis=1)
    X_all = X_all[valid]
    y     = y[valid]
    X     = X_all[TRAINING_FEATURES]
    print(f"\n  Done. Shape: {X.shape}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3 — Split + Balance + Scale
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  STEP 3: Split + Balance + Scale")
    print("=" * 65)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Oversample minority (same as train_model.py)
    X_tr = pd.DataFrame(X_train, columns=TRAINING_FEATURES)
    X_tr["__lbl__"] = y_train
    maj    = int(pd.Series(y_train).value_counts().idxmax())
    mn     = 1 - maj
    df_maj = X_tr[X_tr["__lbl__"] == maj]
    df_min = X_tr[X_tr["__lbl__"] == mn]
    ratio  = len(df_maj) / max(len(df_min), 1)

    if ratio > 1.3:
        df_min_up   = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
        df_bal      = pd.concat([df_maj, df_min_up]).sample(frac=1, random_state=42)
        X_train_bal = df_bal.drop("__lbl__", axis=1).values
        y_train_bal = df_bal["__lbl__"].values
    else:
        X_train_bal = X_train.values
        y_train_bal = y_train

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train_bal)
    X_test_sc   = scaler.transform(X_test.values)
    print(f"  Train: {X_train_sc.shape}  |  Test: {X_test_sc.shape}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4 — Feature Selection Analysis (SIRF DISPLAY — koi change nahi)
    # Models same TRAINING_FEATURES use karenge — yeh sirf analysis hai
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  STEP 4: Feature Selection Analysis (display only)")
    print("=" * 65)

    from sklearn.ensemble import RandomForestClassifier as _RF
    from sklearn.feature_selection import (
        SelectKBest, chi2, f_classif,
        RFE, mutual_info_classif,
        VarianceThreshold,
    )
    from sklearn.inspection import permutation_importance

    # ── 4a: Variance Threshold — near-zero variance features dikhao ───────
    print("\n  [4a] Variance Threshold (threshold=0.01)")
    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X_train_sc)
    low_var = [TRAINING_FEATURES[i] for i, s in enumerate(vt.get_support()) if not s]
    kept_vt = [TRAINING_FEATURES[i] for i, s in enumerate(vt.get_support()) if s]
    print(f"  Total features   : {len(TRAINING_FEATURES)}")
    print(f"  Low-variance (dropped would be) : {len(low_var)}")
    if low_var:
        for f in low_var:
            print(f"    - {f}")
    else:
        print("  Sab features ka variance theek hai.")
    print(f"  Kept by VT       : {len(kept_vt)}")

    # ── 4b: Random Forest Feature Importances ────────────────────────────
    print("\n  [4b] RandomForest Feature Importances")
    _rf_fs = _RF(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    _rf_fs.fit(X_train_sc, y_train_bal)
    importances = _rf_fs.feature_importances_
    fi_pairs = sorted(zip(TRAINING_FEATURES, importances), key=lambda x: -x[1])

    print(f"\n  {'Rank':<5} {'Feature':<35} {'Importance':>10}  {'Bar'}")
    print(f"  {'-'*5} {'-'*35} {'-'*10}  {'-'*20}")
    for rank, (feat, imp) in enumerate(fi_pairs, 1):
        bar = "█" * int(imp * 200)
        marker = " ◄ top" if rank <= 5 else (" ← low" if rank > len(fi_pairs) - 3 else "")
        print(f"  {rank:<5} {feat:<35} {imp:>10.4f}  {bar}{marker}")

    # Features with importance < 1%
    low_imp = [f for f, i in fi_pairs if i < 0.01]
    print(f"\n  Features with importance < 1%  ({len(low_imp)} total):")
    for f in low_imp:
        print(f"    - {f}")

    # ── 4c: SelectKBest (ANOVA F-score) ──────────────────────────────────
    print("\n  [4c] SelectKBest — ANOVA F-score (top 10)")
    skb = SelectKBest(f_classif, k=min(10, len(TRAINING_FEATURES)))
    skb.fit(X_train_sc, y_train_bal)
    skb_scores = skb.scores_
    skb_pairs  = sorted(zip(TRAINING_FEATURES, skb_scores), key=lambda x: -x[1])
    print(f"\n  {'Rank':<5} {'Feature':<35} {'F-Score':>10}")
    print(f"  {'-'*5} {'-'*35} {'-'*10}")
    for rank, (feat, score) in enumerate(skb_pairs[:10], 1):
        print(f"  {rank:<5} {feat:<35} {score:>10.2f}")

    # ── 4d: Mutual Information ────────────────────────────────────────────
    print("\n  [4d] Mutual Information — top 10 features")
    mi_scores = mutual_info_classif(X_train_sc, y_train_bal, random_state=42)
    mi_pairs  = sorted(zip(TRAINING_FEATURES, mi_scores), key=lambda x: -x[1])
    print(f"\n  {'Rank':<5} {'Feature':<35} {'MI Score':>10}")
    print(f"  {'-'*5} {'-'*35} {'-'*10}")
    for rank, (feat, score) in enumerate(mi_pairs[:10], 1):
        print(f"  {rank:<5} {feat:<35} {score:>10.4f}")

    # ── 4e: RFE (Recursive Feature Elimination) ───────────────────────────
    print("\n  [4e] RFE — Recursive Feature Elimination (keep top 10)")
    _rf_rfe = _RF(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)
    rfe = RFE(_rf_rfe, n_features_to_select=min(10, len(TRAINING_FEATURES)), step=2)
    rfe.fit(X_train_sc, y_train_bal)
    rfe_selected   = [f for f, s in zip(TRAINING_FEATURES, rfe.support_) if s]
    rfe_eliminated = [f for f, s in zip(TRAINING_FEATURES, rfe.support_) if not s]
    print(f"\n  RFE Selected ({len(rfe_selected)}):")
    for f in rfe_selected:
        print(f"    + {f}")
    print(f"\n  RFE Eliminated ({len(rfe_eliminated)}):")
    for f in rfe_eliminated:
        print(f"    - {f}")

    # ── 4f: Agreement Summary ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FEATURE SELECTION SUMMARY")
    print("=" * 65)
    top5_rf  = set(f for f, _ in fi_pairs[:5])
    top5_skb = set(f for f, _ in skb_pairs[:5])
    top5_mi  = set(f for f, _ in mi_pairs[:5])
    top5_rfe = set(rfe_selected[:5])
    agreed_all  = top5_rf & top5_skb & top5_mi & top5_rfe
    agreed_any3 = set()
    for f in TRAINING_FEATURES:
        count = sum([f in top5_rf, f in top5_skb, f in top5_mi, f in top5_rfe])
        if count >= 3:
            agreed_any3.add(f)

    print(f"\n  Methods used: RF Importance, ANOVA F-score, Mutual Info, RFE")
    print(f"\n  Features in TOP-5 of ALL 4 methods ({len(agreed_all)}):")
    for f in agreed_all:
        print(f"    ★ {f}")
    print(f"\n  Features in TOP-5 of at least 3 methods ({len(agreed_any3)}):")
    for f in sorted(agreed_any3):
        print(f"    ✓ {f}")
    print(f"\n  NOTE: Training features UNCHANGED = {len(TRAINING_FEATURES)}")
    print(f"  Yeh analysis sirf informational hai.")
    print("=" * 65)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5 — Define All Candidate Models
    # ══════════════════════════════════════════════════════════════════════
    MODELS = {


        # ── Already in production ──────────────────────────────────────
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=15, min_samples_split=10,
            min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=5,
            min_samples_leaf=30, l2_regularization=2.0,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, class_weight="balanced", random_state=42,
        ),

        # ── New candidates ─────────────────────────────────────────────
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150, max_depth=15, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5,
            min_samples_leaf=20, subsample=0.8, random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42,
        ),
        "SVM_RBF": SVC(
            C=1.0, kernel="rbf", class_weight="balanced",
            probability=True, random_state=42,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=10,
            class_weight="balanced", random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=9, weights="distance", n_jobs=1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            learning_rate_init=0.001, max_iter=200,
            early_stopping=True, random_state=42,
        ),
    }

    # Optional: XGBoost / LightGBM (agar installed ho)
    try:
        from xgboost import XGBClassifier
        MODELS["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
            eval_metric="logloss", random_state=42, n_jobs=1,
        )
        print("  [+] XGBoost found and added.")
    except ImportError:
        print("  [-] XGBoost not installed (pip install xgboost to include).")

    try:
        from lightgbm import LGBMClassifier
        MODELS["LightGBM"] = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            num_leaves=31, class_weight="balanced",
            random_state=42, n_jobs=1, verbose=-1,
        )
        print("  [+] LightGBM found and added.")
    except ImportError:
        print("  [-] LightGBM not installed (pip install lightgbm to include).")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5 — Train & Evaluate All Models
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print(f"  STEP 6: Training {len(MODELS)} Models")
    print("=" * 65)

    CV      = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = {}

    for name, clf in MODELS.items():
        print(f"\n  [{name}] Training...", flush=True)
        t0 = time.time()
        try:
            clf.fit(X_train_sc, y_train_bal)
            elapsed = time.time() - t0

            y_pred = clf.predict(X_test_sc)
            proba  = (
                clf.predict_proba(X_test_sc)[:, 1]
                if hasattr(clf, "predict_proba") else None
            )

            # Fast CV on 3-fold only for speed
            cv_f1 = cross_val_score(
                clf, X_train_sc, y_train_bal,
                cv=CV, scoring="f1", n_jobs=1
            ).mean()

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            auc  = roc_auc_score(y_test, proba) if proba is not None else None
            cm   = confusion_matrix(y_test, y_pred)
            fp_rate = cm[0][1] / max(cm[0][0] + cm[0][1], 1)

            results[name] = {
                "acc": acc, "prec": prec, "rec": rec,
                "f1": f1, "auc": auc, "cv_f1": cv_f1,
                "fp_rate": fp_rate, "train_time": elapsed,
                "cm": cm, "y_pred": y_pred,
            }

            auc_str = f"{auc:.4f}" if auc else "  N/A  "
            print(f"  [{name}]  Acc={acc*100:.2f}%  F1={f1:.4f}  AUC={auc_str}  FP%={fp_rate*100:.1f}%  time={elapsed:.1f}s")

        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            results[name] = None

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6 — Final Comparison Table
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  STEP 7 — FINAL COMPARISON TABLE")
    print("=" * 90)
    print(f"\n  {'Model':<25} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7} {'FP%':>6} {'Time':>7}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*7}")

    valid_results = {k: v for k, v in results.items() if v is not None}
    sorted_models = sorted(valid_results.items(), key=lambda x: -x[1]["f1"])

    for name, r in sorted_models:
        auc_str  = f"{r['auc']:.4f}" if r["auc"] else "  N/A  "
        marker   = " ◄ BEST F1" if name == sorted_models[0][0] else ""
        print(
            f"  {name:<25} {r['acc']*100:>6.2f}%"
            f"  {r['prec']:>6.4f}  {r['rec']:>6.4f}"
            f"  {r['f1']:>6.4f}  {auc_str:>7}"
            f"  {r['fp_rate']*100:>5.1f}%  {r['train_time']:>6.1f}s"
            f"{marker}"
        )

    # ── Best model confusion matrix ───────────────────────────────────
    best_name, best_r = sorted_models[0]
    print(f"\n{'='*65}")
    print(f"  BEST MODEL: {best_name}  (F1 = {best_r['f1']:.4f})")
    print(f"{'='*65}")
    cm = best_r["cm"]
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, best_r["y_pred"],
        target_names=["Legitimate", "Phishing"],
        digits=4
    ))
    print(f"  Confusion Matrix:")
    print(f"                     Pred:Legit   Pred:Phish")
    print(f"    Actual:Legit     {cm[0][0]:>8}     {cm[0][1]:>8}   ← False Positives")
    print(f"    Actual:Phish     {cm[1][0]:>8}     {cm[1][1]:>8}")
    print(f"\n  False Positive Rate : {cm[0][1]}/{cm[0][0]+cm[0][1]} = {best_r['fp_rate']*100:.2f}%")

    # ── Production model reminder ──────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  NOTE: Yeh sirf comparison hai.")
    print(f"  Production model (model_final.pkl) UNCHANGED hai.")
    print(f"  Agar best model change karna ho → train_model.py mein karo.")
    print(f"{'='*65}\n")

    input("Enter dabao band karne ke liye...")
