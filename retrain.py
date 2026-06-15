"""
retrain.py — Retrain Pipeline สำหรับ Diabetes Risk Predictor
─────────────────────────────────────────────────────────────
รันด้วย:  python3 retrain.py
หรือเรียกจาก API endpoint: POST /admin/retrain

Logic:
  1. โหลด diabetes.csv (base dataset)
  2. ดึง follow-up ที่ได้รับการยืนยัน actual_outcome จาก DB
  3. Merge → retrain LightGBM + Platt calibration
  4. บันทึก model ลง models/model_v{timestamp}.pkl
  5. อัปเดต models/model_current.pkl (symlink หรือ copy)
"""

import os
import shutil
import joblib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score, brier_score_loss,
)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

from database import init_db, get_confirmed_followups, log_action

warnings.filterwarnings("ignore")

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigree", "Age",
]
COLS_TO_FIX = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")

BEST_PARAMS = dict(
    n_estimators=340, learning_rate=0.019, num_leaves=10,
    min_child_samples=22, feature_fraction=0.60,
    bagging_fraction=0.69, bagging_freq=6,
    random_state=42, verbose=-1,
)
MIN_FOLLOWUP_SAMPLES = 5   # ต้องมี follow-up อย่างน้อย N ราย จึงจะ retrain


def load_base_data() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
    df = pd.read_csv(csv_path)
    df["source"] = "original"
    return df


def merge_datasets(base_df: pd.DataFrame, followup_df: pd.DataFrame) -> pd.DataFrame:
    if followup_df.empty:
        return base_df

    # follow-up data มี weight สูงกว่า (ข้อมูลใหม่จากโรงพยาบาลนี้โดยตรง)
    # duplicate follow-up 3x เพื่อให้โมเดลให้น้ำหนักมากขึ้น
    followup_weighted = pd.concat([followup_df] * 3, ignore_index=True)
    combined = pd.concat([base_df, followup_weighted], ignore_index=True)
    return combined


def preprocess_for_training(df: pd.DataFrame):
    X = df[FEATURE_NAMES].copy()
    y = df["Outcome"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    for d in [X_train, X_test, y_train, y_test]:
        d.reset_index(drop=True, inplace=True)

    # Impute zeros ด้วย per-class median
    fill_vals = {}
    for col in COLS_TO_FIX:
        medians = []
        for cls in [0, 1]:
            mask = (X_train[col] != 0) & (y_train == cls)
            med  = X_train.loc[mask, col].median()
            medians.append(med)
            X_train.loc[(X_train[col] == 0) & (y_train == cls), col] = med
        fill_vals[col] = np.mean(medians)
    for col in COLS_TO_FIX:
        X_test[col] = X_test[col].replace(0, np.nan).fillna(fill_vals[col])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # SMOTE ถ้า class imbalance (ratio < 0.6)
    ratio = (y_train == 1).sum() / len(y_train)
    if ratio < 0.4:
        sm = SMOTE(random_state=42)
        X_train_s, y_train_arr = sm.fit_resample(X_train_s, y_train.values)
        y_train = pd.Series(y_train_arr)
        print(f"  SMOTE applied: {len(y_train)} samples after oversampling")

    return X_train_s, X_test_s, y_train, y_test, scaler, fill_vals


def train_and_evaluate(X_train_s, X_test_s, y_train, y_test, fill_vals):
    scale_w = (y_train == 0).sum() / (y_train == 1).sum()
    lgbm = lgb.LGBMClassifier(scale_pos_weight=scale_w, **BEST_PARAMS)
    lgbm.fit(X_train_s, y_train.values if hasattr(y_train, "values") else y_train)

    # Platt scaling
    calibrated = CalibratedClassifierCV(lgbm, method="sigmoid", cv="prefit")
    calibrated.fit(X_test_s, y_test.values)

    # Evaluate
    probs = calibrated.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "auc":       round(float(roc_auc_score(y_test, probs)), 4),
        "f1":        round(float(f1_score(y_test, preds)), 4),
        "recall":    round(float(recall_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds)), 4),
        "brier":     round(float(brier_score_loss(y_test, probs)), 4),
    }
    return calibrated, metrics


def save_model(model, scaler, fill_vals, metrics, n_original, n_followup):
    os.makedirs(MODELS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "model":       model,
        "scaler":      scaler,
        "fill_vals":   fill_vals,
        "trained_at":  ts,
        "n_original":  n_original,
        "n_followup":  n_followup,
        "metrics":     metrics,
        "feature_names": FEATURE_NAMES,
    }

    versioned_path = os.path.join(MODELS_DIR, f"model_v{ts}.pkl")
    current_path   = os.path.join(MODELS_DIR, "model_current.pkl")

    joblib.dump(payload, versioned_path)
    shutil.copy2(versioned_path, current_path)

    return versioned_path, ts


def list_model_versions() -> list[dict]:
    if not os.path.exists(MODELS_DIR):
        return []
    versions = []
    for fname in sorted(os.listdir(MODELS_DIR), reverse=True):
        if fname.startswith("model_v") and fname.endswith(".pkl"):
            fpath = os.path.join(MODELS_DIR, fname)
            try:
                payload = joblib.load(fpath)
                versions.append({
                    "filename":   fname,
                    "trained_at": payload.get("trained_at"),
                    "n_original": payload.get("n_original"),
                    "n_followup": payload.get("n_followup"),
                    "metrics":    payload.get("metrics"),
                })
            except Exception:
                versions.append({"filename": fname, "error": "cannot load"})
    return versions


def run_retrain(triggered_by: str = "manual") -> dict:
    """
    Main retrain function — คืน dict ของผลลัพธ์
    เรียกได้จากทั้ง CLI และ API endpoint
    """
    init_db()
    print(f"\n{'='*55}")
    print(f"  Retrain Pipeline  |  triggered by: {triggered_by}")
    print(f"{'='*55}")

    # 1. โหลดข้อมูล
    base_df     = load_base_data()
    followup_df = get_confirmed_followups()
    n_original  = len(base_df)
    n_followup  = len(followup_df)

    print(f"  Base dataset:    {n_original} samples")
    print(f"  Follow-up data:  {n_followup} confirmed cases")

    if n_followup < MIN_FOLLOWUP_SAMPLES:
        msg = (f"Follow-up data มีเพียง {n_followup} ราย "
               f"(ต้องการอย่างน้อย {MIN_FOLLOWUP_SAMPLES}) — ยังไม่ retrain")
        print(f"\n  ⚠️  {msg}")
        return {"status": "skipped", "reason": msg,
                "n_followup": n_followup, "min_required": MIN_FOLLOWUP_SAMPLES}

    # 2. Merge
    combined_df = merge_datasets(base_df, followup_df)
    print(f"  Combined:        {len(combined_df)} samples (follow-up weighted x3)")

    # 3. Preprocess + train
    print("\n  Preprocessing...")
    X_train_s, X_test_s, y_train, y_test, scaler, fill_vals = preprocess_for_training(combined_df)

    print("  Training LightGBM + Platt calibration...")
    model, metrics = train_and_evaluate(X_train_s, X_test_s, y_train, y_test, fill_vals)

    print(f"\n  📊 Metrics on test set:")
    for k, v in metrics.items():
        print(f"      {k:12s}: {v}")

    # 4. Save
    path, ts = save_model(model, scaler, fill_vals, metrics, n_original, n_followup)
    print(f"\n  ✅ Saved: {os.path.basename(path)}")
    print(f"  ✅ Updated: models/model_current.pkl")

    # 5. Log to audit
    log_action(
        actor=triggered_by, actor_type="system", action="retrain",
        detail=(f"n_original={n_original}, n_followup={n_followup}, "
                f"auc={metrics['auc']}, recall={metrics['recall']}"),
    )

    return {
        "status":      "success",
        "trained_at":  ts,
        "n_original":  n_original,
        "n_followup":  n_followup,
        "metrics":     metrics,
        "model_file":  os.path.basename(path),
    }


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_retrain(triggered_by="cli")
    if result["status"] == "success":
        print(f"\n{'='*55}")
        print("  Retrain เสร็จแล้ว!")
        print(f"  AUC    : {result['metrics']['auc']}")
        print(f"  Recall : {result['metrics']['recall']}")
        print(f"  F1     : {result['metrics']['f1']}")
        print(f"{'='*55}\n")
