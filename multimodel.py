"""
multimodel.py — Multi-disease Model Registry
Train และ Load โมเดลสำหรับทุกโรคใน DISEASES

Usage:
    registry = load_all_models()          # โหลดจากไฟล์ หรือ train ใหม่
    state    = registry["diabetes"]
    prob     = state["model"].predict_proba(X)[0, 1]
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
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

from diseases import DISEASES

warnings.filterwarnings("ignore")

_DIR       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_DIR, "models")

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.02, num_leaves=15,
    min_child_samples=20, feature_fraction=0.7,
    bagging_fraction=0.7, bagging_freq=5,
    random_state=42, verbose=-1,
)


# ── Single disease training ────────────────────────────────────
def train_disease(disease_key: str) -> dict:
    cfg = DISEASES[disease_key]
    df  = pd.read_csv(cfg["csv"])

    features = cfg["features"]
    target   = cfg["target"]
    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    for d in [X_train, X_test, y_train, y_test]:
        d.reset_index(drop=True, inplace=True)

    # Zero-imputation (diabetes only)
    fill_vals = {}
    for col in cfg.get("cols_to_fix", []):
        medians = []
        for cls in [0, 1]:
            mask = (X_train[col] != 0) & (y_train == cls)
            med  = X_train.loc[mask, col].median()
            medians.append(med)
            X_train.loc[(X_train[col] == 0) & (y_train == cls), col] = med
        fill_vals[col] = float(np.mean(medians))
    for col in cfg.get("cols_to_fix", []):
        X_test[col] = X_test[col].replace(0, np.nan).fillna(fill_vals[col])

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # SMOTE ถ้า minority < 40%
    ratio = (y_train == 1).sum() / len(y_train)
    if ratio < 0.4:
        sm = SMOTE(random_state=42)
        X_train_s, y_arr = sm.fit_resample(X_train_s, y_train.values)
        y_train = pd.Series(y_arr)

    scale_w = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    lgbm = lgb.LGBMClassifier(scale_pos_weight=scale_w, **LGBM_PARAMS)
    lgbm.fit(X_train_s, y_train.values if hasattr(y_train, "values") else y_train)

    calibrated = CalibratedClassifierCV(lgbm, method="sigmoid", cv="prefit")
    calibrated.fit(X_test_s, y_test.values)

    probs = calibrated.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "auc":    round(float(roc_auc_score(y_test, probs)), 4),
        "f1":     round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
    }

    return {
        "model":        calibrated,
        "scaler":       scaler,
        "fill_vals":    fill_vals,
        "features":     features,
        "disease_key":  disease_key,
        "trained_at":   datetime.now().strftime("%Y%m%d_%H%M%S"),
        "metrics":      metrics,
        "n_samples":    len(df),
    }


def save_disease_model(disease_key: str, state: dict) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"model_{disease_key}.pkl")
    joblib.dump(state, path)
    return path


def load_disease_model(disease_key: str) -> dict | None:
    path = os.path.join(MODELS_DIR, f"model_{disease_key}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


# ── Load or train all diseases ─────────────────────────────────
def load_all_models(force_retrain: bool = False) -> dict[str, dict]:
    """
    คืน dict: {disease_key: model_state}
    โหลดจากไฟล์ถ้ามี / train ใหม่ถ้ายังไม่มีหรือ force_retrain=True
    """
    registry: dict[str, dict] = {}

    for key in DISEASES:
        if not force_retrain:
            state = load_disease_model(key)
            if state:
                print(f"  ✅ {key}: loaded from file  (AUC={state['metrics']['auc']})")
                registry[key] = state
                continue

        print(f"  ⏳ {key}: training...")
        state = train_disease(key)
        save_disease_model(key, state)
        print(f"  ✅ {key}: trained  "
              f"AUC={state['metrics']['auc']}  "
              f"Recall={state['metrics']['recall']}")
        registry[key] = state

    return registry


# ── Preprocess single row ──────────────────────────────────────
def preprocess_input(disease_key: str, raw: dict, state: dict) -> np.ndarray:
    features  = state["features"]
    fill_vals = state["fill_vals"]
    cfg       = DISEASES[disease_key]

    df_in = pd.DataFrame([{f: raw.get(f, 0) for f in features}], columns=features)
    for col in cfg.get("cols_to_fix", []):
        df_in[col] = df_in[col].replace(0, np.nan).fillna(fill_vals.get(col, 0))
    return state["scaler"].transform(df_in)


# ── Risk level ────────────────────────────────────────────────
def risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "สูง"
    elif prob >= 0.4:
        return "กลาง"
    return "ต่ำ"


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Training all disease models...")
    reg = load_all_models(force_retrain=True)
    print(f"\nDone — {len(reg)} models ready")
    for k, s in reg.items():
        print(f"  {k:12s}  AUC={s['metrics']['auc']}  "
              f"F1={s['metrics']['f1']}  Recall={s['metrics']['recall']}")
