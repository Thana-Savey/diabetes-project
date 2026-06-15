"""
setup_data.py — Download & prepare all disease datasets
รัน 1 ครั้ง:  python3 setup_data.py
"""
import os, io, subprocess
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_DIR, "data")
os.makedirs(DATA, exist_ok=True)


def _get(url: str, dest: str, label: str) -> bool:
    print(f"  ↓ {label} ...")
    try:
        result = subprocess.run(
            ["curl", "-fsSL", "--retry", "2", "-o", dest, url],
            capture_output=True, timeout=60
        )
        if result.returncode != 0 or not os.path.exists(dest) or os.path.getsize(dest) < 1000:
            raise RuntimeError(result.stderr.decode()[:200] or "empty file")
        print(f"    ✅ {os.path.basename(dest)}  ({os.path.getsize(dest)//1024} KB)")
        return True
    except Exception as e:
        print(f"    ❌ {e}")
        if os.path.exists(dest): os.unlink(dest)
        return False


# ── 1. Breast Cancer (sklearn built-in, no download) ──────────
def make_breast_cancer():
    out = os.path.join(DATA, "breast_cancer.csv")
    if os.path.exists(out):
        print("  ✅ breast_cancer.csv exists")
        return True
    bc   = load_breast_cancer()
    cols = [f.replace(" ", "_") for f in bc.feature_names]
    df   = pd.DataFrame(bc.data, columns=cols)
    df["target"] = (bc.target == 0).astype(int)   # sklearn 0=malignant → flip: 1=malignant
    df.to_csv(out, index=False)
    print(f"  ✅ breast_cancer.csv  ({len(df)} rows, {int(df['target'].sum())} malignant)")
    return True


# ── 2. Stroke (synthetic based on Kaggle dataset statistics) ──
def make_stroke():
    out = os.path.join(DATA, "stroke.csv")
    if os.path.exists(out):
        print("  ✅ stroke.csv exists")
        return True

    # Try download first (works if SSL OK)
    urls = [
        "https://raw.githubusercontent.com/rashida048/Datasets/master/healthcare-dataset-stroke-data.csv",
        "https://raw.githubusercontent.com/dsrscientist/dataset1/master/stroke.csv",
    ]
    tmp = out + ".tmp"
    for url in urls:
        if not _get(url, tmp, "stroke"):
            continue
        try:
            df = pd.read_csv(tmp)
            df = df.drop(columns=["id"], errors="ignore")
            if "stroke" in df.columns:
                df = df.rename(columns={"stroke": "target"})
            elif "target" not in df.columns:
                os.unlink(tmp); continue
            gender_map = {"female": 0, "male": 1, "other": 2}
            if df["gender"].dtype == object:
                df["gender"] = df["gender"].str.lower().map(gender_map).fillna(0).astype(int)
            if "ever_married" in df.columns and df["ever_married"].dtype == object:
                df["ever_married"] = df["ever_married"].str.lower().map({"no": 0, "yes": 1}).fillna(0).astype(int)
            if "smoking_status" in df.columns and df["smoking_status"].dtype == object:
                df["smoking_status"] = df["smoking_status"].str.lower().map(
                    {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "unknown": 3}
                ).fillna(3).astype(int)
            df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce").fillna(df["bmi"].median())
            keep = ["gender", "age", "hypertension", "heart_disease",
                    "ever_married", "avg_glucose_level", "bmi", "smoking_status", "target"]
            df = df[[c for c in keep if c in df.columns]].dropna()
            df.to_csv(out, index=False); os.unlink(tmp)
            print(f"    ✅ stroke.csv  ({len(df)} rows, {int(df['target'].sum())} stroke cases)")
            return True
        except Exception as e:
            print(f"    ❌ preprocessing: {e}")
            if os.path.exists(tmp): os.unlink(tmp)

    # Fallback: generate synthetic dataset from Kaggle published statistics
    print("  ⏳ generating synthetic stroke dataset (based on Kaggle statistics)...")
    rng = np.random.default_rng(42)
    n = 5110

    age            = np.clip(rng.normal(43.2, 22.6, n), 0.5, 82).round(0)
    gender         = rng.choice([0, 1, 2], n, p=[0.590, 0.408, 0.002])
    hypertension   = (rng.random(n) < (0.065 + 0.005 * (age - 43) / 22)).astype(int).clip(0, 1)
    heart_disease  = (rng.random(n) < (0.030 + 0.003 * (age - 43) / 22)).astype(int).clip(0, 1)
    ever_married   = (age > 25).astype(int) * (rng.random(n) < 0.80).astype(int)
    avg_glucose    = np.clip(rng.lognormal(np.log(100), 0.35, n), 55, 300).round(2)
    bmi            = np.clip(rng.normal(28.9, 7.85, n), 10, 60).round(1)
    smoking_status = rng.choice([0, 1, 2, 3], n, p=[0.362, 0.174, 0.174, 0.290])

    # Stroke probability model (approximate)
    logit = (-7.0
             + 0.065  * age
             + 0.008  * avg_glucose
             + 0.03   * bmi
             + 1.2    * hypertension
             + 1.0    * heart_disease
             + 0.4    * (smoking_status == 2))
    p_stroke = 1 / (1 + np.exp(-logit))
    target = rng.random(n) < p_stroke

    df = pd.DataFrame({
        "gender": gender, "age": age, "hypertension": hypertension,
        "heart_disease": heart_disease, "ever_married": ever_married,
        "avg_glucose_level": avg_glucose, "bmi": bmi,
        "smoking_status": smoking_status, "target": target.astype(int),
    })
    df.to_csv(out, index=False)
    print(f"  ✅ stroke.csv (synthetic, {len(df)} rows, {int(df['target'].sum())} stroke cases)")
    return True


# ── 3. Liver Disease (ILPD — UCI) ─────────────────────────────
def make_liver():
    out = os.path.join(DATA, "liver.csv")
    if os.path.exists(out):
        print("  ✅ liver.csv exists")
        return True

    col_names = ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
                 "Alkaline_Phosphotase", "Alamine_Aminotransferase",
                 "Aspartate_Aminotransferase", "Total_Proteins",
                 "Albumin", "Albumin_and_Globulin_Ratio", "Dataset"]

    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
        "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Indian_Liver_Patient.csv",
    ]
    tmp = out + ".tmp"
    for url in urls:
        if not _get(url, tmp, "liver (ILPD)"):
            continue
        try:
            # ILPD has no header row
            df = pd.read_csv(tmp, header=None, names=col_names)
            df["Gender"] = df["Gender"].str.lower().map({"male": 1, "female": 0, "m": 1, "f": 0}).fillna(0).astype(int)
            df["Albumin_and_Globulin_Ratio"] = pd.to_numeric(
                df["Albumin_and_Globulin_Ratio"], errors="coerce"
            ).fillna(df["Albumin_and_Globulin_Ratio"].median())
            df["target"] = (df["Dataset"] == 1).astype(int)   # 1=liver patient, 2=healthy
            df = df.drop(columns=["Dataset"]).dropna()
            df.to_csv(out, index=False)
            os.unlink(tmp)
            print(f"    ✅ liver.csv  ({len(df)} rows, {int(df['target'].sum())} liver patients)")
            return True
        except Exception as e:
            print(f"    ❌ preprocessing: {e}")
            if os.path.exists(tmp): os.unlink(tmp)
    return False


# ── 4. Chronic Kidney Disease (via ucimlrepo id=336) ──────────
def make_ckd():
    out = os.path.join(DATA, "ckd.csv")
    if os.path.exists(out):
        print("  ✅ ckd.csv exists")
        return True

    try:
        from ucimlrepo import fetch_ucirepo
        print("  ↓ CKD from UCI ML Repo (id=336)...")
        ds  = fetch_ucirepo(id=336)
        X   = ds.data.features.copy()
        y   = ds.data.targets.copy()

        num_cols = [c for c in ["age","bp","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"] if c in X.columns]
        bin_cols = [c for c in ["htn","dm","cad","ane"] if c in X.columns]

        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())

        for col in bin_cols:
            X[col] = X[col].astype(str).str.strip().str.lower().map(
                {"yes": 1, "no": 0}
            ).fillna(0).astype(int)

        X["target"] = y.iloc[:, 0].astype(str).str.strip().str.lower().map(
            {"ckd": 1, "notckd": 0}
        ).fillna(0).astype(int)

        df = X[num_cols + bin_cols + ["target"]].dropna()
        df.to_csv(out, index=False)
        print(f"  ✅ ckd.csv  ({len(df)} rows, {int(df['target'].sum())} CKD, cols: {num_cols+bin_cols})")
        return True
    except Exception as e:
        print(f"  ❌ ucimlrepo failed: {e}")
        return False


if __name__ == "__main__":
    print("Setting up disease datasets...\n")
    results = {
        "Breast Cancer":          make_breast_cancer(),
        "Stroke":                 make_stroke(),
        "Liver Disease (ILPD)":   make_liver(),
        "Chronic Kidney Disease": make_ckd(),
    }
    print("\n── Summary ──")
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    if not all(results.values()):
        print("\n[ข้อมูล] dataset ที่ fail อาจต้องดาวน์โหลดเองจาก Kaggle/UCI")
