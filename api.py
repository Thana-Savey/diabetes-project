"""
Multi-Disease Risk Prediction API
รันด้วย: uvicorn api:app --reload --port 8000
Docs:    http://localhost:8000/docs
"""

import os
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from pydantic import BaseModel, Field
from typing import Any
from auth import LoginRequest, TokenResponse, CurrentUser
from auth import verify_any, verify_jwt, require, create_token, pwd_context, USERS_DB
from database import (
    init_db, log_action, get_audit_logs, get_audit_summary,
    schedule_followup, complete_followup,
    get_followups_by_patient, get_pending_followups, get_followup_stats,
)
from retrain import run_retrain, list_model_versions, MODELS_DIR
from multimodel import load_all_models, preprocess_input, risk_level as get_risk_level
from diseases import DISEASES
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# ── สร้าง App ──────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Disease Risk Prediction API",
    description="ระบบประเมินความเสี่ยงโรคสำหรับโรงพยาบาล | LightGBM + Probability Calibration",
    version="2.0.0",
)

# ── โหลดโมเดลทุกโรคตอน startup ───────────────────────────────
print("⏳ Loading disease models...")
MODEL_REGISTRY: dict[str, dict] = load_all_models()
init_db()
print(f"✅ {len(MODEL_REGISTRY)} models ready: {list(MODEL_REGISTRY.keys())}")

# backward-compat สำหรับ retrain pipeline (diabetes only)
_CURRENT_MODEL_PATH = os.path.join(MODELS_DIR, "model_current.pkl")
_model_state: dict  = MODEL_REGISTRY.get("diabetes", {})

def _load_model_state():
    global _model_state, MODEL_REGISTRY
    from multimodel import load_disease_model
    state = load_disease_model("diabetes")
    if state:
        _model_state = state
        MODEL_REGISTRY["diabetes"] = state

# compat helpers (ใช้โดย /predict endpoint เดิม)
FEATURE_NAMES = DISEASES["diabetes"]["features"]
COLS_TO_FIX   = DISEASES["diabetes"]["cols_to_fix"]

def _get_model():  return _model_state["model"]
def _get_scaler(): return _model_state["scaler"]
def _get_fills():  return _model_state["fill_vals"]


# ── Schema ─────────────────────────────────────────────────────
class PatientData(BaseModel):
    pregnancies:       float = Field(..., ge=0, le=20,  description="จำนวนครั้งที่ตั้งครรภ์ (0 ถ้าไม่เคย)")
    glucose:           float = Field(..., ge=0, le=300, description="ระดับน้ำตาลในเลือด (mg/dL), ใส่ 0 ถ้าไม่ทราบ")
    blood_pressure:    float = Field(..., ge=0, le=200, description="ความดันโลหิต (mm Hg), ใส่ 0 ถ้าไม่ทราบ")
    skin_thickness:    float = Field(..., ge=0, le=100, description="ความหนาผิวหนัง (mm), ใส่ 0 ถ้าไม่ทราบ")
    insulin:           float = Field(..., ge=0, le=900, description="ระดับ insulin (μU/mL), ใส่ 0 ถ้าไม่ทราบ")
    bmi:               float = Field(..., ge=0, le=70,  description="ดัชนีมวลกาย (kg/m²)")
    diabetes_pedigree: float = Field(..., ge=0, le=3,   description="ค่าพันธุกรรมเบาหวาน (Diabetes Pedigree Function)")
    age:               int   = Field(..., ge=1, le=120, description="อายุ (ปี)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pregnancies": 2,
                "glucose": 148,
                "blood_pressure": 72,
                "skin_thickness": 35,
                "insulin": 0,
                "bmi": 33.6,
                "diabetes_pedigree": 0.627,
                "age": 50,
            }
        }
    }

class PredictionResult(BaseModel):
    risk_probability: float
    risk_level:       str
    prediction:       int
    interpretation:   str


# ── Helpers ────────────────────────────────────────────────────
def preprocess(data: PatientData) -> np.ndarray:
    row = {
        "Pregnancies":      data.pregnancies,
        "Glucose":          data.glucose,
        "BloodPressure":    data.blood_pressure,
        "SkinThickness":    data.skin_thickness,
        "Insulin":          data.insulin,
        "BMI":              data.bmi,
        "DiabetesPedigree": data.diabetes_pedigree,
        "Age":              data.age,
    }
    df_in = pd.DataFrame([row], columns=FEATURE_NAMES)
    fills = _get_fills()
    for col in COLS_TO_FIX:
        df_in[col] = df_in[col].replace(0, np.nan).fillna(fills[col])
    return _get_scaler().transform(df_in)


# ── Endpoints ──────────────────────────────────────────────────
@app.post("/token", response_model=TokenResponse, tags=["Auth"])
def login(body: LoginRequest, request: Request):
    """
    แพทย์/พยาบาล Login → ได้ JWT token (ใช้ได้ 8 ชั่วโมง)

    ทดสอบ: username=dr.somchai password=doctor1234
    """
    user = USERS_DB.get(body.username)
    if not user or not pwd_context.verify(body.password, user["hashed_pw"]):
        log_action(actor=body.username, actor_type="user", action="login",
                   ip_address=request.client.host, status="error",
                   detail="Invalid credentials")
        raise HTTPException(status_code=401, detail="Username หรือ Password ไม่ถูกต้อง")
    token = create_token({"sub": body.username})
    log_action(actor=body.username, actor_type="user", action="login",
               ip_address=request.client.host, detail=user["role"])
    return TokenResponse(
        access_token=token,
        expires_in=8 * 3600,
        full_name=user["full_name"],
        role=user["role"],
    )


# ── Multi-disease Endpoints ────────────────────────────────────
class DiseaseInput(BaseModel):
    features: dict[str, float]

class DiseaseResult(BaseModel):
    disease:          str
    disease_name_th:  str
    risk_probability: float
    risk_level:       str
    prediction:       int
    interpretation:   str


@app.get("/diseases", tags=["Multi-disease"])
def list_diseases():
    """รายชื่อโรคทั้งหมดที่รองรับ พร้อม features และ metrics"""
    return {
        key: {
            "name_th":     cfg["name_th"],
            "name_en":     cfg["name_en"],
            "icon":        cfg["icon"],
            "description": cfg["description"],
            "features":    cfg["features"],
            "input_fields": cfg["input_fields"],
            "metrics":     MODEL_REGISTRY.get(key, {}).get("metrics"),
        }
        for key, cfg in DISEASES.items()
        if key in MODEL_REGISTRY
    }




@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(patients: list[PatientData], caller=Security(require("predict"))):
    """
    ประเมินความเสี่ยงหลายคนพร้อมกัน (สำหรับ batch processing จาก HIS)

    รับ list ของผู้ป่วย → คืน list ของผลลัพธ์
    """
    if len(patients) > 1000:
        raise HTTPException(status_code=400, detail="Batch size ต้องไม่เกิน 1,000 รายการ")
    results = []
    for patient in patients:
        try:
            X = preprocess(patient)
            prob = float(_get_model().predict_proba(X)[0, 1])
            prediction = int(prob >= 0.5)
            risk_level = "สูง" if prob >= 0.7 else "กลาง" if prob >= 0.4 else "ต่ำ"
            results.append({
                "risk_probability": round(prob, 4),
                "risk_level": risk_level,
                "prediction": prediction,
            })
        except Exception as e:
            results.append({"error": str(e)})
    return {"total": len(patients), "results": results}


@app.post("/predict/{disease}", response_model=DiseaseResult, tags=["Multi-disease"])
def predict_disease(
    disease: str,
    body: DiseaseInput,
    request: Request,
    caller=Security(require("predict")),
):
    """
    ประเมินความเสี่ยงโรคที่ระบุ

    - **disease**: `diabetes` | `heart`
    - **features**: dict ของค่า feature ตาม /diseases
    """
    if disease not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"ไม่รองรับโรค '{disease}' — ที่รองรับ: {list(MODEL_REGISTRY.keys())}",
        )

    state = MODEL_REGISTRY[disease]
    cfg   = DISEASES[disease]

    try:
        X    = preprocess_input(disease, body.features, state)
        prob = float(state["model"].predict_proba(X)[0, 1])
    except Exception as e:
        log_action(actor=caller["id"], actor_type=caller["type"],
                   action=f"predict_{disease}",
                   ip_address=request.client.host, status="error", detail=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    pred       = int(prob >= 0.5)
    rlevel     = get_risk_level(prob)
    label      = cfg["risk_labels"].get(pred, "")

    if rlevel == "สูง":
        interpretation = f"{label} — ความเสี่ยงสูง แนะนำส่งพบแพทย์เฉพาะทางโดยด่วน"
    elif rlevel == "กลาง":
        interpretation = f"{label} — ความเสี่ยงปานกลาง แนะนำนัดติดตามและปรับพฤติกรรม"
    else:
        interpretation = f"{label} — ความเสี่ยงต่ำ แนะนำตรวจสุขภาพประจำปีตามปกติ"

    log_action(
        actor=caller["id"], actor_type=caller["type"],
        action=f"predict_{disease}",
        input_data=body.features,
        risk_prob=round(prob, 4), risk_level=rlevel, prediction=pred,
        ip_address=request.client.host,
    )
    return DiseaseResult(
        disease=disease,
        disease_name_th=cfg["name_th"],
        risk_probability=round(prob, 4),
        risk_level=rlevel,
        prediction=pred,
        interpretation=interpretation,
    )


@app.get("/me", tags=["Auth"])
def whoami(caller=Security(require("predict"))):
    """ดูข้อมูลและสิทธิ์ของตัวเอง"""
    from auth import PERMISSION_MAP, ROLE_THAI
    role = caller.get("role", "api_key")
    my_permissions = [p for p, roles in PERMISSION_MAP.items() if role in roles]
    return {
        "id":          caller["id"],
        "role":        role,
        "role_thai":   ROLE_THAI.get(role, role),
        "permissions": my_permissions,
    }


@app.get("/health", tags=["System"])
def health_check():
    """ตรวจสอบว่า API ทำงานปกติ"""
    return {"status": "ok", "model": "LightGBM + Platt Calibration", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(patient: PatientData, request: Request, caller=Security(require("predict"))):
    """
    ประเมินความเสี่ยงเบาหวานจากค่าทางการแพทย์

    - **risk_probability**: ความน่าจะเป็น 0.0–1.0
    - **risk_level**: ต่ำ / กลาง / สูง
    - **prediction**: 0 = ไม่เสี่ยง, 1 = เสี่ยง
    """
    try:
        X = preprocess(patient)
        prob = float(_get_model().predict_proba(X)[0, 1])
    except Exception as e:
        log_action(actor=caller["id"], actor_type=caller["type"], action="predict",
                   ip_address=request.client.host, status="error", detail=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    prediction = int(prob >= 0.5)
    if prob >= 0.7:
        risk_level = "สูง"
        interpretation = "ความเสี่ยงสูง แนะนำส่งพบแพทย์เฉพาะทางโดยด่วน"
    elif prob >= 0.4:
        risk_level = "กลาง"
        interpretation = "ความเสี่ยงปานกลาง แนะนำนัดติดตามและปรับพฤติกรรม"
    else:
        risk_level = "ต่ำ"
        interpretation = "ความเสี่ยงต่ำ แนะนำตรวจสุขภาพประจำปีตามปกติ"

    log_action(
        actor=caller["id"], actor_type=caller["type"], action="predict",
        input_data=patient.model_dump(),
        risk_prob=round(prob, 4), risk_level=risk_level, prediction=prediction,
        ip_address=request.client.host,
    )
    return PredictionResult(
        risk_probability=round(prob, 4),
        risk_level=risk_level,
        prediction=prediction,
        interpretation=interpretation,
    )


@app.get("/audit/logs", tags=["Audit"])
def audit_logs(
    limit: int = 100,
    actor: str | None = None,
    action: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    caller=Security(require("audit:read")),
):
    """
    ดู Audit Log — เฉพาะ doctor / admin เท่านั้น

    ตัวกรอง: actor, action (login/predict), date_from, date_to (YYYY-MM-DD)
    """
    df = get_audit_logs(limit=limit, actor=actor, action=action,
                        date_from=date_from, date_to=date_to)
    return {"total": len(df), "logs": df.to_dict("records")}


@app.get("/audit/summary", tags=["Audit"])
def audit_summary(caller=Security(require("audit:read"))):
    """สรุปสถิติการใช้งานระบบ — เฉพาะ doctor / admin"""
    return get_audit_summary()


# ── Follow-up Endpoints ────────────────────────────────────────
class FollowUpCreate(BaseModel):
    patient_id:     int
    scheduled_date: str   # YYYY-MM-DD
    note:           str | None = None

class FollowUpComplete(BaseModel):
    actual_date:    str         # YYYY-MM-DD
    actual_outcome: int | None = None   # 0=ไม่เป็น, 1=เป็นเบาหวาน, None=ยังไม่ทราบ
    glucose_new:    float | None = None
    bmi_new:        float | None = None
    note:           str | None = None
    status:         str = "completed"   # completed / missed / cancelled


@app.post("/followups", tags=["Follow-up"])
def create_followup(body: FollowUpCreate, caller=Security(require("followup:write"))):
    """นัด follow-up ให้ผู้ป่วย (nurse / doctor / admin)"""
    fid = schedule_followup(
        patient_id=body.patient_id,
        scheduled_date=body.scheduled_date,
        note=body.note,
        created_by=caller["id"],
    )
    return {"followup_id": fid, "message": "บันทึกนัดติดตามเรียบร้อย"}


@app.patch("/followups/{followup_id}", tags=["Follow-up"])
def update_followup(followup_id: int, body: FollowUpComplete, caller=Security(require("followup:write"))):
    """บันทึกผลการมาติดตาม"""
    ok = complete_followup(
        followup_id=followup_id,
        actual_date=body.actual_date,
        actual_outcome=body.actual_outcome,
        glucose_new=body.glucose_new,
        bmi_new=body.bmi_new,
        note=body.note,
        status=body.status,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="ไม่พบ follow-up นี้")
    return {"message": "อัปเดตผลติดตามเรียบร้อย"}


@app.get("/followups/pending", tags=["Follow-up"])
def pending_followups(days: int = 30, caller=Security(require("followup:read"))):
    """รายชื่อผู้ป่วยที่ต้องติดตามภายใน {days} วัน"""
    df = get_pending_followups(days_ahead=days)
    return {"total": len(df), "followups": df.to_dict("records")}


@app.get("/followups/stats", tags=["Follow-up"])
def followup_stats(caller=Security(require("followup:read"))):
    """สถิติ follow-up และ accuracy ของโมเดลเทียบกับผลจริง"""
    return get_followup_stats()


# ── Admin: Retrain ─────────────────────────────────────────────
@app.post("/admin/retrain", tags=["Admin"])
def trigger_retrain(request: Request, caller=Security(require("model:retrain"))):
    """
    Retrain โมเดลด้วยข้อมูล follow-up ที่ยืนยันแล้ว — เฉพาะ admin เท่านั้น

    - ดึง follow-up จาก DB + merge กับ diabetes.csv
    - Retrain LightGBM + Platt calibration
    - บันทึก model ใหม่ → hot-reload ทันที (ไม่ต้อง restart)
    - ต้องมี follow-up อย่างน้อย 5 ราย จึงจะ retrain
    """
    result = run_retrain(triggered_by=caller["id"])

    if result["status"] == "skipped":
        return result

    # Hot-reload: อัปเดต state ในหน่วยความจำทันที
    _load_model_state()

    return result


@app.get("/admin/models", tags=["Admin"])
def model_versions(caller=Security(require("model:read"))):
    """รายการ model versions ทั้งหมดที่บันทึกไว้ — doctor / admin"""
    versions = list_model_versions()
    current = _model_state.get("trained_at")
    return {
        "current_model": current,
        "n_followup_in_current": _model_state.get("n_followup", 0),
        "current_metrics": _model_state.get("metrics"),
        "versions": versions,
    }


@app.get("/followups/patient/{patient_id}", tags=["Follow-up"])
def patient_followups(patient_id: int, caller=Security(require("followup:read"))):
    """ประวัติ follow-up ทั้งหมดของผู้ป่วย"""
    df = get_followups_by_patient(patient_id)
    return {"patient_id": patient_id, "total": len(df), "followups": df.to_dict("records")}


