"""
database.py — จัดการฐานข้อมูลสำหรับระบบ Diabetes Risk Predictor

รองรับทั้ง 2 โหมดผ่าน environment variable DATABASE_URL:
  ไม่ set  → SQLite  (dev / ทดสอบ)
  set      → PostgreSQL (production โรงพยาบาล)

ตัวอย่าง:
  DATABASE_URL=postgresql://user:pass@localhost:5432/diabetes
"""

import os
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    create_engine, text,
    Column, Integer, String, Float, Text, DateTime, ForeignKey
)
from sqlalchemy.orm import relationship
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.sql import func

# ── Database URL ───────────────────────────────────────────────
_sqlite_path = Path(__file__).parent / "patients.db"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{_sqlite_path}"
)

# SQLite ต้องการ connect_args พิเศษ
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
Base   = declarative_base()


# ── Models (ORM) ───────────────────────────────────────────────
class Patient(Base):
    __tablename__ = "patients"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    hn                = Column(String(50))
    name              = Column(String(200), nullable=False)
    age               = Column(Integer)
    pregnancies       = Column(Float)
    glucose           = Column(Float)
    blood_pressure    = Column(Float)
    skin_thickness    = Column(Float)
    insulin           = Column(Float)
    bmi               = Column(Float)
    diabetes_pedigree = Column(Float)
    disease           = Column(String(50))   # disease key: "diabetes","heart","stroke", ...
    risk_prob         = Column(Float)
    risk_level        = Column(String(10))
    prediction        = Column(Integer)
    note              = Column(Text)
    created_at        = Column(DateTime, server_default=func.now())
    followups         = relationship("FollowUp", back_populates="patient")


class FollowUp(Base):
    __tablename__ = "followups"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    patient_id      = Column(Integer, ForeignKey("patients.id"), nullable=False)
    scheduled_date  = Column(String(20), nullable=False)   # วันที่นัด
    actual_date     = Column(String(20))                   # วันที่มาจริง
    status          = Column(String(20), nullable=False, default="scheduled")
                     # scheduled / completed / missed / cancelled
    actual_outcome  = Column(Integer)                      # 0/1 หรือ None ถ้ายังไม่รู้
    glucose_new     = Column(Float)                        # ค่า glucose ตอน follow-up
    bmi_new         = Column(Float)
    note            = Column(Text)
    created_by      = Column(String(100))
    created_at      = Column(DateTime, server_default=func.now())
    updated_at      = Column(DateTime, server_default=func.now(), onupdate=func.now())

    patient = relationship("Patient", back_populates="followups")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    timestamp   = Column(DateTime, server_default=func.now())
    actor       = Column(String(100), nullable=False)
    actor_type  = Column(String(20),  nullable=False)
    action      = Column(String(50),  nullable=False)
    patient_hn  = Column(String(50))
    input_hash  = Column(String(32))
    risk_prob   = Column(Float)
    risk_level  = Column(String(10))
    prediction  = Column(Integer)
    ip_address  = Column(String(50))
    status      = Column(String(20), nullable=False, default="success")
    detail      = Column(Text)


# ── Migration helper ───────────────────────────────────────────
def _add_col_if_missing(table: str, col: str, col_type: str) -> None:
    with engine.connect() as conn:
        try:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
            conn.commit()
        except Exception:
            pass  # column already exists


# ── Init ───────────────────────────────────────────────────────
def init_db():
    """สร้างตารางทั้งหมดถ้ายังไม่มี และ migrate columns ใหม่"""
    Base.metadata.create_all(engine)
    _migrate()


def _migrate():
    """เพิ่ม column ใหม่สำหรับ DB เก่าที่ยังไม่มี"""
    _add_col_if_missing("patients", "disease", "VARCHAR(50)")


# ── Audit Log ──────────────────────────────────────────────────
def log_action(
    actor: str,
    actor_type: str,
    action: str,
    patient_hn: str | None = None,
    input_data: dict | None = None,
    risk_prob: float | None = None,
    risk_level: str | None = None,
    prediction: int | None = None,
    ip_address: str | None = None,
    status: str = "success",
    detail: str | None = None,
) -> None:
    input_hash = None
    if input_data:
        raw = str(sorted(input_data.items())).encode()
        input_hash = hashlib.sha256(raw).hexdigest()[:16]

    with Session(engine) as session:
        session.add(AuditLog(
            actor=actor, actor_type=actor_type, action=action,
            patient_hn=patient_hn, input_hash=input_hash,
            risk_prob=risk_prob, risk_level=risk_level, prediction=prediction,
            ip_address=ip_address, status=status, detail=detail,
        ))
        session.commit()


def get_audit_logs(
    limit: int = 100,
    actor: str | None = None,
    action: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    with Session(engine) as session:
        q = session.query(AuditLog)
        if actor:      q = q.filter(AuditLog.actor.ilike(f"%{actor}%"))
        if action:     q = q.filter(AuditLog.action == action)
        if date_from:  q = q.filter(AuditLog.timestamp >= date_from)
        if date_to:    q = q.filter(AuditLog.timestamp <= date_to)
        rows = q.order_by(AuditLog.timestamp.desc()).limit(limit).all()

    df = pd.DataFrame([{
        "id": r.id, "timestamp": r.timestamp, "actor": r.actor,
        "actor_type": r.actor_type, "action": r.action,
        "patient_hn": r.patient_hn, "risk_prob": r.risk_prob,
        "risk_level": r.risk_level, "prediction": r.prediction,
        "ip_address": r.ip_address, "status": r.status, "detail": r.detail,
    } for r in rows])
    return df.astype(object).where(df.notna(), None)


def get_audit_summary() -> dict:
    with Session(engine) as session:
        total  = session.query(AuditLog).count()
        today  = session.query(AuditLog).filter(
            func.date(AuditLog.timestamp) == func.current_date()
        ).count()
        errors = session.query(AuditLog).filter(AuditLog.status == "error").count()

        top_actors = session.execute(text(
            "SELECT actor, COUNT(*) as count FROM audit_logs "
            "GROUP BY actor ORDER BY count DESC LIMIT 5"
        )).fetchall()
        by_action = session.execute(text(
            "SELECT action, COUNT(*) as count FROM audit_logs "
            "GROUP BY action ORDER BY count DESC"
        )).fetchall()

    return {
        "total": total, "today": today, "errors": errors,
        "top_actors": [{"actor": r[0], "count": r[1]} for r in top_actors],
        "by_action":  [{"action": r[0], "count": r[1]} for r in by_action],
    }


# ── Patients ───────────────────────────────────────────────────
def save_patient(data: dict) -> int:
    with Session(engine) as session:
        p = Patient(**{k: data.get(k) for k in [
            "hn","name","age","disease","pregnancies","glucose","blood_pressure",
            "skin_thickness","insulin","bmi","diabetes_pedigree",
            "risk_prob","risk_level","prediction","note"
        ]})
        session.add(p)
        session.commit()
        session.refresh(p)
        return p.id


def search_patients(query: str) -> pd.DataFrame:
    q = f"%{query.strip()}%"
    with Session(engine) as session:
        rows = session.query(Patient).filter(
            (Patient.name.ilike(q)) | (Patient.hn.ilike(q))
        ).order_by(Patient.created_at.desc()).all()
    return _patients_to_df(rows)


def get_patient_by_id(pid: int) -> dict | None:
    with Session(engine) as session:
        p = session.get(Patient, pid)
        if p is None:
            return None
        return {c.name: getattr(p, c.name) for c in Patient.__table__.columns}


def get_all_patients() -> pd.DataFrame:
    with Session(engine) as session:
        rows = session.query(Patient).order_by(Patient.created_at.desc()).all()
    return _patients_to_df(rows)


def _patients_to_df(rows) -> pd.DataFrame:
    def _disease(r):
        # backward compat: old rows have disease=None → parse from note or default diabetes
        if getattr(r, "disease", None):
            return r.disease
        note = r.note or ""
        if note.startswith("[") and "]" in note:
            # format "[โรคหลอดเลือดสมอง] ..."
            return None   # keep None, resolved to label in app.py
        return "diabetes"

    return pd.DataFrame([{
        "id": r.id, "hn": r.hn, "name": r.name, "age": r.age,
        "disease": _disease(r),
        "glucose": r.glucose, "bmi": r.bmi,
        "risk_prob": r.risk_prob, "risk_level": r.risk_level,
        "prediction": r.prediction,
        "note": r.note,
        "created_at": r.created_at,
    } for r in rows])


def seed_demo_data():
    with Session(engine) as session:
        if session.query(Patient).count() > 0:
            return

    demos = [
        dict(hn="HN-0001", name="สมชาย ใจดี",    age=52,
             pregnancies=0, glucose=148, blood_pressure=72, skin_thickness=35,
             insulin=0, bmi=33.6, diabetes_pedigree=0.627,
             risk_prob=0.92, risk_level="สูง", prediction=1, note="ส่งต่อแพทย์เฉพาะทาง"),
        dict(hn="HN-0002", name="สมหญิง รักสุข", age=31,
             pregnancies=1, glucose=85,  blood_pressure=66, skin_thickness=29,
             insulin=0, bmi=26.6, diabetes_pedigree=0.351,
             risk_prob=0.06, risk_level="ต่ำ", prediction=0, note=""),
        dict(hn="HN-0003", name="วิชัย มั่นคง",  age=45,
             pregnancies=0, glucose=120, blood_pressure=70, skin_thickness=30,
             insulin=0, bmi=30.0, diabetes_pedigree=0.500,
             risk_prob=0.52, risk_level="กลาง", prediction=1, note="นัดติดตาม 3 เดือน"),
        dict(hn="HN-0004", name="มาลี สุขใส",    age=38,
             pregnancies=3, glucose=110, blood_pressure=68, skin_thickness=25,
             insulin=140, bmi=27.5, diabetes_pedigree=0.320,
             risk_prob=0.28, risk_level="ต่ำ", prediction=0, note=""),
        dict(hn="HN-0005", name="ประยุทธ์ ขยัน", age=60,
             pregnancies=0, glucose=175, blood_pressure=88, skin_thickness=40,
             insulin=0, bmi=38.2, diabetes_pedigree=0.810,
             risk_prob=0.97, risk_level="สูง", prediction=1, note="รับยาแล้ว"),
    ]
    for d in demos:
        save_patient(d)


# ── Follow-up CRUD ────────────────────────────────────────────
def schedule_followup(
    patient_id: int,
    scheduled_date: str,
    note: str | None = None,
    created_by: str | None = None,
) -> int:
    with Session(engine) as session:
        f = FollowUp(
            patient_id=patient_id,
            scheduled_date=scheduled_date,
            status="scheduled",
            note=note,
            created_by=created_by,
        )
        session.add(f)
        session.commit()
        session.refresh(f)
        return f.id


def complete_followup(
    followup_id: int,
    actual_date: str,
    actual_outcome: int | None = None,
    glucose_new: float | None = None,
    bmi_new: float | None = None,
    note: str | None = None,
    status: str = "completed",
) -> bool:
    with Session(engine) as session:
        f = session.get(FollowUp, followup_id)
        if f is None:
            return False
        f.actual_date    = actual_date
        f.status         = status
        f.actual_outcome = actual_outcome
        f.glucose_new    = glucose_new
        f.bmi_new        = bmi_new
        if note:
            f.note = note
        session.commit()
        return True


def get_followups_by_patient(patient_id: int) -> pd.DataFrame:
    with Session(engine) as session:
        rows = session.query(FollowUp).filter(
            FollowUp.patient_id == patient_id
        ).order_by(FollowUp.scheduled_date.desc()).all()
    return _followups_to_df(rows)


def get_pending_followups(days_ahead: int = 30) -> pd.DataFrame:
    """คืนนัดที่ยังไม่ได้ติดตาม (scheduled) และนัดที่เลยกำหนดแล้ว (missed)"""
    from datetime import date, timedelta
    today     = date.today().isoformat()
    cutoff    = (date.today() + timedelta(days=days_ahead)).isoformat()
    with Session(engine) as session:
        rows = (
            session.query(FollowUp, Patient)
            .join(Patient, FollowUp.patient_id == Patient.id)
            .filter(
                FollowUp.status == "scheduled",
                FollowUp.scheduled_date <= cutoff,
            )
            .order_by(FollowUp.scheduled_date)
            .all()
        )
    records = []
    for f, p in rows:
        overdue = f.scheduled_date < today
        records.append({
            "followup_id":    f.id,
            "patient_id":     p.id,
            "hn":             p.hn,
            "name":           p.name,
            "scheduled_date": f.scheduled_date,
            "risk_level":     p.risk_level,
            "risk_prob":      p.risk_prob,
            "note":           f.note,
            "overdue":        overdue,
        })
    return pd.DataFrame(records)


def get_followup_stats() -> dict:
    """สถิติ follow-up: accuracy ของโมเดล vs ผลจริง"""
    with Session(engine) as session:
        total     = session.query(FollowUp).count()
        completed = session.query(FollowUp).filter(FollowUp.status == "completed").count()
        missed    = session.query(FollowUp).filter(FollowUp.status == "missed").count()
        pending   = session.query(FollowUp).filter(FollowUp.status == "scheduled").count()

        # เปรียบ prediction เดิม vs actual_outcome จริง
        rows = (
            session.query(FollowUp.actual_outcome, Patient.prediction)
            .join(Patient, FollowUp.patient_id == Patient.id)
            .filter(FollowUp.actual_outcome.isnot(None))
            .all()
        )

    correct = sum(1 for actual, pred in rows if actual == pred)
    accuracy = round(correct / len(rows) * 100, 1) if rows else None

    return {
        "total":         total,
        "completed":     completed,
        "missed":        missed,
        "pending":       pending,
        "confirmed_cases": len(rows),
        "model_accuracy_on_followup": accuracy,
    }


def get_confirmed_followups() -> pd.DataFrame:
    """
    ดึงเคส follow-up ที่แพทย์ยืนยันผลแล้ว (actual_outcome ไม่เป็น None)
    คืน DataFrame พร้อม feature columns เหมือน diabetes.csv
    เพื่อนำไป retrain โมเดล
    """
    with Session(engine) as session:
        rows = (
            session.query(FollowUp, Patient)
            .join(Patient, FollowUp.patient_id == Patient.id)
            .filter(FollowUp.actual_outcome.isnot(None))
            .all()
        )

    records = []
    for f, p in rows:
        records.append({
            "Pregnancies":       p.pregnancies or 0,
            "Glucose":           f.glucose_new  if f.glucose_new  else (p.glucose or 0),
            "BloodPressure":     p.blood_pressure or 0,
            "SkinThickness":     p.skin_thickness or 0,
            "Insulin":           p.insulin or 0,
            "BMI":               f.bmi_new if f.bmi_new else (p.bmi or 0),
            "DiabetesPedigree":  p.diabetes_pedigree or 0,
            "Age":               p.age or 0,
            "Outcome":           int(f.actual_outcome),
            "source":            "followup",
        })
    return pd.DataFrame(records)


def _followups_to_df(rows) -> pd.DataFrame:
    return pd.DataFrame([{
        "id":             r.id,
        "patient_id":     r.patient_id,
        "scheduled_date": r.scheduled_date,
        "actual_date":    r.actual_date,
        "status":         r.status,
        "actual_outcome": r.actual_outcome,
        "glucose_new":    r.glucose_new,
        "bmi_new":        r.bmi_new,
        "note":           r.note,
        "created_by":     r.created_by,
        "created_at":     r.created_at,
    } for r in rows])
