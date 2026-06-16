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

    id         = Column(Integer, primary_key=True, autoincrement=True)
    hn         = Column(String(50))
    name       = Column(String(200), nullable=False)
    age        = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())

    assessments = relationship("Assessment", back_populates="patient",
                               order_by="desc(Assessment.created_at)")
    followups   = relationship("FollowUp", back_populates="patient")


class Assessment(Base):
    __tablename__ = "assessments"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    patient_id        = Column(Integer, ForeignKey("patients.id"), nullable=False)
    disease           = Column(String(50), nullable=False, default="diabetes")
    features_json     = Column(Text)
    pregnancies       = Column(Float)
    glucose           = Column(Float)
    blood_pressure    = Column(Float)
    skin_thickness    = Column(Float)
    insulin           = Column(Float)
    bmi               = Column(Float)
    diabetes_pedigree = Column(Float)
    risk_prob         = Column(Float)
    risk_level        = Column(String(10))
    prediction        = Column(Integer)
    note              = Column(Text)
    created_at        = Column(DateTime, server_default=func.now())

    patient = relationship("Patient", back_populates="assessments")


class FollowUp(Base):
    __tablename__ = "followups"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    patient_id      = Column(Integer, ForeignKey("patients.id"), nullable=False)
    scheduled_date  = Column(String(20), nullable=False)
    actual_date     = Column(String(20))
    status          = Column(String(20), nullable=False, default="scheduled")
    actual_outcome  = Column(Integer)
    glucose_new     = Column(Float)
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


def _migrate_to_assessments():
    """Create assessments table and copy existing patient assessment data."""
    Assessment.__table__.create(engine, checkfirst=True)

    with engine.connect() as conn:
        try:
            result = conn.execute(text(
                "SELECT id, hn, name, age, disease, features_json, "
                "pregnancies, glucose, blood_pressure, skin_thickness, "
                "insulin, bmi, diabetes_pedigree, risk_prob, risk_level, "
                "prediction, note, created_at FROM patients WHERE risk_prob IS NOT NULL"
            )).fetchall()
        except Exception:
            return  # table doesn't have old columns, already migrated

        for row in result:
            row = row._asdict() if hasattr(row, '_asdict') else dict(row._mapping)
            existing = conn.execute(text(
                "SELECT id FROM assessments WHERE patient_id = :pid LIMIT 1"
            ), {"pid": row["id"]}).fetchone()
            if existing:
                continue
            conn.execute(text("""
                INSERT INTO assessments
                (patient_id, disease, features_json, pregnancies, glucose, blood_pressure,
                 skin_thickness, insulin, bmi, diabetes_pedigree, risk_prob, risk_level,
                 prediction, note, created_at)
                VALUES (:patient_id, :disease, :features_json, :pregnancies, :glucose,
                 :blood_pressure, :skin_thickness, :insulin, :bmi, :diabetes_pedigree,
                 :risk_prob, :risk_level, :prediction, :note, :created_at)
            """), {
                "patient_id":        row["id"],
                "disease":           row.get("disease") or "diabetes",
                "features_json":     row.get("features_json"),
                "pregnancies":       row.get("pregnancies"),
                "glucose":           row.get("glucose"),
                "blood_pressure":    row.get("blood_pressure"),
                "skin_thickness":    row.get("skin_thickness"),
                "insulin":           row.get("insulin"),
                "bmi":               row.get("bmi"),
                "diabetes_pedigree": row.get("diabetes_pedigree"),
                "risk_prob":         row.get("risk_prob"),
                "risk_level":        row.get("risk_level"),
                "prediction":        row.get("prediction"),
                "note":              row.get("note"),
                "created_at":        row.get("created_at"),
            })
        conn.commit()


# ── Init ───────────────────────────────────────────────────────
def init_db():
    """สร้างตารางทั้งหมดถ้ายังไม่มี และ migrate columns ใหม่"""
    Base.metadata.create_all(engine)
    _migrate()


def _migrate():
    """เพิ่ม column ใหม่สำหรับ DB เก่าที่ยังไม่มี และย้ายข้อมูลไป assessments"""
    _add_col_if_missing("patients", "disease",       "VARCHAR(50)")
    _add_col_if_missing("patients", "features_json", "TEXT")
    _migrate_to_assessments()


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
def get_or_create_patient(hn: str | None, name: str, age: int) -> int:
    """Find patient by HN (if provided) or create new. Returns patient_id."""
    with Session(engine) as session:
        if hn:
            existing = session.query(Patient).filter(Patient.hn == hn).first()
            if existing:
                if existing.name != name or existing.age != age:
                    existing.name = name
                    existing.age  = age
                    session.commit()
                return existing.id
        p = Patient(hn=hn or None, name=name, age=age)
        session.add(p)
        session.commit()
        session.refresh(p)
        return p.id


def save_assessment(patient_id: int, data: dict) -> int:
    """Save one disease assessment for a patient. Returns assessment_id."""
    with Session(engine) as session:
        a = Assessment(**{k: data.get(k) for k in [
            "disease", "features_json", "pregnancies", "glucose",
            "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree",
            "risk_prob", "risk_level", "prediction", "note"
        ]})
        a.patient_id = patient_id
        session.add(a)
        session.commit()
        session.refresh(a)
        return a.id


def save_patient(data: dict) -> int:
    """Backward-compat: get/create patient + save assessment. Returns patient_id."""
    patient_id = get_or_create_patient(
        hn=data.get("hn"), name=data["name"], age=data.get("age", 0)
    )
    save_assessment(patient_id, data)
    return patient_id


def get_patient_by_id(pid: int) -> dict | None:
    """Returns patient identity + latest assessment merged (backward compat)."""
    with Session(engine) as session:
        p = session.get(Patient, pid)
        if p is None:
            return None
        result = {
            "id": p.id, "hn": p.hn, "name": p.name,
            "age": p.age, "created_at": p.created_at,
        }
        latest = session.query(Assessment).filter(
            Assessment.patient_id == pid
        ).order_by(Assessment.created_at.desc()).first()
        if latest:
            for col in Assessment.__table__.columns:
                if col.name not in ("id", "patient_id"):
                    result[col.name] = getattr(latest, col.name)
        return result


def get_assessments_by_patient(patient_id: int) -> pd.DataFrame:
    """All assessments for one patient, newest first."""
    with Session(engine) as session:
        rows = session.query(Assessment).filter(
            Assessment.patient_id == patient_id
        ).order_by(Assessment.created_at.desc()).all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "id": a.id, "disease": a.disease, "features_json": a.features_json,
        "glucose": a.glucose, "bmi": a.bmi, "risk_prob": a.risk_prob,
        "risk_level": a.risk_level, "prediction": a.prediction,
        "note": a.note, "created_at": a.created_at,
    } for a in rows])


def get_all_patients() -> pd.DataFrame:
    """One row per patient. Aggregates diseases from assessments."""
    with Session(engine) as session:
        patients = session.query(Patient).order_by(Patient.created_at.desc()).all()
        result = []
        for p in patients:
            assessments = session.query(Assessment).filter(
                Assessment.patient_id == p.id
            ).order_by(Assessment.created_at.desc()).all()
            diseases  = list(dict.fromkeys(a.disease for a in assessments if a.disease))
            latest    = assessments[0] if assessments else None
            worst     = max(assessments, key=lambda a: a.risk_prob or 0) if assessments else None
            result.append({
                "id":             p.id,
                "hn":             p.hn,
                "name":           p.name,
                "age":            p.age,
                "diseases":       diseases,
                "n_assessments":  len(assessments),
                "risk_prob":      worst.risk_prob if worst else None,
                "risk_level":     worst.risk_level if worst else None,
                "prediction":     1 if any(a.prediction == 1 for a in assessments) else 0,
                "latest_disease": latest.disease if latest else None,
                "disease":        latest.disease if latest else None,
                "features_json":  latest.features_json if latest else None,
                "glucose":        latest.glucose if latest else None,
                "bmi":            latest.bmi if latest else None,
                "note":           latest.note if latest else None,
                "created_at":     p.created_at,
            })
    return pd.DataFrame(result) if result else pd.DataFrame()


def search_patients(query: str) -> pd.DataFrame:
    q = f"%{query.strip()}%"
    with Session(engine) as session:
        patients = session.query(Patient).filter(
            (Patient.name.ilike(q)) | (Patient.hn.ilike(q))
        ).order_by(Patient.created_at.desc()).all()
        result = []
        for p in patients:
            assessments = session.query(Assessment).filter(
                Assessment.patient_id == p.id
            ).order_by(Assessment.created_at.desc()).all()
            diseases = list(dict.fromkeys(a.disease for a in assessments if a.disease))
            latest   = assessments[0] if assessments else None
            worst    = max(assessments, key=lambda a: a.risk_prob or 0) if assessments else None
            result.append({
                "id":             p.id,
                "hn":             p.hn,
                "name":           p.name,
                "age":            p.age,
                "diseases":       diseases,
                "n_assessments":  len(assessments),
                "risk_prob":      worst.risk_prob if worst else None,
                "risk_level":     worst.risk_level if worst else None,
                "prediction":     1 if any(a.prediction == 1 for a in assessments) else 0,
                "latest_disease": latest.disease if latest else None,
                "disease":        latest.disease if latest else None,
                "features_json":  latest.features_json if latest else None,
                "glucose":        latest.glucose if latest else None,
                "bmi":            latest.bmi if latest else None,
                "note":           latest.note if latest else None,
                "created_at":     p.created_at,
            })
    return pd.DataFrame(result) if result else pd.DataFrame()


def seed_demo_data():
    with Session(engine) as session:
        if session.query(Patient).count() > 0:
            return

    demos = [
        dict(hn="HN-0001", name="สมชาย ใจดี", age=52, disease="diabetes",
             pregnancies=0, glucose=148, blood_pressure=72, skin_thickness=35,
             insulin=0, bmi=33.6, diabetes_pedigree=0.627,
             risk_prob=0.92, risk_level="สูง", prediction=1, note="ส่งต่อแพทย์เฉพาะทาง"),
        dict(hn="HN-0002", name="สมหญิง รักสุข", age=31, disease="diabetes",
             pregnancies=1, glucose=85, blood_pressure=66, skin_thickness=29,
             insulin=0, bmi=26.6, diabetes_pedigree=0.351,
             risk_prob=0.06, risk_level="ต่ำ", prediction=0, note=""),
        dict(hn="HN-0003", name="วิชัย มั่นคง", age=45, disease="diabetes",
             pregnancies=0, glucose=120, blood_pressure=70, skin_thickness=30,
             insulin=0, bmi=30.0, diabetes_pedigree=0.500,
             risk_prob=0.52, risk_level="กลาง", prediction=1, note="นัดติดตาม 3 เดือน"),
        dict(hn="HN-0004", name="มาลี สุขใส", age=38, disease="diabetes",
             pregnancies=3, glucose=110, blood_pressure=68, skin_thickness=25,
             insulin=140, bmi=27.5, diabetes_pedigree=0.320,
             risk_prob=0.28, risk_level="ต่ำ", prediction=0, note=""),
        dict(hn="HN-0005", name="ประยุทธ์ ขยัน", age=60, disease="diabetes",
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
    today  = date.today().isoformat()
    cutoff = (date.today() + timedelta(days=days_ahead)).isoformat()
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
        with Session(engine) as s2:
            latest_a = s2.query(Assessment).filter(
                Assessment.patient_id == p.id
            ).order_by(Assessment.created_at.desc()).first()
        overdue = f.scheduled_date < today
        records.append({
            "followup_id":    f.id,
            "patient_id":     p.id,
            "hn":             p.hn,
            "name":           p.name,
            "scheduled_date": f.scheduled_date,
            "risk_level":     latest_a.risk_level if latest_a else None,
            "risk_prob":      latest_a.risk_prob if latest_a else None,
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

        rows = (
            session.query(FollowUp.actual_outcome, Assessment.prediction)
            .join(Patient, FollowUp.patient_id == Patient.id)
            .join(Assessment, Assessment.patient_id == Patient.id)
            .filter(FollowUp.actual_outcome.isnot(None))
            .distinct()
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
        with Session(engine) as s2:
            a = s2.query(Assessment).filter(
                Assessment.patient_id == p.id,
                Assessment.disease == "diabetes"
            ).order_by(Assessment.created_at.desc()).first()
        if not a:
            continue
        records.append({
            "Pregnancies":      a.pregnancies or 0,
            "Glucose":          f.glucose_new if f.glucose_new else (a.glucose or 0),
            "BloodPressure":    a.blood_pressure or 0,
            "SkinThickness":    a.skin_thickness or 0,
            "Insulin":          a.insulin or 0,
            "BMI":              f.bmi_new if f.bmi_new else (a.bmi or 0),
            "DiabetesPedigree": a.diabetes_pedigree or 0,
            "Age":              p.age or 0,
            "Outcome":          int(f.actual_outcome),
            "source":           "followup",
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
