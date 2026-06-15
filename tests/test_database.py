"""
test_database.py — ทดสอบ database CRUD functions
"""
import pytest
from database import (
    save_patient, get_patient_by_id, get_all_patients, search_patients,
    schedule_followup, complete_followup,
    get_followups_by_patient, get_pending_followups, get_followup_stats,
    get_confirmed_followups, log_action, get_audit_logs,
)


# ── Patient CRUD ───────────────────────────────────────────────
class TestPatient:
    def test_save_and_retrieve_patient(self):
        pid = save_patient({
            "hn": "TEST-001", "name": "ทดสอบ ระบบ", "age": 40,
            "pregnancies": 1, "glucose": 130, "blood_pressure": 75,
            "skin_thickness": 25, "insulin": 0, "bmi": 28.5,
            "diabetes_pedigree": 0.4,
            "risk_prob": 0.65, "risk_level": "กลาง", "prediction": 1,
        })
        assert isinstance(pid, int) and pid > 0

        p = get_patient_by_id(pid)
        assert p is not None
        assert p["name"] == "ทดสอบ ระบบ"
        assert p["hn"] == "TEST-001"
        assert p["risk_level"] == "กลาง"

    def test_get_nonexistent_patient_returns_none(self):
        p = get_patient_by_id(999999)
        assert p is None

    def test_search_by_name(self):
        save_patient({
            "name": "ค้นหา ทดสอบ", "age": 35,
            "risk_prob": 0.3, "risk_level": "ต่ำ", "prediction": 0,
        })
        df = search_patients("ค้นหา")
        assert len(df) >= 1
        assert any("ค้นหา" in n for n in df["name"].tolist())

    def test_search_by_hn(self):
        save_patient({
            "hn": "HN-SEARCH", "name": "ทดสอบ HN", "age": 50,
            "risk_prob": 0.8, "risk_level": "สูง", "prediction": 1,
        })
        df = search_patients("HN-SEARCH")
        assert len(df) >= 1

    def test_get_all_patients_returns_dataframe(self):
        df = get_all_patients()
        assert not df.empty
        assert "name" in df.columns
        assert "risk_level" in df.columns


# ── Follow-up CRUD ─────────────────────────────────────────────
class TestFollowUp:
    @pytest.fixture
    def patient_id(self):
        return save_patient({
            "hn": "FU-TEST", "name": "Follow-up ทดสอบ", "age": 45,
            "glucose": 140, "bmi": 31.0,
            "risk_prob": 0.75, "risk_level": "สูง", "prediction": 1,
        })

    def test_schedule_and_retrieve(self, patient_id):
        fid = schedule_followup(
            patient_id=patient_id,
            scheduled_date="2026-08-01",
            note="นัดตรวจ 3 เดือน",
            created_by="nurse.malee",
        )
        assert isinstance(fid, int) and fid > 0

        df = get_followups_by_patient(patient_id)
        assert len(df) >= 1
        assert any(r["id"] == fid for r in df.to_dict("records"))

    def test_complete_followup(self, patient_id):
        fid = schedule_followup(patient_id=patient_id, scheduled_date="2026-08-15")
        ok  = complete_followup(
            followup_id=fid,
            actual_date="2026-08-15",
            actual_outcome=1,
            glucose_new=162.0,
            bmi_new=32.1,
            status="completed",
        )
        assert ok is True

        df = get_followups_by_patient(patient_id)
        row = df[df["id"] == fid].iloc[0]
        assert row["status"] == "completed"
        assert row["actual_outcome"] == 1
        assert row["glucose_new"] == 162.0

    def test_complete_nonexistent_returns_false(self):
        ok = complete_followup(followup_id=999999, actual_date="2026-08-01")
        assert ok is False

    def test_get_pending_followups(self, patient_id):
        schedule_followup(patient_id=patient_id, scheduled_date="2026-06-20")
        df = get_pending_followups(days_ahead=365)
        assert "name" in df.columns

    def test_followup_stats_keys(self):
        stats = get_followup_stats()
        for key in ("total", "completed", "missed", "pending", "confirmed_cases"):
            assert key in stats

    def test_confirmed_followups_for_retrain(self, patient_id):
        # สร้าง follow-up ที่มี actual_outcome
        fid = schedule_followup(patient_id=patient_id, scheduled_date="2026-07-01")
        complete_followup(fid, actual_date="2026-07-01", actual_outcome=0, status="completed")

        df = get_confirmed_followups()
        assert "Outcome" in df.columns
        assert "Glucose" in df.columns
        assert len(df) >= 1
        assert set(df["Outcome"].unique()).issubset({0, 1})


# ── Audit Log ──────────────────────────────────────────────────
class TestAuditLog:
    def test_log_and_retrieve(self):
        log_action(
            actor="test.user", actor_type="user",
            action="predict", risk_prob=0.75, risk_level="สูง", prediction=1,
        )
        df = get_audit_logs(limit=10, actor="test.user")
        assert len(df) >= 1
        assert df.iloc[0]["actor"] == "test.user"

    def test_log_with_input_hash(self):
        log_action(
            actor="his-system", actor_type="api_key",
            action="predict",
            input_data={"glucose": 148, "bmi": 33.6},
        )
        df = get_audit_logs(limit=5, actor="his-system")
        assert len(df) >= 1
