"""
test_api.py — ทดสอบ API endpoints และการ enforce RBAC
"""
import pytest
from tests.conftest import SAMPLE_PATIENT


# ── Health ─────────────────────────────────────────────────────
class TestHealth:
    def test_health_ok(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_no_auth_needed(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200


# ── Predict ────────────────────────────────────────────────────
class TestPredict:
    def test_predict_with_api_key(self, api_client):
        r = api_client.post(
            "/predict", json=SAMPLE_PATIENT,
            headers={"X-API-Key": "test-api-key-123"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "risk_probability" in data
        assert "risk_level" in data
        assert data["risk_level"] in ("ต่ำ", "กลาง", "สูง")
        assert 0.0 <= data["risk_probability"] <= 1.0

    def test_predict_with_doctor_jwt(self, api_client, doctor_token):
        r = api_client.post(
            "/predict", json=SAMPLE_PATIENT,
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert r.status_code == 200

    def test_predict_with_nurse_jwt(self, api_client, nurse_token):
        r = api_client.post(
            "/predict", json=SAMPLE_PATIENT,
            headers={"Authorization": f"Bearer {nurse_token}"},
        )
        assert r.status_code == 200

    def test_predict_no_auth_rejected(self, api_client):
        r = api_client.post("/predict", json=SAMPLE_PATIENT)
        assert r.status_code == 401

    def test_predict_invalid_glucose(self, api_client):
        bad = {**SAMPLE_PATIENT, "glucose": 9999}
        r = api_client.post(
            "/predict", json=bad,
            headers={"X-API-Key": "test-api-key-123"},
        )
        assert r.status_code == 422

    def test_predict_batch(self, api_client):
        r = api_client.post(
            "/predict/batch", json=[SAMPLE_PATIENT, SAMPLE_PATIENT],
            headers={"X-API-Key": "test-api-key-123"},
        )
        assert r.status_code == 200
        assert r.json()["total"] == 2


# ── Audit (doctor/admin only) ───────────────────────────────────
class TestAudit:
    def test_doctor_can_see_audit_logs(self, api_client, doctor_token):
        r = api_client.get(
            "/audit/logs",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert r.status_code == 200
        assert "logs" in r.json()

    def test_nurse_cannot_see_audit_logs(self, api_client, nurse_token):
        r = api_client.get(
            "/audit/logs",
            headers={"Authorization": f"Bearer {nurse_token}"},
        )
        assert r.status_code == 403
        assert "สิทธิ์ไม่เพียงพอ" in r.json()["detail"]

    def test_api_key_cannot_see_audit_logs(self, api_client):
        r = api_client.get(
            "/audit/logs",
            headers={"X-API-Key": "test-api-key-123"},
        )
        assert r.status_code == 403

    def test_audit_summary_doctor(self, api_client, doctor_token):
        r = api_client.get(
            "/audit/summary",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert r.status_code == 200
        assert "total" in r.json()


# ── Follow-up ──────────────────────────────────────────────────
class TestFollowUp:
    def test_nurse_can_schedule_followup(self, api_client, nurse_token):
        r = api_client.post(
            "/followups",
            json={"patient_id": 1, "scheduled_date": "2026-07-01", "note": "test"},
            headers={"Authorization": f"Bearer {nurse_token}"},
        )
        assert r.status_code == 200
        assert "followup_id" in r.json()

    def test_api_key_cannot_schedule_followup(self, api_client):
        r = api_client.post(
            "/followups",
            json={"patient_id": 1, "scheduled_date": "2026-07-01"},
            headers={"X-API-Key": "test-api-key-123"},
        )
        assert r.status_code == 403

    def test_nurse_can_see_pending(self, api_client, nurse_token):
        r = api_client.get(
            "/followups/pending",
            headers={"Authorization": f"Bearer {nurse_token}"},
        )
        assert r.status_code == 200

    def test_nurse_can_see_stats(self, api_client, nurse_token):
        r = api_client.get(
            "/followups/stats",
            headers={"Authorization": f"Bearer {nurse_token}"},
        )
        assert r.status_code == 200

    def test_complete_followup(self, api_client, doctor_token):
        # สร้างนัดก่อน
        r = api_client.post(
            "/followups",
            json={"patient_id": 1, "scheduled_date": "2026-06-20"},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        fid = r.json()["followup_id"]

        # บันทึกผล
        r2 = api_client.patch(
            f"/followups/{fid}",
            json={"actual_date": "2026-06-20", "actual_outcome": 1,
                  "glucose_new": 155.0, "status": "completed"},
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert r2.status_code == 200


# ── Admin (admin only) ─────────────────────────────────────────
class TestAdmin:
    def test_doctor_can_see_model_versions(self, api_client, doctor_token):
        r = api_client.get(
            "/admin/models",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert r.status_code == 200
        assert "current_model" in r.json()

    def test_nurse_cannot_see_model_versions(self, api_client, nurse_token):
        r = api_client.get(
            "/admin/models",
            headers={"Authorization": f"Bearer {nurse_token}"},
        )
        assert r.status_code == 403

    def test_doctor_cannot_retrain(self, api_client, doctor_token):
        r = api_client.post(
            "/admin/retrain",
            headers={"Authorization": f"Bearer {doctor_token}"},
        )
        assert r.status_code == 403

    def test_admin_retrain_skipped_when_no_followup(self, api_client, admin_token):
        r = api_client.post(
            "/admin/retrain",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 200
        # ไม่มี confirmed follow-up → status=skipped
        assert r.json()["status"] in ("skipped", "success")
