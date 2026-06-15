"""
test_auth.py — ทดสอบ RBAC permission matrix และ authentication
"""
import pytest
from auth import PERMISSION_MAP, PERMISSION_MAP, ROLE_THAI


# ── Permission Matrix ──────────────────────────────────────────
class TestPermissionMatrix:
    def test_all_permissions_defined(self):
        expected = {
            "predict", "followup:read", "followup:write",
            "audit:read", "model:read", "model:retrain",
        }
        assert set(PERMISSION_MAP.keys()) == expected

    def test_admin_has_all_permissions(self):
        for perm, roles in PERMISSION_MAP.items():
            assert "admin" in roles, f"admin ขาด permission: {perm}"

    def test_only_admin_can_retrain(self):
        assert PERMISSION_MAP["model:retrain"] == {"admin"}

    def test_nurse_cannot_read_audit(self):
        assert "nurse" not in PERMISSION_MAP["audit:read"]

    def test_nurse_cannot_read_model(self):
        assert "nurse" not in PERMISSION_MAP["model:read"]

    def test_api_key_cannot_access_followup(self):
        assert "api_key" not in PERMISSION_MAP["followup:read"]
        assert "api_key" not in PERMISSION_MAP["followup:write"]

    def test_api_key_can_predict(self):
        assert "api_key" in PERMISSION_MAP["predict"]

    def test_doctor_can_read_audit(self):
        assert "doctor" in PERMISSION_MAP["audit:read"]

    def test_nurse_can_write_followup(self):
        assert "nurse" in PERMISSION_MAP["followup:write"]

    def test_role_thai_covers_all_roles(self):
        for role in ["api_key", "nurse", "doctor", "admin"]:
            assert role in ROLE_THAI


# ── Token / Login ──────────────────────────────────────────────
class TestLogin:
    def test_doctor_login_success(self, api_client):
        r = api_client.post("/token", json={"username": "dr.somchai", "password": "doctor1234"})
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data
        assert data["role"] == "doctor"

    def test_nurse_login_success(self, api_client):
        r = api_client.post("/token", json={"username": "nurse.malee", "password": "nurse1234"})
        assert r.status_code == 200
        assert r.json()["role"] == "nurse"

    def test_admin_login_success(self, api_client):
        r = api_client.post("/token", json={"username": "admin", "password": "admin1234"})
        assert r.status_code == 200
        assert r.json()["role"] == "admin"

    def test_wrong_password_rejected(self, api_client):
        r = api_client.post("/token", json={"username": "dr.somchai", "password": "wrong"})
        assert r.status_code == 401

    def test_unknown_user_rejected(self, api_client):
        r = api_client.post("/token", json={"username": "hacker", "password": "pass"})
        assert r.status_code == 401


# ── /me endpoint ───────────────────────────────────────────────
class TestWhoAmI:
    def test_doctor_sees_correct_permissions(self, api_client, doctor_token):
        r = api_client.get("/me", headers={"Authorization": f"Bearer {doctor_token}"})
        assert r.status_code == 200
        data = r.json()
        assert data["role"] == "doctor"
        assert "predict" in data["permissions"]
        assert "audit:read" in data["permissions"]
        assert "model:retrain" not in data["permissions"]

    def test_nurse_sees_correct_permissions(self, api_client, nurse_token):
        r = api_client.get("/me", headers={"Authorization": f"Bearer {nurse_token}"})
        assert r.status_code == 200
        data = r.json()
        assert data["role"] == "nurse"
        assert "followup:write" in data["permissions"]
        assert "audit:read" not in data["permissions"]

    def test_api_key_sees_permissions(self, api_client):
        r = api_client.get("/me", headers={"X-API-Key": "test-api-key-123"})
        assert r.status_code == 200
        data = r.json()
        assert data["role"] == "api_key"
        assert "predict" in data["permissions"]
        assert "followup:read" not in data["permissions"]

    def test_no_auth_rejected(self, api_client):
        r = api_client.get("/me")
        assert r.status_code == 401
