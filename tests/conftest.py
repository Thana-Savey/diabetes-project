"""
conftest.py — pytest fixtures และ environment setup
ตั้ง DATABASE_URL เป็น in-memory SQLite ก่อน import api/database
"""
import os
import pytest

# ต้อง set ก่อน import ใดๆ
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_temp.sqlite")
os.environ.setdefault("SECRET_KEY",   "test-secret-key-not-for-production")
os.environ.setdefault("API_KEYS",     "test-api-key-123")


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    from database import init_db, seed_demo_data
    init_db()
    seed_demo_data()
    yield
    # ลบ test DB หลังเสร็จ
    if os.path.exists("test_temp.sqlite"):
        os.remove("test_temp.sqlite")


@pytest.fixture(scope="session")
def api_client():
    from fastapi.testclient import TestClient
    from api import app
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="session")
def doctor_token(api_client):
    r = api_client.post("/token", json={"username": "dr.somchai", "password": "doctor1234"})
    return r.json()["access_token"]


@pytest.fixture(scope="session")
def nurse_token(api_client):
    r = api_client.post("/token", json={"username": "nurse.malee", "password": "nurse1234"})
    return r.json()["access_token"]


@pytest.fixture(scope="session")
def admin_token(api_client):
    r = api_client.post("/token", json={"username": "admin", "password": "admin1234"})
    return r.json()["access_token"]


SAMPLE_PATIENT = {
    "hn": "HN-TEST",
    "name": "Test Patient",
    "age": 45,
    "disease": "diabetes",
    "pregnancies": 2,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree": 0.627,
}
