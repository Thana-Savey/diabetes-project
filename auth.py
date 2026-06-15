"""
auth.py — Authentication สำหรับ Diabetes API
- API Key  : สำหรับ HIS / ระบบโรงพยาบาลเรียก /predict
- JWT      : สำหรับแพทย์/พยาบาล login ผ่าน /token
"""

import os
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── Config (ในระบบจริงใช้ environment variable) ───────────────
SECRET_KEY  = os.getenv("SECRET_KEY", "b066b264e9aa75748bf95d3db16fddb9543a0c85bc5e4aad261777fe16388d8c")
ALGORITHM   = "HS256"
TOKEN_EXPIRE_HOURS = 8   # token หมดอายุหลัง 8 ชั่วโมง (1 กะงาน)

VALID_API_KEYS = set(
    os.getenv("API_KEYS", "his-system-key-abc123,lab-system-key-xyz789").split(",")
)

# ── ผู้ใช้งานระบบ (จริงๆ ควรเก็บใน database) ─────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

USERS_DB = {
    "dr.somchai": {
        "username":   "dr.somchai",
        "full_name":  "นพ.สมชาย ใจดี",
        "role":       "doctor",
        "hashed_pw":  pwd_context.hash("doctor1234"),
    },
    "nurse.malee": {
        "username":   "nurse.malee",
        "full_name":  "พยาบาลมาลี สุขใส",
        "role":       "nurse",
        "hashed_pw":  pwd_context.hash("nurse1234"),
    },
    "admin": {
        "username":   "admin",
        "full_name":  "ผู้ดูแลระบบ",
        "role":       "admin",
        "hashed_pw":  pwd_context.hash("admin1234"),
    },
}

# ── Schemas ───────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int
    full_name:    str
    role:         str

class CurrentUser(BaseModel):
    username:  str
    full_name: str
    role:      str


# ── API Key Auth (สำหรับ HIS) ─────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key and api_key in VALID_API_KEYS:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API Key",
    )


# ── JWT Auth (สำหรับแพทย์/พยาบาล) ────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)

def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> CurrentUser:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="กรุณา Login ก่อน",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username or username not in USERS_DB:
            raise ValueError
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token ไม่ถูกต้องหรือหมดอายุ",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = USERS_DB[username]
    return CurrentUser(username=username, full_name=user["full_name"], role=user["role"])


# ── Combined: ยอมรับทั้ง API Key และ JWT ──────────────────────
def verify_any(
    api_key:     str                           = Security(api_key_header),
    credentials: HTTPAuthorizationCredentials  = Security(bearer_scheme),
) -> dict:
    """
    HIS เรียกด้วย X-API-Key header
    แพทย์เรียกด้วย Bearer JWT token
    อย่างใดอย่างหนึ่งถูกต้อง → ผ่าน
    คืน dict: {"type": "api_key"|"jwt", "id": ..., "role": ...}
    """
    if api_key and api_key in VALID_API_KEYS:
        return {"type": "api_key", "id": api_key[:8] + "...", "role": "api_key"}

    if credentials:
        try:
            payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username and username in USERS_DB:
                user = USERS_DB[username]
                return {"type": "jwt", "id": username, "role": user["role"]}
        except JWTError:
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="กรุณาใส่ X-API-Key หรือ Bearer token",
    )


# ── RBAC ───────────────────────────────────────────────────────
# permission → roles ที่ได้รับอนุญาต
PERMISSION_MAP: dict[str, set[str]] = {
    "predict":          {"api_key", "nurse", "doctor", "admin"},
    "followup:read":    {"nurse", "doctor", "admin"},
    "followup:write":   {"nurse", "doctor", "admin"},
    "audit:read":       {"doctor", "admin"},
    "model:read":       {"doctor", "admin"},
    "model:retrain":    {"admin"},
}

ROLE_THAI = {
    "api_key": "ระบบ HIS",
    "nurse":   "พยาบาล",
    "doctor":  "แพทย์",
    "admin":   "ผู้ดูแลระบบ",
}


def require(permission: str):
    """
    Dependency factory สำหรับ RBAC
    ใช้: caller = Security(require("followup:write"))

    ตรวจทั้ง API Key และ JWT แล้ว map role → permission
    """
    def _checker(
        api_key:     str                           = Security(api_key_header),
        credentials: HTTPAuthorizationCredentials  = Security(bearer_scheme),
    ) -> dict:
        caller = verify_any(api_key, credentials)
        role   = caller.get("role", "api_key")

        allowed = PERMISSION_MAP.get(permission, set())
        if role not in allowed:
            role_th   = ROLE_THAI.get(role, role)
            allowed_th = " / ".join(ROLE_THAI.get(r, r) for r in sorted(allowed) if r != "api_key")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"สิทธิ์ไม่เพียงพอ — {role_th} ไม่สามารถเข้าถึง [{permission}] ได้ "
                       f"(ต้องเป็น: {allowed_th})",
            )
        return caller

    # ตั้งชื่อให้ FastAPI แสดงใน /docs ได้ถูกต้อง
    _checker.__name__ = f"require_{permission.replace(':', '_')}"
    return _checker
