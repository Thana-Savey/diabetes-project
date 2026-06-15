"""
diseases.py — Disease Registry
เพิ่มโรคใหม่ได้โดยเพิ่ม entry ใน DISEASES dict
"""
import os

_DIR = os.path.dirname(os.path.abspath(__file__))


DISEASES: dict[str, dict] = {

    # ── Diabetes ───────────────────────────────────────────────
    "diabetes": {
        "name_en":     "Diabetes",
        "name_th":     "เบาหวาน",
        "icon":        "🩸",
        "csv":         os.path.join(_DIR, "diabetes.csv"),
        "target":      "Outcome",
        "features": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigree", "Age",
        ],
        "cols_to_fix": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "input_fields": [
            {"key": "Pregnancies",      "label": "จำนวนการตั้งครรภ์",       "min": 0,   "max": 20,   "default": 1,    "step": 1,    "fmt": "%d"},
            {"key": "Glucose",          "label": "ระดับน้ำตาล (mg/dL)",     "min": 0,   "max": 300,  "default": 120,  "step": 1,    "fmt": "%d"},
            {"key": "BloodPressure",    "label": "ความดันโลหิต (mm Hg)",    "min": 0,   "max": 200,  "default": 70,   "step": 1,    "fmt": "%d"},
            {"key": "SkinThickness",    "label": "ความหนาผิวหนัง (mm)",     "min": 0,   "max": 100,  "default": 20,   "step": 1,    "fmt": "%d"},
            {"key": "Insulin",          "label": "Insulin (μU/mL)",          "min": 0,   "max": 900,  "default": 0,    "step": 1,    "fmt": "%d"},
            {"key": "BMI",              "label": "BMI (kg/m²)",              "min": 0.0, "max": 70.0, "default": 28.0, "step": 0.1,  "fmt": "%.1f"},
            {"key": "DiabetesPedigree", "label": "Diabetes Pedigree",        "min": 0.0, "max": 3.0,  "default": 0.5,  "step": 0.01, "fmt": "%.3f"},
            {"key": "Age",              "label": "อายุ (ปี)",                "min": 1,   "max": 120,  "default": 35,   "step": 1,    "fmt": "%d"},
        ],
        "description": "ประเมินความเสี่ยงโรคเบาหวานจากค่าทางการแพทย์ (Pima Indians Dataset)",
        "risk_labels": {0: "ไม่เสี่ยงเบาหวาน", 1: "เสี่ยงเป็นเบาหวาน"},
    },

    # ── Heart Disease ──────────────────────────────────────────
    "heart": {
        "name_en":     "Heart Disease",
        "name_th":     "โรคหัวใจ",
        "icon":        "❤️",
        "csv":         os.path.join(_DIR, "data", "heart.csv"),
        "target":      "target",
        "features": [
            "age", "sex", "cp", "trestbps", "chol",
            "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal",
        ],
        "cols_to_fix": [],   # Heart dataset ไม่มีค่า 0 ที่ต้อง impute
        "input_fields": [
            {"key": "age",      "label": "อายุ (ปี)",                    "min": 1,   "max": 120,  "default": 50,   "step": 1,    "fmt": "%d"},
            {"key": "sex",      "label": "เพศ (1=ชาย, 0=หญิง)",         "min": 0,   "max": 1,    "default": 1,    "step": 1,    "fmt": "%d"},
            {"key": "cp",       "label": "อาการเจ็บหน้าอก (0–3)",       "min": 0,   "max": 3,    "default": 0,    "step": 1,    "fmt": "%d"},
            {"key": "trestbps", "label": "ความดัน Resting (mm Hg)",      "min": 80,  "max": 250,  "default": 130,  "step": 1,    "fmt": "%d"},
            {"key": "chol",     "label": "Cholesterol (mg/dL)",           "min": 100, "max": 600,  "default": 240,  "step": 1,    "fmt": "%d"},
            {"key": "fbs",      "label": "น้ำตาลขณะอดอาหาร >120 (1=ใช่)","min": 0,   "max": 1,    "default": 0,    "step": 1,    "fmt": "%d"},
            {"key": "restecg",  "label": "ผล ECG ขณะพัก (0–2)",          "min": 0,   "max": 2,    "default": 0,    "step": 1,    "fmt": "%d"},
            {"key": "thalach",  "label": "อัตราหัวใจสูงสุด (bpm)",       "min": 60,  "max": 220,  "default": 150,  "step": 1,    "fmt": "%d"},
            {"key": "exang",    "label": "เจ็บหน้าอกเวลาออกกำลัง (1=ใช่)","min": 0,  "max": 1,    "default": 0,    "step": 1,    "fmt": "%d"},
            {"key": "oldpeak",  "label": "ST Depression",                 "min": 0.0, "max": 7.0,  "default": 1.0,  "step": 0.1,  "fmt": "%.1f"},
            {"key": "slope",    "label": "ความชัน ST Segment (0–2)",      "min": 0,   "max": 2,    "default": 1,    "step": 1,    "fmt": "%d"},
            {"key": "ca",       "label": "จำนวนหลอดเลือดหลัก (0–4)",     "min": 0,   "max": 4,    "default": 0,    "step": 1,    "fmt": "%d"},
            {"key": "thal",     "label": "Thalassemia (1=Normal,2=Fixed,3=Reversible)", "min": 0, "max": 3, "default": 2, "step": 1, "fmt": "%d"},
        ],
        "description": "ประเมินความเสี่ยงโรคหัวใจจากค่าตรวจทางคลินิก (Cleveland Heart Disease Dataset)",
        "risk_labels": {0: "ไม่พบความเสี่ยงโรคหัวใจ", 1: "เสี่ยงโรคหัวใจ"},
    },
}
