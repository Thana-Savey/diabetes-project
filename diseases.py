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

    # ── Stroke ────────────────────────────────────────────────
    "stroke": {
        "name_en":     "Stroke",
        "name_th":     "โรคหลอดเลือดสมอง",
        "icon":        "🧠",
        "csv":         os.path.join(_DIR, "data", "stroke.csv"),
        "target":      "target",
        "features": [
            "gender", "age", "hypertension", "heart_disease",
            "ever_married", "avg_glucose_level", "bmi", "smoking_status",
        ],
        "cols_to_fix": [],
        "input_fields": [
            {"key": "gender",            "label": "เพศ (0=หญิง, 1=ชาย, 2=อื่นๆ)",              "min": 0,    "max": 2,    "default": 1,    "step": 1,   "fmt": "%d"},
            {"key": "age",               "label": "อายุ (ปี)",                                   "min": 1,    "max": 100,  "default": 50,   "step": 1,   "fmt": "%d"},
            {"key": "hypertension",      "label": "ความดันโลหิตสูง (0=ไม่มี, 1=มี)",            "min": 0,    "max": 1,    "default": 0,    "step": 1,   "fmt": "%d"},
            {"key": "heart_disease",     "label": "โรคหัวใจ (0=ไม่มี, 1=มี)",                  "min": 0,    "max": 1,    "default": 0,    "step": 1,   "fmt": "%d"},
            {"key": "ever_married",      "label": "เคยแต่งงาน (0=ไม่, 1=เคย)",                 "min": 0,    "max": 1,    "default": 1,    "step": 1,   "fmt": "%d"},
            {"key": "avg_glucose_level", "label": "ระดับน้ำตาลเฉลี่ย (mg/dL)",                 "min": 50.0, "max": 300.0,"default": 106.0,"step": 0.1, "fmt": "%.1f"},
            {"key": "bmi",               "label": "BMI (kg/m²)",                                 "min": 10.0, "max": 60.0, "default": 28.9, "step": 0.1, "fmt": "%.1f"},
            {"key": "smoking_status",    "label": "สถานะสูบบุหรี่ (0=ไม่เคย,1=เคย,2=สูบ,3=ไม่ทราบ)", "min": 0, "max": 3, "default": 0, "step": 1, "fmt": "%d"},
        ],
        "description": "ประเมินความเสี่ยงโรคหลอดเลือดสมองจากปัจจัยสุขภาพพื้นฐาน (Kaggle Stroke Prediction, 5,110 rows)",
        "risk_labels": {0: "ไม่พบความเสี่ยงโรคหลอดเลือดสมอง", 1: "เสี่ยงโรคหลอดเลือดสมอง"},
    },

    # ── Breast Cancer ─────────────────────────────────────────
    "breast_cancer": {
        "name_en":     "Breast Cancer",
        "name_th":     "มะเร็งเต้านม",
        "icon":        "🎗️",
        "csv":         os.path.join(_DIR, "data", "breast_cancer.csv"),
        "target":      "target",
        "features": [
            "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
            "mean_smoothness", "mean_compactness", "mean_concavity",
            "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
        ],
        "cols_to_fix": [],
        "input_fields": [
            {"key": "mean_radius",           "label": "รัศมีเฉลี่ย (Mean Radius)",          "min": 5.0,   "max": 30.0,  "default": 14.1, "step": 0.1,   "fmt": "%.1f"},
            {"key": "mean_texture",          "label": "เนื้อเยื่อเฉลี่ย (Mean Texture)",    "min": 5.0,   "max": 40.0,  "default": 19.3, "step": 0.1,   "fmt": "%.1f"},
            {"key": "mean_perimeter",        "label": "เส้นรอบวงเฉลี่ย (mm)",               "min": 40.0,  "max": 200.0, "default": 92.0, "step": 0.1,   "fmt": "%.1f"},
            {"key": "mean_area",             "label": "พื้นที่เฉลี่ย (Mean Area)",           "min": 100.0, "max": 2500.0,"default": 655.0,"step": 1.0,   "fmt": "%.1f"},
            {"key": "mean_smoothness",       "label": "ความเรียบเฉลี่ย",                     "min": 0.05,  "max": 0.20,  "default": 0.096,"step": 0.001, "fmt": "%.3f"},
            {"key": "mean_compactness",      "label": "ความแน่นเฉลี่ย",                      "min": 0.01,  "max": 0.40,  "default": 0.104,"step": 0.001, "fmt": "%.3f"},
            {"key": "mean_concavity",        "label": "ความเว้าเฉลี่ย",                      "min": 0.0,   "max": 0.50,  "default": 0.089,"step": 0.001, "fmt": "%.3f"},
            {"key": "mean_concave_points",   "label": "จุดเว้าเฉลี่ย",                       "min": 0.0,   "max": 0.25,  "default": 0.049,"step": 0.001, "fmt": "%.3f"},
            {"key": "mean_symmetry",         "label": "ความสมมาตรเฉลี่ย",                    "min": 0.10,  "max": 0.35,  "default": 0.181,"step": 0.001, "fmt": "%.3f"},
            {"key": "mean_fractal_dimension","label": "Fractal Dimension เฉลี่ย",             "min": 0.04,  "max": 0.10,  "default": 0.063,"step": 0.001, "fmt": "%.3f"},
        ],
        "description": "ตรวจวิเคราะห์ความเสี่ยงมะเร็งเต้านมจากผลตรวจ FNA (Wisconsin Breast Cancer Dataset, 569 rows)",
        "risk_labels": {0: "ก้อนเนื้อไม่ร้าย (Benign)", 1: "เสี่ยงมะเร็งเต้านม (Malignant)"},
    },

    # ── Liver Disease (ILPD) ──────────────────────────────────
    "liver": {
        "name_en":     "Liver Disease",
        "name_th":     "โรคตับ",
        "icon":        "🫀",
        "csv":         os.path.join(_DIR, "data", "liver.csv"),
        "target":      "target",
        "features": [
            "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
            "Alkaline_Phosphotase", "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase", "Total_Proteins",
            "Albumin", "Albumin_and_Globulin_Ratio",
        ],
        "cols_to_fix": [],
        "input_fields": [
            {"key": "Age",                         "label": "อายุ (ปี)",                        "min": 1,    "max": 90,   "default": 40,   "step": 1,   "fmt": "%d"},
            {"key": "Gender",                      "label": "เพศ (0=หญิง, 1=ชาย)",             "min": 0,    "max": 1,    "default": 1,    "step": 1,   "fmt": "%d"},
            {"key": "Total_Bilirubin",             "label": "Total Bilirubin (mg/dL)",          "min": 0.1,  "max": 75.0, "default": 3.3,  "step": 0.1, "fmt": "%.1f"},
            {"key": "Direct_Bilirubin",            "label": "Direct Bilirubin (mg/dL)",         "min": 0.1,  "max": 20.0, "default": 1.5,  "step": 0.1, "fmt": "%.1f"},
            {"key": "Alkaline_Phosphotase",        "label": "Alkaline Phosphotase (IU/L)",      "min": 50,   "max": 2500, "default": 290,  "step": 1,   "fmt": "%d"},
            {"key": "Alamine_Aminotransferase",    "label": "ALT / SGPT (IU/L)",               "min": 5,    "max": 2000, "default": 80,   "step": 1,   "fmt": "%d"},
            {"key": "Aspartate_Aminotransferase",  "label": "AST / SGOT (IU/L)",               "min": 5,    "max": 5000, "default": 80,   "step": 1,   "fmt": "%d"},
            {"key": "Total_Proteins",              "label": "Total Proteins (g/dL)",            "min": 1.0,  "max": 10.0, "default": 6.8,  "step": 0.1, "fmt": "%.1f"},
            {"key": "Albumin",                     "label": "Albumin (g/dL)",                   "min": 0.5,  "max": 6.0,  "default": 3.1,  "step": 0.1, "fmt": "%.1f"},
            {"key": "Albumin_and_Globulin_Ratio",  "label": "Albumin/Globulin Ratio",           "min": 0.1,  "max": 3.0,  "default": 0.9,  "step": 0.01,"fmt": "%.2f"},
        ],
        "description": "ประเมินความเสี่ยงโรคตับจากค่าตรวจเลือด (Indian Liver Patient Dataset, 583 rows)",
        "risk_labels": {0: "ไม่พบปัญหาตับ", 1: "เสี่ยงโรคตับ"},
    },

    # ── Chronic Kidney Disease (CKD) ──────────────────────────
    "ckd": {
        "name_en":     "Chronic Kidney Disease",
        "name_th":     "โรคไตเรื้อรัง",
        "icon":        "🫘",
        "csv":         os.path.join(_DIR, "data", "ckd.csv"),
        "target":      "target",
        "features": [
            "age", "bp", "bgr", "bu", "sc", "sod", "pot",
            "hemo", "pcv", "wbcc", "rbcc",
            "htn", "dm", "cad", "ane",
        ],
        "cols_to_fix": [],
        "input_fields": [
            {"key": "age",  "label": "อายุ (ปี)",                          "min": 1,    "max": 100,  "default": 50,   "step": 1,    "fmt": "%d"},
            {"key": "bp",   "label": "ความดันโลหิต Diastolic (mm Hg)",     "min": 40,   "max": 200,  "default": 80,   "step": 1,    "fmt": "%d"},
            {"key": "bgr",  "label": "Blood Glucose Random (mg/dL)",        "min": 50,   "max": 500,  "default": 120,  "step": 1,    "fmt": "%d"},
            {"key": "bu",   "label": "Blood Urea (mg/dL)",                  "min": 5,    "max": 400,  "default": 35,   "step": 1,    "fmt": "%d"},
            {"key": "sc",   "label": "Serum Creatinine (mg/dL)",            "min": 0.4,  "max": 15.0, "default": 1.2,  "step": 0.1,  "fmt": "%.1f"},
            {"key": "sod",  "label": "Sodium (mEq/L)",                      "min": 100,  "max": 160,  "default": 137,  "step": 1,    "fmt": "%d"},
            {"key": "pot",  "label": "Potassium (mEq/L)",                   "min": 2.0,  "max": 10.0, "default": 4.5,  "step": 0.1,  "fmt": "%.1f"},
            {"key": "hemo", "label": "Hemoglobin (g/dL)",                   "min": 3.0,  "max": 18.0, "default": 12.5, "step": 0.1,  "fmt": "%.1f"},
            {"key": "pcv",  "label": "Packed Cell Volume (%)",              "min": 9,    "max": 54,   "default": 38,   "step": 1,    "fmt": "%d"},
            {"key": "wbcc", "label": "White Blood Cell Count (cells/cumm)", "min": 2000, "max": 26400,"default": 8000, "step": 100,  "fmt": "%d"},
            {"key": "rbcc", "label": "Red Blood Cell Count (millions/cmm)", "min": 2.0,  "max": 7.0,  "default": 4.5,  "step": 0.1,  "fmt": "%.1f"},
            {"key": "htn",  "label": "ความดันโลหิตสูง (0=ไม่มี, 1=มี)",   "min": 0,    "max": 1,    "default": 0,    "step": 1,    "fmt": "%d"},
            {"key": "dm",   "label": "เบาหวาน (0=ไม่มี, 1=มี)",            "min": 0,    "max": 1,    "default": 0,    "step": 1,    "fmt": "%d"},
            {"key": "cad",  "label": "โรคหลอดเลือดหัวใจ (0=ไม่มี, 1=มี)", "min": 0,    "max": 1,    "default": 0,    "step": 1,    "fmt": "%d"},
            {"key": "ane",  "label": "โลหิตจาง (0=ไม่มี, 1=มี)",           "min": 0,    "max": 1,    "default": 0,    "step": 1,    "fmt": "%d"},
        ],
        "description": "ประเมินความเสี่ยงโรคไตเรื้อรังจากค่าตรวจเลือดและปัสสาวะ (UCI CKD Dataset, 400 rows)",
        "risk_labels": {0: "ไม่พบความเสี่ยงโรคไต", 1: "เสี่ยงโรคไตเรื้อรัง (CKD)"},
    },
}
