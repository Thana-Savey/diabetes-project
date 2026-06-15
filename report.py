"""
report.py — PDF Report Generator สำหรับ Diabetes Risk Predictor
ใช้ fpdf2 + Sarabun font (รองรับภาษาไทย)
"""

import io
import os
import tempfile
import urllib.request
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

matplotlib.use("Agg")

# ── Font setup ─────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR  = os.path.join(_DIR, "fonts")
FONT_REG   = os.path.join(FONTS_DIR, "Sarabun-Regular.ttf")
FONT_BOLD  = os.path.join(FONTS_DIR, "Sarabun-Bold.ttf")

_FONT_URLS = {
    "Sarabun-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf",
    "Sarabun-Bold.ttf":    "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Bold.ttf",
}


def _ensure_fonts() -> bool:
    os.makedirs(FONTS_DIR, exist_ok=True)
    for fname, url in _FONT_URLS.items():
        fpath = os.path.join(FONTS_DIR, fname)
        if not os.path.exists(fpath):
            try:
                urllib.request.urlretrieve(url, fpath)
            except Exception:
                return False
    return os.path.exists(FONT_REG) and os.path.exists(FONT_BOLD)


# ── Color palette ──────────────────────────────────────────────
C_HIGH   = (216, 90, 48)    # แดง — สูง
C_MED    = (245, 166, 35)   # เหลือง — กลาง
C_LOW    = (29, 158, 117)   # เขียว — ต่ำ
C_HEADER = (44, 62, 80)     # navy — header bg
C_LIGHT  = (245, 246, 250)  # light gray — row bg
C_WHITE  = (255, 255, 255)
C_TEXT   = (30, 30, 30)


def _risk_color(risk_level: str) -> tuple:
    return {"สูง": C_HIGH, "กลาง": C_MED, "ต่ำ": C_LOW}.get(risk_level, C_MED)


# ── SHAP chart → PNG bytes ─────────────────────────────────────
def _shap_png(shap_values, feature_names: list, feature_vals: list) -> bytes:
    order       = np.argsort(np.abs(shap_values))[::-1]
    labels      = [f"{feature_names[i]} = {feature_vals[i]:.1f}" for i in order]
    vals_sorted = shap_values[order]
    colors      = ["#D85A30" if v > 0 else "#1D9E75" for v in vals_sorted]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.barh(labels[::-1], vals_sorted[::-1], color=colors[::-1], alpha=0.88)
    for bar, val in zip(bars, vals_sorted[::-1]):
        offset = 0.002 if val >= 0 else -0.002
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value", fontsize=8)
    ax.set_title("Feature Importance (Red = increases risk | Green = decreases risk)",
                 fontsize=8.5, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── PDF class ──────────────────────────────────────────────────
class _ReportPDF(FPDF):
    def __init__(self, has_thai_font: bool):
        super().__init__()
        self._thai = has_thai_font

    def _f(self, style="", size=11):
        name = "Sarabun" if self._thai else "Helvetica"
        self.set_font(name, style, size)

    def header(self):
        self._f("B", 15)
        self.set_text_color(*C_HEADER)
        self.cell(0, 10, "Diabetes Risk Assessment Report", align="C")
        self.ln(6)
        self._f("", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f"ระบบประเมินความเสี่ยงโรคเบาหวาน | LightGBM + Platt Calibration", align="C")
        self.ln(3)
        self.set_draw_color(*C_HEADER)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)
        self.set_text_color(*C_TEXT)

    def footer(self):
        self.set_y(-22)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
        self._f("I", 7.5)
        self.set_text_color(130, 130, 130)
        self.multi_cell(
            0, 4,
            "ผลนี้เป็นเพียงการประเมินเบื้องต้นจากโมเดล AI ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์ได้  "
            "| This report is AI-generated and must not replace professional medical diagnosis.",
            align="C",
        )


def _section_title(pdf: _ReportPDF, title: str):
    pdf._f("B", 11)
    pdf.set_fill_color(*C_HEADER)
    pdf.set_text_color(*C_WHITE)
    pdf.cell(0, 7, f"  {title}", fill=True)
    pdf.ln(4)
    pdf.set_text_color(*C_TEXT)


def _two_col(pdf: _ReportPDF, label: str, value: str, w1=55, w2=85, fill=False):
    pdf._f("B", 10)
    pdf.set_fill_color(*C_LIGHT)
    pdf.cell(w1, 7, label, border=0, fill=fill)
    pdf._f("", 10)
    pdf.cell(w2, 7, str(value), border=0, fill=fill)
    pdf.ln()


# ── Public API ─────────────────────────────────────────────────
def generate_report(
    patient: dict,
    shap_values=None,
    feature_names: list | None = None,
    followup_records: list | None = None,
    printed_by: str = "ระบบ",
) -> bytes:
    """
    สร้าง PDF report สำหรับคนไข้ 1 ราย
    คืน bytes ของ PDF ที่พร้อม download

    patient: dict จาก get_patient_by_id()
    shap_values: numpy array [n_features] (optional)
    feature_names: list ชื่อ features (optional)
    followup_records: list of dict จาก get_followups_by_patient().to_dict("records")
    """
    has_font = _ensure_fonts()

    pdf = _ReportPDF(has_thai_font=has_font)
    if has_font:
        pdf.add_font("Sarabun",  "", FONT_REG,  uni=True)
        pdf.add_font("Sarabun", "B", FONT_BOLD, uni=True)
        pdf.add_font("Sarabun", "I", FONT_REG,  uni=True)

    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()

    # ── ข้อมูลพื้นฐาน ──────────────────────────────────────────
    _section_title(pdf, "ข้อมูลผู้ป่วย  |  Patient Information")

    col1_x = 12
    col2_x = 110
    y_start = pdf.get_y()

    pdf.set_xy(col1_x, y_start)
    _two_col(pdf, "HN:", patient.get("hn") or "-")
    _two_col(pdf, "ชื่อ-นามสกุล:", patient.get("name") or "-")
    _two_col(pdf, "อายุ:", f"{patient.get('age') or '-'} ปี")
    _two_col(pdf, "วันที่ประเมิน:",
             str(patient.get("created_at", ""))[:19] if patient.get("created_at") else "-")
    _two_col(pdf, "พิมพ์โดย:", printed_by)
    _two_col(pdf, "หมายเหตุ:", patient.get("note") or "-")

    pdf.ln(4)

    # ── ผลประเมิน ──────────────────────────────────────────────
    _section_title(pdf, "ผลการประเมินความเสี่ยง  |  Risk Assessment Result")

    risk_level = patient.get("risk_level", "กลาง")
    prob       = patient.get("risk_prob", 0.0)
    prediction = patient.get("prediction", 0)
    rc         = _risk_color(risk_level)

    # กล่องสี
    pdf.set_fill_color(*rc)
    pdf.set_text_color(*C_WHITE)
    pdf._f("B", 13)
    verdict = "[!] เสี่ยงเป็นเบาหวาน" if prediction == 1 else "[OK] ไม่พบความเสี่ยงสูง"
    pdf.cell(0, 14, verdict, fill=True, align="C")
    pdf.ln(3)
    pdf.set_text_color(*C_TEXT)

    # ตาราง 3 ช่อง
    pdf._f("B", 11)
    pdf.set_fill_color(*C_LIGHT)
    w = 62
    pdf.cell(w, 8, "ความน่าจะเป็น", border=1, fill=True, align="C")
    pdf.cell(w, 8, "ระดับความเสี่ยง", border=1, fill=True, align="C")
    pdf.cell(w, 8, "ผลการทำนาย", border=1, fill=True, align="C")
    pdf.ln()

    pdf._f("B", 14)
    pdf.set_text_color(*rc)
    pdf.cell(w, 10, f"{prob:.1%}", border=1, align="C")
    pdf.cell(w, 10, risk_level, border=1, align="C")
    pdf.set_text_color(*C_TEXT)
    pdf._f("", 11)
    pdf.cell(w, 10, "เสี่ยง (1)" if prediction == 1 else "ไม่เสี่ยง (0)", border=1, align="C")
    pdf.ln(6)
    pdf.set_text_color(*C_TEXT)

    # ── ค่าทางการแพทย์ ─────────────────────────────────────────
    _section_title(pdf, "ค่าทางการแพทย์  |  Medical Values")

    headers = ["Feature", "ค่า", "Feature", "ค่า"]
    col_w   = [55, 35, 55, 35]

    pdf._f("B", 10)
    pdf.set_fill_color(*C_HEADER)
    pdf.set_text_color(*C_WHITE)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(*C_TEXT)

    fields = [
        ("Pregnancies",      patient.get("pregnancies")),
        ("Glucose (mg/dL)",  patient.get("glucose")),
        ("Blood Pressure",   patient.get("blood_pressure")),
        ("Skin Thickness",   patient.get("skin_thickness")),
        ("Insulin (μU/mL)",  patient.get("insulin")),
        ("BMI",              patient.get("bmi")),
        ("Diabetes Pedigree",patient.get("diabetes_pedigree")),
        ("Age (ปี)",         patient.get("age")),
    ]

    for i in range(0, len(fields), 2):
        fill = (i // 2) % 2 == 0
        pdf.set_fill_color(*C_LIGHT if fill else C_WHITE)
        l1, v1 = fields[i]
        l2, v2 = fields[i + 1] if i + 1 < len(fields) else ("", "")
        pdf._f("", 10)
        pdf.cell(col_w[0], 7, l1, border=1, fill=fill)
        pdf._f("B", 10)
        pdf.cell(col_w[1], 7, f"{v1:.1f}" if isinstance(v1, float) else str(v1 or "-"),
                 border=1, fill=fill, align="C")
        pdf._f("", 10)
        pdf.cell(col_w[2], 7, l2, border=1, fill=fill)
        pdf._f("B", 10)
        pdf.cell(col_w[3], 7, f"{v2:.1f}" if isinstance(v2, float) else str(v2 or "-"),
                 border=1, fill=fill, align="C")
        pdf.ln()
    pdf.ln(4)

    # ── SHAP Chart ─────────────────────────────────────────────
    if shap_values is not None and feature_names is not None:
        _section_title(pdf, "SHAP Feature Importance")

        feat_vals = [
            patient.get(f.lower().replace("diabetespedigree", "diabetes_pedigree")
                        .replace("bloodpressure", "blood_pressure")
                        .replace("skinthickness", "skin_thickness"), 0)
            for f in feature_names
        ]
        png_bytes = _shap_png(shap_values, feature_names, feat_vals)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png_bytes)
            tmp_path = tmp.name

        try:
            pdf.image(tmp_path, x=12, w=186)
        finally:
            os.unlink(tmp_path)

        pdf.ln(4)

    # ── Follow-up History ──────────────────────────────────────
    if followup_records:
        _section_title(pdf, "ประวัติการติดตาม  |  Follow-up History")

        fu_headers = ["วันนัด", "วันจริง", "สถานะ", "ผลตรวจ", "Glucose", "BMI", "หมายเหตุ"]
        fu_w       = [26, 26, 24, 20, 22, 18, 44]

        pdf._f("B", 8.5)
        pdf.set_fill_color(*C_HEADER)
        pdf.set_text_color(*C_WHITE)
        for h, w in zip(fu_headers, fu_w):
            pdf.cell(w, 7, h, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_text_color(*C_TEXT)

        for i, fu in enumerate(followup_records):
            fill = i % 2 == 0
            pdf.set_fill_color(*C_LIGHT if fill else C_WHITE)
            pdf._f("", 8)
            outcome = {0: "ไม่เป็น", 1: "เป็น", None: "-"}.get(fu.get("actual_outcome"))
            row = [
                str(fu.get("scheduled_date") or "-")[:10],
                str(fu.get("actual_date") or "-")[:10],
                str(fu.get("status") or "-"),
                outcome,
                f"{fu['glucose_new']:.0f}" if fu.get("glucose_new") else "-",
                f"{fu['bmi_new']:.1f}" if fu.get("bmi_new") else "-",
                str(fu.get("note") or "")[:30],
            ]
            for val, w in zip(row, fu_w):
                pdf.cell(w, 6.5, val, border=1, fill=fill, align="C")
            pdf.ln()
        pdf.ln(4)

    # ── ลายเซ็นแพทย์ ──────────────────────────────────────────
    pdf.ln(4)
    pdf._f("", 10)
    pdf.cell(90, 6, "ลายเซ็นแพทย์: _______________________")
    pdf.cell(90, 6, f"วันที่พิมพ์: {datetime.now().strftime('%d/%m/%Y  %H:%M')}", align="R")
    pdf.ln(5)
    pdf._f("", 10)
    pdf.cell(90, 6, "ชื่อแพทย์: _______________________")

    # ── Output ────────────────────────────────────────────────
    return bytes(pdf.output())
