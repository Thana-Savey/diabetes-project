# ============================================================
#  Diabetes Project — Step 1: Load & Explore Data
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. โหลดข้อมูลตรงจาก URL (ไม่ต้องดาวน์โหลด) ──────────────
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies",        # จำนวนครั้งที่ตั้งครรภ์
    "Glucose",            # ระดับน้ำตาลในเลือด (mg/dL)
    "BloodPressure",      # ความดันโลหิต (mm Hg)
    "SkinThickness",      # ความหนาผิวหนัง (mm)
    "Insulin",            # ระดับ Insulin (μU/mL)
    "BMI",                # ดัชนีมวลกาย
    "DiabetesPedigree",   # ประวัติครอบครัว (score)
    "Age",                # อายุ
    "Outcome"             # 1 = เป็นเบาหวาน, 0 = ไม่เป็น
]

df = pd.read_csv(url, names=columns)

# ── 2. ดูภาพรวมข้อมูล ─────────────────────────────────────────
print("=" * 55)
print("  ขนาดข้อมูล:", df.shape)
print("=" * 55)
print(df.head(10).to_string())

print("\n── สถิติเบื้องต้น ──────────────────────────────────────")
print(df.describe().round(2).to_string())

print("\n── สัดส่วน Outcome ─────────────────────────────────────")
counts = df["Outcome"].value_counts()
pct    = df["Outcome"].value_counts(normalize=True) * 100
summary = pd.DataFrame({"จำนวน": counts, "เปอร์เซ็นต์": pct.round(1)})
summary.index = ["ไม่เป็นเบาหวาน (0)", "เป็นเบาหวาน (1)"]
print(summary.to_string())

print("\n── Missing values (ค่า 0 ที่ไม่สมเหตุสมผล) ───────────")
suspect_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
zero_counts = (df[suspect_cols] == 0).sum()
zero_pct    = ((df[suspect_cols] == 0).sum() / len(df) * 100).round(1)
missing_df  = pd.DataFrame({"จำนวน 0": zero_counts, "% ของทั้งหมด": zero_pct})
print(missing_df.to_string())
print("\n* ค่า 0 ใน Glucose/Insulin/BMI ฯลฯ = ข้อมูลหายจริง ไม่ใช่ศูนย์จริง")

# ── 3. Plot: Distribution + Class balance ─────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#F8F8F6")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38)

feature_cols = [c for c in df.columns if c != "Outcome"]
colors = ["#1D9E75", "#D85A30"]   # teal = ไม่เป็น, coral = เป็น

# Histogram แยกตาม Outcome สำหรับแต่ละ feature
for i, col in enumerate(feature_cols):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    for outcome, color in zip([0, 1], colors):
        subset = df[df["Outcome"] == outcome][col]
        ax.hist(subset, bins=22, alpha=0.65, color=color,
                label=["ไม่เป็น", "เป็น"][outcome], density=True)
    ax.set_title(col, fontsize=10, fontweight="500", pad=4)
    ax.set_xlabel("")
    ax.tick_params(labelsize=8)
    ax.set_facecolor("#F8F8F6")
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

# Legend + Class balance bar ใน subplot สุดท้าย
ax_bar = fig.add_subplot(gs[1, 3])
bars = ax_bar.bar(["ไม่เป็น\nเบาหวาน", "เป็น\nเบาหวาน"],
                  [counts[0], counts[1]],
                  color=colors, width=0.5)
for bar, val in zip(bars, [counts[0], counts[1]]):
    ax_bar.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 8, str(val),
                ha="center", va="bottom", fontsize=9, fontweight="500")
ax_bar.set_title("Class balance", fontsize=10, fontweight="500", pad=4)
ax_bar.set_facecolor("#F8F8F6")
ax_bar.tick_params(labelsize=8)
for spine in ax_bar.spines.values():
    spine.set_linewidth(0.4)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#1D9E75", alpha=0.7, label="ไม่เป็นเบาหวาน"),
                   Patch(facecolor="#D85A30", alpha=0.7, label="เป็นเบาหวาน")]
fig.legend(handles=legend_elements, loc="upper center",
           ncol=2, fontsize=9, framealpha=0.5,
           bbox_to_anchor=(0.5, 0.99))

fig.suptitle("Pima Indians Diabetes — EDA Overview", fontsize=13,
             fontweight="500", y=1.01)

plt.savefig("/mnt/user-data/outputs/step1_eda.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\n✓ บันทึก chart → step1_eda.png")
