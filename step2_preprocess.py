# ============================================================
#  Diabetes Project — Step 2: Preprocessing
#  เป้าหมาย: เตรียมข้อมูลให้พร้อมก่อนใส่โมเดล
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── โหลดข้อมูลจาก local file ─────────────────────────────────
df = pd.read_csv("diabetes.csv")

print("=" * 60)
print("  STEP 2 : PREPROCESSING")
print("=" * 60)


# ──────────────────────────────────────────────────────────────
# CONCEPT 1 : IMPUTATION (การเติมค่าที่หายไป)
# ──────────────────────────────────────────────────────────────
#
# ปัญหา: คอลัมน์บางตัวมีค่า 0 ซึ่งเป็นไปไม่ได้ในทางการแพทย์
#   เช่น ระดับน้ำตาล (Glucose) = 0 หมายความว่าคนนั้นตายไปแล้ว
#
# วิธีแก้: แทนค่า 0 ด้วย MEDIAN แยกตาม Class (เป็น/ไม่เป็น)
#
# ทำไมใช้ MEDIAN ไม่ใช้ MEAN?
#   → Median ทนทานต่อ outlier กว่า เช่น ถ้ามีคนอ้วนมากๆ
#     คนเดียวใน group จะทำให้ Mean สูงผิดปกติ แต่ Median ไม่เปลี่ยน
#
# ทำไมแยกตาม Class?
#   → คนเป็นเบาหวานกับไม่เป็น มีค่าเฉลี่ย Glucose ต่างกันมาก
#     ถ้าเติมค่าเดียวกันทั้งหมด = ใส่ bias ผิดๆ เข้าไปในข้อมูล

cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df_clean = df.copy()   # ทำสำเนาไว้เปรียบเทียบ ไม่แก้ของเดิม

for col in cols_to_fix:
    for outcome_val in [0, 1]:
        # คำนวณ median จากแถวที่ค่าไม่ใช่ 0 และ Outcome ตรงกัน
        mask_valid  = (df_clean[col] != 0) & (df_clean["Outcome"] == outcome_val)
        median_val  = df_clean.loc[mask_valid, col].median()

        # แทนค่า 0 ใน group นั้น
        mask_replace = (df_clean[col] == 0) & (df_clean["Outcome"] == outcome_val)
        df_clean.loc[mask_replace, col] = median_val

print("\n── ตรวจสอบหลัง Imputation ──────────────────────────────")
print("จำนวนค่า 0 ที่เหลือในคอลัมน์สำคัญ:")
print((df_clean[cols_to_fix] == 0).sum().to_string())
print("\n(ควรเป็น 0 ทุกคอลัมน์ ✓)")


# ──────────────────────────────────────────────────────────────
# CONCEPT 2 : TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────
#
# ทำไมต้องแบ่ง?
#   → เพื่อจำลองสถานการณ์จริง: โมเดลควร "ไม่เคยเห็น" ข้อมูล test
#     มาก่อนเลย ไม่ต่างจากการสอบ ถ้าโจทย์สอบซ้ำกับโจทย์ที่ฝึก
#     นักเรียนจะได้คะแนนดีแต่ไม่ได้หมายความว่าเข้าใจจริง
#
# stratify=Outcome หมายความว่าอะไร?
#   → บังคับให้ train และ test มีสัดส่วนเบาหวาน/ไม่เบาหวาน
#     เท่ากัน (~65%/35%) ไม่งั้น test อาจได้แต่ผู้ป่วยเบาหวาน
#     ทั้งหมดก็ได้ ซึ่งจะทำให้ผลการทดสอบไม่น่าเชื่อถือ
#
# test_size=0.2 → ใช้ 20% (≈154 แถว) เป็น test, 80% เป็น train

X = df_clean.drop(columns=["Outcome"])   # features (input)
y = df_clean["Outcome"]                  # label (สิ่งที่ต้องทาย)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,       # seed ตายตัว → ผลเหมือนเดิมทุกครั้ง
    stratify=y             # รักษาสัดส่วน class
)

print("\n── ขนาดหลัง Train/Test Split ────────────────────────────")
print(f"  Train : {X_train.shape[0]} แถว")
print(f"  Test  : {X_test.shape[0]} แถว")
print(f"\n  สัดส่วนเบาหวานใน Train : {y_train.mean():.1%}")
print(f"  สัดส่วนเบาหวานใน Test  : {y_test.mean():.1%}")
print("  (ควรใกล้เคียงกัน = stratify ทำงานถูกต้อง ✓)")


# ──────────────────────────────────────────────────────────────
# CONCEPT 3 : FEATURE SCALING (StandardScaler)
# ──────────────────────────────────────────────────────────────
#
# ปัญหา: แต่ละ feature มีสเกลต่างกันมาก เช่น
#   Insulin: 0-800+   vs   Pregnancies: 0-17
#
# ถ้าไม่ scale โมเดลอย่าง Logistic Regression จะคิดว่า
#   Insulin "สำคัญกว่า" แค่เพราะตัวเลขใหญ่กว่า
#   ทั้งที่ความสำคัญจริงอาจไม่ใช่
#
# StandardScaler ทำอะไร?
#   แปลงทุก feature ให้ mean=0, std=1
#   สูตร: z = (x - mean) / std
#
# ⚠️ กฎสำคัญ: fit ONLY บน train แล้วค่อย transform ทั้งสอง
#   ทำไม? เพราะถ้า fit บน test ด้วย = โมเดลแอบ "เห็น" ข้อมูล test
#   ก่อนแล้ว = ผลที่ได้ดีเกินจริง (Data Leakage)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)   # เรียนรู้ mean/std จาก train
X_test_scaled  = scaler.transform(X_test)        # ใช้ mean/std เดิม transform test

# แปลงกลับเป็น DataFrame เพื่อให้อ่านง่าย
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X.columns)

print("\n── ตัวอย่างข้อมูลหลัง Scaling (3 แถวแรก) ─────────────")
print(X_train_scaled.head(3).round(3).to_string())
print("\n  สังเกต: ทุก feature อยู่ในสเกลใกล้เคียงกันแล้ว ✓")

print("\n── ค่า mean หลัง scale (ควรใกล้ 0) ────────────────────")
print(X_train_scaled.mean().round(4).to_string())


# ──────────────────────────────────────────────────────────────
# บันทึกข้อมูลเพื่อใช้ใน Step 3
# ──────────────────────────────────────────────────────────────

X_train_scaled.to_csv("X_train.csv", index=False)
X_test_scaled.to_csv("X_test.csv",   index=False)
y_train.to_csv("y_train.csv",        index=False)
y_test.to_csv("y_test.csv",          index=False)

print("\n✓ บันทึกไฟล์สำหรับ Step 3:")
print("  X_train.csv, X_test.csv, y_train.csv, y_test.csv")


# ──────────────────────────────────────────────────────────────
# BONUS: Visualize — ก่อน vs หลัง Imputation
# ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.patch.set_facecolor("#F8F8F6")
fig.suptitle("Before vs After Imputation\n(ค่าสีแดง = ค่าที่ถูกแทนด้วย 0 เดิม)",
             fontsize=13, fontweight="500")

compare_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for i, col in enumerate(compare_cols):
    ax = axes[i // 3][i % 3]
    ax.hist(df[col],       bins=30, alpha=0.5, color="#1D9E75", label="ก่อน (มีค่า 0)")
    ax.hist(df_clean[col], bins=30, alpha=0.5, color="#D85A30", label="หลัง (เติมค่าแล้ว)")
    ax.set_title(col, fontsize=10, fontweight="500")
    ax.legend(fontsize=7)
    ax.set_facecolor("#F8F8F6")
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

axes[1][2].set_visible(False)   # ซ่อน subplot ที่ไม่ใช้
plt.tight_layout()
plt.savefig("step2_imputation.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✓ บันทึก chart → step2_imputation.png")

print("\n" + "=" * 60)
print("  Step 2 เสร็จสมบูรณ์!")
print("  Step ถัดไป → Step 3: สร้างและเทรนโมเดลแรก")
print("=" * 60)
