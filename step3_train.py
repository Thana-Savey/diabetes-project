# ============================================================
#  Diabetes Project — Step 3: Model Training
#  เป้าหมาย: เทรนโมเดล 2 ตัว แล้วเปรียบเทียบประสิทธิภาพ
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ── โหลดข้อมูลที่เตรียมไว้จาก Step 2 ─────────────────────────
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()   # squeeze: DataFrame → Series
y_test  = pd.read_csv("y_test.csv").squeeze()

print("=" * 60)
print("  STEP 3 : MODEL TRAINING")
print("=" * 60)
print(f"  Train: {X_train.shape[0]} แถว | Test: {X_test.shape[0]} แถว\n")


# ──────────────────────────────────────────────────────────────
# โมเดลที่ 1 : LOGISTIC REGRESSION (Baseline)
# ──────────────────────────────────────────────────────────────
#
# ทำงานอย่างไร?
#   คำนวณ "น้ำหนัก" ให้แต่ละ feature แล้วรวมกัน
#   สูตร: P(เบาหวาน) = sigmoid( w1*Glucose + w2*BMI + ... + b )
#   sigmoid แปลงตัวเลขใดก็ได้ → ค่าระหว่าง 0-1 (ความน่าจะเป็น)
#
# ทำไมใช้เป็น Baseline?
#   → เร็ว, เข้าใจง่าย, ตีความได้ (ดูน้ำหนักแต่ละ feature ได้)
#   → ถ้าโมเดลซับซ้อนกว่า "ไม่ดีกว่านี้" = มีปัญหาอื่นที่ต้องแก้

print("── โมเดล 1: Logistic Regression ────────────────────────")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

y_pred_lr  = lr.predict(X_test)
y_prob_lr  = lr.predict_proba(X_test)[:, 1]   # ความน่าจะเป็นว่าเป็นเบาหวาน

acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"  Accuracy : {acc_lr:.4f}  ({acc_lr*100:.1f}%)")
print(f"  AUC-ROC  : {auc_lr:.4f}")


# ──────────────────────────────────────────────────────────────
# CONCEPT: METRICS ที่สำคัญ
# ──────────────────────────────────────────────────────────────
#
# สมมติโมเดลทายว่า "ไม่มีใครเป็นเบาหวานเลย"
# → Accuracy = 65% (เพราะมีแค่ 35% ที่เป็นเบาหวาน)
# → แต่โมเดลนี้ไร้ประโยชน์ทางการแพทย์!
#
# Confusion Matrix อธิบายได้ดีกว่า:
#
#                   ทายว่า "ไม่เป็น"   ทายว่า "เป็น"
#   จริงๆ "ไม่เป็น"       TN              FP   ← False Positive (alarm เกิน)
#   จริงๆ "เป็น"          FN              TP   ← False Negative (พลาดคนไข้)
#
# Precision = TP / (TP + FP)  → บอกว่า "ที่ทายว่าเป็น ถูกกี่ %"
# Recall    = TP / (TP + FN)  → บอกว่า "คนเป็นจริง จับได้กี่ %"
#
# ⚠️ ในทางการแพทย์ → RECALL สำคัญกว่า PRECISION
#   เพราะ "พลาดคนไข้" (FN ↑) อันตรายกว่า "วินิจฉัยผิดบวก" (FP ↑)
#
# AUC-ROC = พื้นที่ใต้ ROC curve
#   0.5 = เดาสุ่ม, 1.0 = สมบูรณ์แบบ, > 0.8 = ดี

print("\n  Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr,
      target_names=["ไม่เป็นเบาหวาน", "เป็นเบาหวาน"]))


# ──────────────────────────────────────────────────────────────
# โมเดลที่ 2 : LIGHTGBM
# ──────────────────────────────────────────────────────────────
#
# ทำงานอย่างไร? (Gradient Boosting)
#   รอบที่ 1: สร้าง decision tree ต้นแรก → ทายผิดหลายแถว
#   รอบที่ 2: สร้าง tree ต้นที่ 2 เพื่อแก้ข้อผิดพลาดของต้นแรก
#   รอบที่ 3: ต้นที่ 3 แก้ข้อผิดพลาดที่เหลือ ...
#   ทำซ้ำ n_estimators ครั้ง แล้วรวมผลทุกต้น
#
# Parameter สำคัญ:
#   n_estimators  = จำนวน tree (มากขึ้น = แม่นขึ้น แต่ช้าขึ้น)
#   learning_rate = ก้าวการเรียน (น้อย = ค่อยๆ เรียน แต่ stable กว่า)
#   num_leaves    = ความซับซ้อนของแต่ละ tree
#   scale_pos_weight = ชดเชย class imbalance (ตั้งค่าตาม negative/positive)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / pos   # ≈ 1.86 → บอก LightGBM ว่า class เบาหวานหายาก

print("── โมเดล 2: LightGBM ────────────────────────────────────")
print(f"  scale_pos_weight = {scale:.2f}  (neg={neg}, pos={pos})")

lgbm = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    scale_pos_weight=scale,
    random_state=42,
    verbose=-1           # ปิด log ระหว่างเทรน
)
lgbm.fit(X_train, y_train)

y_pred_lgbm = lgbm.predict(X_test)
y_prob_lgbm = lgbm.predict_proba(X_test)[:, 1]

acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
auc_lgbm = roc_auc_score(y_test, y_prob_lgbm)
print(f"\n  Accuracy : {acc_lgbm:.4f}  ({acc_lgbm*100:.1f}%)")
print(f"  AUC-ROC  : {auc_lgbm:.4f}")

print("\n  Classification Report (LightGBM):")
print(classification_report(y_test, y_pred_lgbm,
      target_names=["ไม่เป็นเบาหวาน", "เป็นเบาหวาน"]))


# ──────────────────────────────────────────────────────────────
# เปรียบเทียบผล
# ──────────────────────────────────────────────────────────────

print("=" * 60)
print("  สรุปเปรียบเทียบ")
print("=" * 60)
print(f"  {'Model':<25} {'Accuracy':>10} {'AUC-ROC':>10}")
print(f"  {'-'*45}")
print(f"  {'Logistic Regression':<25} {acc_lr:>10.4f} {auc_lr:>10.4f}")
print(f"  {'LightGBM':<25} {acc_lgbm:>10.4f} {auc_lgbm:>10.4f}")


# ──────────────────────────────────────────────────────────────
# Visualization: ROC Curve + Feature Importance + Confusion Matrix
# ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 5))
fig.patch.set_facecolor("#F8F8F6")
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

# --- Plot 1: ROC Curve ---
ax1 = fig.add_subplot(gs[0])

fpr_lr,   tpr_lr,   _ = roc_curve(y_test, y_prob_lr)
fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, y_prob_lgbm)

ax1.plot(fpr_lr,   tpr_lr,   color="#1D9E75", lw=2,
         label=f"Logistic Reg (AUC={auc_lr:.3f})")
ax1.plot(fpr_lgbm, tpr_lgbm, color="#D85A30", lw=2,
         label=f"LightGBM    (AUC={auc_lgbm:.3f})")
ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC=0.5)")
ax1.set_xlabel("False Positive Rate", fontsize=9)
ax1.set_ylabel("True Positive Rate",  fontsize=9)
ax1.set_title("ROC Curve", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8)
ax1.set_facecolor("#F8F8F6")
for spine in ax1.spines.values():
    spine.set_linewidth(0.4)

# --- Plot 2: Feature Importance (LightGBM) ---
ax2 = fig.add_subplot(gs[1])
importance = pd.Series(
    lgbm.feature_importances_,
    index=X_train.columns
).sort_values()

colors_bar = ["#D85A30" if v == importance.max() else "#1D9E75"
              for v in importance.values]
importance.plot(kind="barh", ax=ax2, color=colors_bar)
ax2.set_title("Feature Importance\n(LightGBM)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Importance score", fontsize=9)
ax2.tick_params(labelsize=8)
ax2.set_facecolor("#F8F8F6")
for spine in ax2.spines.values():
    spine.set_linewidth(0.4)

# --- Plot 3: Confusion Matrix (LightGBM) ---
ax3 = fig.add_subplot(gs[2])
cm = confusion_matrix(y_test, y_pred_lgbm)

im = ax3.imshow(cm, cmap="YlOrRd")
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(["Pred: No", "Pred: Yes"], fontsize=9)
ax3.set_yticklabels(["True: No", "True: Yes"], fontsize=9)
ax3.set_title("Confusion Matrix\n(LightGBM)", fontsize=11, fontweight="bold")

for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i, j]), ha="center", va="center",
                 fontsize=14, fontweight="bold",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

ax3.set_facecolor("#F8F8F6")

fig.suptitle("Step 3 — Model Comparison", fontsize=13,
             fontweight="bold", y=1.02)

plt.savefig("step3_results.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\n✓ บันทึก chart → step3_results.png")

print("\n" + "=" * 60)
print("  Step 3 เสร็จสมบูรณ์!")
print("  Step ถัดไป → Step 4: Evaluation เชิงลึก + Threshold Tuning")
print("=" * 60)
