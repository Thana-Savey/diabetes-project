"""
Diabetes Risk Prediction App
รันด้วย: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from database import (
    init_db, save_patient, search_patients, get_patient_by_id, get_all_patients, seed_demo_data,
    schedule_followup, complete_followup,
    get_followups_by_patient, get_pending_followups, get_followup_stats,
)
from report import generate_report
from diseases import DISEASES
from multimodel import load_all_models, preprocess_input, risk_level as get_risk_level

warnings.filterwarnings("ignore")

# ── Init DB ────────────────────────────────────────────────────
init_db()
seed_demo_data()

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ── โหลดโมเดล (cache) ─────────────────────────────────────────
@st.cache_resource
def load_model():
    COLS_TO_FIX   = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    FEATURE_NAMES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                     "Insulin", "BMI", "DiabetesPedigree", "Age"]

    df = pd.read_csv("diabetes.csv")
    X  = df[FEATURE_NAMES].copy()
    y  = df["Outcome"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_c = X_train.copy()
    fill_vals = {}
    for col in COLS_TO_FIX:
        medians = []
        for cls in [0, 1]:
            mask = (X_train_c[col] != 0) & (y_train == cls)
            medians.append(X_train_c.loc[mask, col].median())
            X_train_c.loc[(X_train_c[col] == 0) & (y_train == cls), col] = medians[-1]
        fill_vals[col] = np.mean(medians)

    scaler    = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train_c), columns=FEATURE_NAMES)

    scale = (y_train == 0).sum() / (y_train == 1).sum()
    model = lgb.LGBMClassifier(
        n_estimators=340, learning_rate=0.019, num_leaves=10,
        min_child_samples=22, feature_fraction=0.60, bagging_fraction=0.69,
        bagging_freq=6, scale_pos_weight=scale, random_state=42, verbose=-1,
    )
    model.fit(X_train_s, y_train)
    explainer = shap.TreeExplainer(model)
    return model, scaler, explainer, fill_vals, FEATURE_NAMES, COLS_TO_FIX


model, scaler, explainer, fill_vals, FEATURE_NAMES, COLS_TO_FIX = load_model()

@st.cache_resource
def load_all_disease_models():
    return load_all_models()

DISEASE_REGISTRY = load_all_disease_models()


# ── Helper ────────────────────────────────────────────────────
def preprocess(raw: dict):
    df_in = pd.DataFrame([raw], columns=FEATURE_NAMES)
    for col in COLS_TO_FIX:
        df_in[col] = df_in[col].replace(0, np.nan).fillna(fill_vals[col])
    df_scaled = pd.DataFrame(scaler.transform(df_in), columns=FEATURE_NAMES)
    return df_in, df_scaled


def risk_badge(level: str) -> str:
    color = {"สูง": "#D85A30", "กลาง": "#f5a623", "ต่ำ": "#1D9E75"}.get(level, "#999")
    return f"<span style='background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:13px'>{level}</span>"


def shap_chart(df_raw, df_scaled):
    shap_values = explainer.shap_values(df_scaled)
    sv_row = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    order  = np.argsort(np.abs(sv_row))[::-1]

    feat_labels = [f"{FEATURE_NAMES[i]} = {df_raw[FEATURE_NAMES[i]].values[0]:.1f}"
                   for i in order]
    shap_sorted = sv_row[order]
    bar_colors  = ["#D85A30" if v > 0 else "#1D9E75" for v in shap_sorted]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#F8F8F6")
    bars = ax.barh(feat_labels[::-1], shap_sorted[::-1],
                   color=bar_colors[::-1], alpha=0.85)
    for bar, val in zip(bars, shap_sorted[::-1]):
        offset = 0.002 if val >= 0 else -0.002
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("SHAP value", fontsize=9)
    ax.set_title("SHAP — แดง = เพิ่มความเสี่ยง | เขียว = ลดความเสี่ยง",
                 fontsize=10, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    plt.tight_layout()
    return fig


def show_prediction_result(prob, prediction, risk_level, df_raw, df_scaled):
    """Render ผลทำนายและ SHAP chart (ใช้ร่วมกันทั้ง 2 tab)"""
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        if prediction == 1:
            st.error("### 🔴 เสี่ยงเป็นเบาหวาน")
        else:
            st.success("### 🟢 ไม่พบความเสี่ยงสูง")
    with c2:
        color = "#D85A30" if prob >= 0.7 else "#f5a623" if prob >= 0.4 else "#1D9E75"
        st.markdown(
            f"<div style='text-align:center;padding:12px;border-radius:10px;background:#f5f5f5'>"
            f"<div style='font-size:12px;color:#666'>ความน่าจะเป็น</div>"
            f"<div style='font-size:38px;font-weight:700;color:{color}'>{prob:.1%}</div>"
            f"</div>", unsafe_allow_html=True)
    with c3:
        icon = "🔴" if risk_level == "สูง" else "🟡" if risk_level == "กลาง" else "🟢"
        st.markdown(
            f"<div style='text-align:center;padding:12px;border-radius:10px;background:#f5f5f5'>"
            f"<div style='font-size:12px;color:#666'>ระดับความเสี่ยง</div>"
            f"<div style='font-size:30px;font-weight:700;color:{color}'>{icon} {risk_level}</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("#### 🔍 SHAP Explanation")
    st.caption("บอกว่า feature ไหนส่งผลต่อการตัดสินใจของโมเดล")
    st.pyplot(shap_chart(df_raw, df_scaled))


# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
st.title("🩺 Diabetes Risk Predictor")
st.caption("ระบบประเมินความเสี่ยงโรคเบาหวาน | LightGBM + SHAP + Patient Database")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["➕ ประเมินคนไข้ใหม่", "🔎 ค้นหาคนไข้", "📅 ติดตามคนไข้"])


# ──────────────────────────────────────────────────────────────
#  TAB 1: ประเมินคนไข้ใหม่
# ──────────────────────────────────────────────────────────────
with tab1:
    # ── เลือกโรค ──────────────────────────────────────────────
    disease_options = {
        f"{cfg['icon']} {cfg['name_th']} ({cfg['name_en']})": key
        for key, cfg in DISEASES.items()
        if key in DISEASE_REGISTRY
    }
    selected_disease_label = st.selectbox(
        "เลือกประเภทการประเมิน", list(disease_options.keys()), key="disease_select"
    )
    selected_disease = disease_options[selected_disease_label]
    disease_cfg      = DISEASES[selected_disease]

    st.caption(f"ℹ️ {disease_cfg['description']}")
    st.markdown("---")

    with st.form("new_patient_form"):
        st.subheader("📋 ข้อมูลคนไข้")

        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            hn   = st.text_input("HN (Hospital Number)", placeholder="เช่น HN-0006")
        with col_info2:
            name = st.text_input("ชื่อ-นามสกุล *", placeholder="กรุณากรอก")
        with col_info3:
            note = st.text_input("หมายเหตุ", placeholder="เช่น นัดติดตาม 3 เดือน")

        st.markdown("##### ค่าทางการแพทย์  *(กรอก 0 หากไม่ทราบค่า)*")

        # สร้าง input fields แบบ dynamic ตาม disease config
        fields_input = disease_cfg["input_fields"]
        n_cols = 4
        cols = st.columns(n_cols)
        field_values: dict[str, float] = {}
        for i, f in enumerate(fields_input):
            with cols[i % n_cols]:
                cast = int if f["fmt"] == "%d" else float
                val = st.number_input(
                    f["label"],
                    min_value=cast(f["min"]),
                    max_value=cast(f["max"]),
                    value=cast(f["default"]),
                    step=cast(f["step"]),
                    format=f["fmt"],
                    key=f"field_{f['key']}",
                )
                field_values[f["key"]] = val

        submitted = st.form_submit_button("🔍 ประเมินและบันทึก", type="primary",
                                          use_container_width=True)

    if submitted:
        if not name.strip():
            st.warning("กรุณากรอกชื่อคนไข้")
        else:
            state = DISEASE_REGISTRY[selected_disease]
            X     = preprocess_input(selected_disease, field_values, state)
            prob  = float(state["model"].predict_proba(X)[0, 1])
            pred  = int(prob >= 0.5)
            rlevel = get_risk_level(prob)

            # บันทึกลง DB (diabetes fields เท่านั้น ถ้าเป็นโรคอื่นเก็บใน note)
            if selected_disease == "diabetes":
                row_id = save_patient(dict(
                    hn=hn.strip() or None, name=name.strip(),
                    age=int(field_values.get("Age", 0)),
                    pregnancies=field_values.get("Pregnancies"),
                    glucose=field_values.get("Glucose"),
                    blood_pressure=field_values.get("BloodPressure"),
                    skin_thickness=field_values.get("SkinThickness"),
                    insulin=field_values.get("Insulin"),
                    bmi=field_values.get("BMI"),
                    diabetes_pedigree=field_values.get("DiabetesPedigree"),
                    risk_prob=round(prob, 4), risk_level=rlevel,
                    prediction=pred,
                    note=note.strip(),
                ))
            else:
                row_id = save_patient(dict(
                    hn=hn.strip() or None, name=name.strip(),
                    age=int(field_values.get("age", 0)),
                    risk_prob=round(prob, 4), risk_level=rlevel,
                    prediction=pred,
                    note=f"[{disease_cfg['name_th']}] {note.strip()}",
                ))

            st.success(f"✅ บันทึกแล้ว (ID: {row_id})")
            st.markdown("---")

            # ผลประเมิน
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                label = disease_cfg["risk_labels"].get(pred, "")
                if pred == 1:
                    st.error(f"### 🔴 {label}")
                else:
                    st.success(f"### 🟢 {label}")
            with c2:
                color = "#D85A30" if prob >= 0.7 else "#f5a623" if prob >= 0.4 else "#1D9E75"
                st.markdown(
                    f"<div style='text-align:center;padding:12px;border-radius:10px;background:#f5f5f5'>"
                    f"<div style='font-size:12px;color:#666'>ความน่าจะเป็น</div>"
                    f"<div style='font-size:38px;font-weight:700;color:{color}'>{prob:.1%}</div>"
                    f"</div>", unsafe_allow_html=True)
            with c3:
                icon = "🔴" if rlevel == "สูง" else "🟡" if rlevel == "กลาง" else "🟢"
                st.markdown(
                    f"<div style='text-align:center;padding:12px;border-radius:10px;background:#f5f5f5'>"
                    f"<div style='font-size:12px;color:#666'>ระดับความเสี่ยง</div>"
                    f"<div style='font-size:30px;font-weight:700;color:{color}'>{icon} {rlevel}</div>"
                    f"</div>", unsafe_allow_html=True)

            # SHAP — เฉพาะ diabetes (มี explainer)
            if selected_disease == "diabetes":
                df_raw_tab1 = pd.DataFrame([{
                    k: field_values.get(k, 0) for k in FEATURE_NAMES
                }])
                _, df_scaled_tab1 = preprocess(
                    {k: field_values.get(k, 0) for k in FEATURE_NAMES}
                )
                st.markdown("#### 🔍 SHAP Explanation")
                st.pyplot(shap_chart(df_raw_tab1, df_scaled_tab1))

            st.info("⚠️ ผลนี้เป็นเพียงการประเมินเบื้องต้น ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์ได้")


# ──────────────────────────────────────────────────────────────
#  TAB 2: ค้นหาคนไข้
# ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🔎 ค้นหาคนไข้")

    search_col, _ = st.columns([2, 3])
    with search_col:
        query = st.text_input("ค้นหาด้วยชื่อหรือ HN",
                              placeholder="เช่น สมชาย หรือ HN-0001")

    # ── ตาราง ──────────────────────────────────────────────────
    if query.strip():
        results = search_patients(query)
    else:
        results = get_all_patients()

    if results.empty:
        st.info("ไม่พบข้อมูลคนไข้")
    else:
        # แสดงตารางพร้อม badge
        display = results.copy()
        display["risk_prob"] = display["risk_prob"].apply(lambda x: f"{x:.1%}")
        display["prediction"] = display["prediction"].apply(
            lambda x: "✅ ไม่เสี่ยง" if x == 0 else "⚠️ เสี่ยง"
        )
        display.columns = ["ID", "HN", "ชื่อ", "อายุ", "Glucose", "BMI",
                           "โอกาส", "ระดับ", "ผล", "วันที่"]

        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(f"พบ {len(results)} รายการ")

        # ── ดูรายละเอียด ───────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📄 ดูรายละเอียดและ SHAP")

        detail_col1, detail_col2 = st.columns([1, 3])
        with detail_col1:
            patient_ids = results["id"].tolist()
            patient_names = [
                f"{row['name']}  ({row['hn'] or '-'})"
                for _, row in results.iterrows()
            ]
            selected_label = st.selectbox("เลือกคนไข้", patient_names)
            selected_idx   = patient_names.index(selected_label)
            selected_id    = patient_ids[selected_idx]

        # เก็บ patient ที่เลือกใน session_state เพื่อให้ข้อมูลคงอยู่ข้าม rerun
        if st.button("📊 ดูผลและ SHAP", type="primary"):
            p = get_patient_by_id(selected_id)
            if p:
                raw = dict(
                    Pregnancies=p["pregnancies"], Glucose=p["glucose"],
                    BloodPressure=p["blood_pressure"], SkinThickness=p["skin_thickness"],
                    Insulin=p["insulin"], BMI=p["bmi"],
                    DiabetesPedigree=p["diabetes_pedigree"], Age=p["age"],
                )
                df_raw, df_scaled = preprocess(raw)
                shap_vals = explainer.shap_values(df_scaled)
                sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
                st.session_state["view_patient"]      = p
                st.session_state["view_df_raw"]       = df_raw
                st.session_state["view_df_scaled"]    = df_scaled
                st.session_state["view_shap"]         = sv
                st.session_state["view_patient_id"]   = selected_id
                st.session_state.pop("pdf_bytes", None)   # reset PDF เมื่อเปลี่ยนคนไข้

        # แสดงผลจาก session_state — คงอยู่แม้ rerun
        if st.session_state.get("view_patient"):
            patient    = st.session_state["view_patient"]
            df_raw     = st.session_state["view_df_raw"]
            df_scaled  = st.session_state["view_df_scaled"]
            sv         = st.session_state["view_shap"]

            with detail_col2:
                st.markdown(f"**{patient['name']}** | HN: `{patient['hn'] or '-'}` | อายุ {patient['age']} ปี")
                if patient['note']:
                    st.caption(f"📝 {patient['note']}")

            show_prediction_result(
                patient["risk_prob"], patient["prediction"],
                patient["risk_level"], df_raw, df_scaled,
            )

            with st.expander("📋 ข้อมูลทางการแพทย์ทั้งหมด"):
                info_df = pd.DataFrame([{k: v for k, v in df_raw.iloc[0].items()}]).T
                info_df.columns = ["ค่า"]
                st.dataframe(info_df, use_container_width=True)

            # ── Export PDF ────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📄 Export PDF Report")
            col_pdf1, col_pdf2 = st.columns([1, 3])
            with col_pdf1:
                printed_by = st.text_input("พิมพ์โดย (แพทย์)", value="dr.somchai",
                                           key="printed_by_input")
            with col_pdf2:
                include_shap = st.checkbox("รวม SHAP chart", value=True, key="include_shap")

            if st.button("📄 สร้าง PDF Report", type="primary", key="gen_pdf"):
                with st.spinner("กำลังสร้าง PDF..."):
                    fu_df = get_followups_by_patient(st.session_state["view_patient_id"])
                    fu_records = fu_df.to_dict("records") if not fu_df.empty else []
                    st.session_state["pdf_bytes"] = generate_report(
                        patient=patient,
                        shap_values=sv if include_shap else None,
                        feature_names=FEATURE_NAMES if include_shap else None,
                        followup_records=fu_records,
                        printed_by=printed_by,
                    )
                    hn = patient.get("hn") or f"patient_{st.session_state['view_patient_id']}"
                    st.session_state["pdf_filename"] = (
                        f"diabetes_report_{hn}_{datetime.now().strftime('%Y%m%d')}.pdf"
                    )

            # ใช้ base64 href แทน st.download_button เพื่อหลีกเลี่ยง rerun loop
            if st.session_state.get("pdf_bytes"):
                import base64
                b64 = base64.b64encode(st.session_state["pdf_bytes"]).decode()
                fname = st.session_state.get("pdf_filename", "report.pdf")
                st.markdown(
                    f'<a href="data:application/pdf;base64,{b64}" download="{fname}" '
                    f'style="display:inline-block;padding:10px 22px;background:#1D9E75;'
                    f'color:white;border-radius:8px;text-decoration:none;font-weight:bold;'
                    f'font-size:15px;">⬇️ Download PDF</a>',
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────────────────────
#  TAB 3: ติดตามคนไข้ (Follow-up Tracking)
# ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📅 ระบบติดตามคนไข้ (Follow-up Tracking)")

    # ── สถิติ ─────────────────────────────────────────────────
    stats = get_followup_stats()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("นัดทั้งหมด",      stats["total"])
    m2.metric("รอติดตาม",        stats["pending"])
    m3.metric("มาครบแล้ว",       stats["completed"])
    m4.metric("ขาดนัด",          stats["missed"])
    m5.metric("Accuracy จริง",
              f"{stats['model_accuracy_on_followup']}%" if stats["model_accuracy_on_followup"] else "-",
              help="เปรียบ prediction เดิม vs ผลตรวจจริงเมื่อมา follow-up")

    st.markdown("---")

    sec1, sec2 = st.columns([1.4, 1])

    # ── คอลัมน์ซ้าย: รายชื่อที่ต้องติดตาม ────────────────────
    with sec1:
        st.markdown("#### 🔔 นัดที่ต้องติดตาม (30 วันข้างหน้า)")
        pending_df = get_pending_followups(days_ahead=30)

        if pending_df.empty:
            st.info("ไม่มีนัดที่ต้องติดตามในช่วงนี้")
        else:
            # เพิ่มสัญลักษณ์เกินนัด
            display_pending = pending_df.copy()
            display_pending["สถานะ"] = display_pending["overdue"].apply(
                lambda x: "🔴 เกินนัด" if x else "🟡 รอ"
            )
            display_pending = display_pending[[
                "followup_id", "hn", "name", "scheduled_date", "risk_level", "สถานะ", "note"
            ]].rename(columns={
                "followup_id": "ID", "hn": "HN", "name": "ชื่อ",
                "scheduled_date": "วันนัด", "risk_level": "ระดับ", "note": "หมายเหตุ",
            })
            st.dataframe(display_pending, use_container_width=True, hide_index=True)

        # ── บันทึกผลการมาติดตาม ───────────────────────────────
        st.markdown("#### ✅ บันทึกผลการติดตาม")
        if not pending_df.empty:
            followup_options = {
                f"#{row['followup_id']} — {row['name']} (นัด {row['scheduled_date']})": row["followup_id"]
                for _, row in pending_df.iterrows()
            }
            selected_label = st.selectbox("เลือกนัดที่ต้องการบันทึก", list(followup_options.keys()))
            selected_fid   = followup_options[selected_label]

            with st.form("complete_followup_form"):
                fc1, fc2 = st.columns(2)
                with fc1:
                    actual_date = st.date_input("วันที่มาจริง")
                    fu_status   = st.selectbox("สถานะ", ["completed", "missed", "cancelled"],
                                               format_func=lambda x: {
                                                   "completed": "✅ มาตามนัด",
                                                   "missed":    "❌ ขาดนัด",
                                                   "cancelled": "🚫 ยกเลิก",
                                               }[x])
                with fc2:
                    actual_outcome = st.selectbox("ผลการตรวจ (ถ้าทราบ)",
                                                  [None, 0, 1],
                                                  format_func=lambda x: {
                                                      None: "ยังไม่ทราบ",
                                                      0: "0 — ไม่เป็นเบาหวาน",
                                                      1: "1 — เป็นเบาหวาน",
                                                  }[x])
                    glucose_new = st.number_input("Glucose ใหม่ (mg/dL) — 0 ถ้าไม่วัด", 0, 300, 0)
                    bmi_new     = st.number_input("BMI ใหม่ — 0 ถ้าไม่วัด", 0.0, 70.0, 0.0, step=0.1)

                fu_note    = st.text_input("หมายเหตุ")
                fu_submit  = st.form_submit_button("💾 บันทึกผล", type="primary")

            if fu_submit:
                ok = complete_followup(
                    followup_id=selected_fid,
                    actual_date=str(actual_date),
                    actual_outcome=actual_outcome,
                    glucose_new=glucose_new if glucose_new > 0 else None,
                    bmi_new=bmi_new if bmi_new > 0 else None,
                    note=fu_note,
                    status=fu_status,
                )
                if ok:
                    st.success("บันทึกผลเรียบร้อยแล้ว")
                    st.rerun()
                else:
                    st.error("ไม่พบ follow-up นี้")
        else:
            st.info("ไม่มีนัดที่รอบันทึก")

    # ── คอลัมน์ขวา: นัดใหม่ + ดูประวัติ ─────────────────────
    with sec2:
        st.markdown("#### 📌 สร้างนัดใหม่")
        all_patients = get_all_patients()

        if not all_patients.empty:
            patient_options = {
                f"{row['name']}  ({row['hn'] or '-'})": row["id"]
                for _, row in all_patients.iterrows()
            }
            with st.form("schedule_form"):
                sel_patient_label = st.selectbox("เลือกคนไข้", list(patient_options.keys()))
                sel_patient_id    = patient_options[sel_patient_label]
                sched_date  = st.date_input("วันที่นัด")
                sched_note  = st.text_input("หมายเหตุ / คำแนะนำ")
                sched_submit = st.form_submit_button("📅 บันทึกนัด", type="primary")

            if sched_submit:
                fid = schedule_followup(
                    patient_id=sel_patient_id,
                    scheduled_date=str(sched_date),
                    note=sched_note,
                    created_by="streamlit_user",
                )
                st.success(f"บันทึกนัดสำเร็จ (Follow-up ID: {fid})")
                st.rerun()

        st.markdown("#### 📋 ประวัติ Follow-up")
        if not all_patients.empty:
            hist_label = st.selectbox(
                "ดูประวัติของ",
                list(patient_options.keys()),
                key="hist_select",
            )
            hist_id = patient_options[hist_label]
            hist_df = get_followups_by_patient(hist_id)

            if hist_df.empty:
                st.info("ยังไม่มีประวัติ follow-up")
            else:
                display_hist = hist_df[[
                    "id", "scheduled_date", "actual_date", "status",
                    "actual_outcome", "glucose_new", "bmi_new",
                ]].rename(columns={
                    "id": "ID", "scheduled_date": "วันนัด",
                    "actual_date": "วันจริง", "status": "สถานะ",
                    "actual_outcome": "ผล", "glucose_new": "Glucose",
                    "bmi_new": "BMI",
                })
                st.dataframe(display_hist, use_container_width=True, hide_index=True)
