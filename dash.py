import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Part D Benefit Comparison & Analysis Engine",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------------------
# 1. Data Loader from GitHub (No file upload required)
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("pharmacy_prescriptions.csv")
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

# ---------------------------------------------------------------------
# 2. Core Simulation Engine
# ---------------------------------------------------------------------
def simulate_part_d(
    df,
    deductible=590.0,
    catastrophic_limit=2000.0,
    beneficiary_coinsurance=0.25,
    plan_coinsurance=0.75,
    initial_manufacturer_discount_rate=0.10,
    catastrophic_manufacturer_discount_rate=0.20,
    federal_reinsurance_rate=0.20,
):
    """Illustrative benefit-stage allocation model for Part D."""
    result = df.copy()

    if deductible < 0 or catastrophic_limit <= 0:
        raise ValueError("Deductible and catastrophic limit must be valid.")
    if deductible > catastrophic_limit:
        raise ValueError("Deductible cannot exceed catastrophic limit.")

    required = ["patient_id", "encounter_id", "pharmacy_order_id", "total_drug_cost"]
    missing = [c for c in required if c not in result.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result["total_drug_cost"] = pd.to_numeric(result["total_drug_cost"], errors="raise")
    result = result.reset_index(drop=True)
    result = result.sort_values(
        ["patient_id", "encounter_id", "pharmacy_order_id"],
        kind="stable"
    ).reset_index(drop=True)

    allocation_cols = [
        "deductible_cost", "initial_cost", "catastrophic_cost",
        "beneficiary_payment", "initial_manufacturer_discount",
        "catastrophic_manufacturer_discount", "federal_reinsurance",
        "plan_liability",
    ]
    for col in allocation_cols:
        result[col] = 0.0

    for patient_id, indices in result.groupby("patient_id", sort=False).groups.items():
        cumulative_deductible = 0.0
        cumulative_beneficiary_payment = 0.0

        for idx in indices:
            remaining = float(result.at[idx, "total_drug_cost"])

            # 1. Deductible stage
            deductible_remaining = max(deductible - cumulative_deductible, 0)
            deductible_cost = min(remaining, deductible_remaining)
            cumulative_deductible += deductible_cost
            remaining -= deductible_cost
            beneficiary_payment = deductible_cost

            # 2. Initial coverage stage
            oop_remaining = max(
                catastrophic_limit - cumulative_beneficiary_payment - deductible_cost,
                0,
            )
            initial_capacity = (
                oop_remaining / beneficiary_coinsurance
                if beneficiary_coinsurance > 0
                else float("inf")
            )
            initial_cost = min(remaining, initial_capacity)

            initial_beneficiary = initial_cost * beneficiary_coinsurance
            initial_plan_gross = initial_cost * plan_coinsurance
            initial_mfg_discount = initial_cost * initial_manufacturer_discount_rate
            initial_plan = max(initial_plan_gross - initial_mfg_discount, 0.0)

            beneficiary_payment += initial_beneficiary
            cumulative_beneficiary_payment += beneficiary_payment
            remaining -= initial_cost

            # 3. Catastrophic stage
            catastrophic_cost = remaining
            catastrophic_mfg_discount = catastrophic_cost * catastrophic_manufacturer_discount_rate
            federal_reinsurance = catastrophic_cost * federal_reinsurance_rate
            catastrophic_plan = max(
                catastrophic_cost
                - catastrophic_mfg_discount
                - federal_reinsurance,
                0.0,
            )
            plan_liability = initial_plan + catastrophic_plan

            result.at[idx, "deductible_cost"] = deductible_cost
            result.at[idx, "initial_cost"] = initial_cost
            result.at[idx, "catastrophic_cost"] = catastrophic_cost
            result.at[idx, "beneficiary_payment"] = beneficiary_payment
            result.at[idx, "initial_manufacturer_discount"] = initial_mfg_discount
            result.at[idx, "catastrophic_manufacturer_discount"] = catastrophic_mfg_discount
            result.at[idx, "federal_reinsurance"] = federal_reinsurance
            result.at[idx, "plan_liability"] = plan_liability

    return result

def make_member_summary(results):
    agg = {
        "gross_drug_cost": ("total_drug_cost", "sum"),
        "deductible_cost": ("deductible_cost", "sum"),
        "initial_cost": ("initial_cost", "sum"),
        "catastrophic_cost": ("catastrophic_cost", "sum"),
        "beneficiary_payment": ("beneficiary_payment", "sum"),
        "plan_liability": ("plan_liability", "sum"),
        "prescription_count": ("pharmacy_order_id", "nunique"),
    }
    if "drug_name" in results.columns:
        agg["unique_drugs"] = ("drug_name", "nunique")
    return results.groupby("patient_id").agg(**agg).reset_index()

def money(value):
    return f"${value:,.0f}"

def pct(value):
    return f"{value:.1%}"

# ---------------------------------------------------------------------
# 3. Sidebar Configuration & GitHub URL
# ---------------------------------------------------------------------
st.title("Part D Benefit Comparison & Analytics Dashboard")

with st.sidebar:
    st.header("Data Source (GitHub)")
    st.divider()
    st.header("Population Filters")

# Load data
raw_df = load_data()

if raw_df is None:
    st.stop()

required_cols = ["patient_id", "encounter_id", "pharmacy_order_id", "total_drug_cost"]
if any(c not in raw_df.columns for c in required_cols):
    st.error(f"Dataset is missing required columns. Ensure columns include: {required_cols}")
    st.stop()

# ---------------------------------------------------------------------
# 4. Global KPIs & Data Summary (Top of Page)
# ---------------------------------------------------------------------
n_patients = raw_df["patient_id"].nunique()
n_rx = raw_df["pharmacy_order_id"].nunique()
total_gross = raw_df["total_drug_cost"].sum()

st.subheader("📊 Dataset Overview & KPIs")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Patients", f"{n_patients:,}")
kpi2.metric("Total Prescriptions", f"{n_rx:,}")
kpi3.metric("Gross Drug Spending", money(total_gross))
kpi4.metric("Avg Cost per Patient", money(total_gross / n_patients if n_patients else 0))

st.divider()

# ---------------------------------------------------------------------
# 5. Side-by-Side Scenario Configuration (Presets + Always-Present Sliders)
# ---------------------------------------------------------------------
presets = {
    "2025 Standard": {"deductible": 590.0, "oop": 2000.0, "coins": 25, "imfg": 10, "cmfg": 20, "freins": 20},
    "2026 Standard": {"deductible": 615.0, "oop": 2100.0, "coins": 25, "imfg": 10, "cmfg": 20, "freins": 20},
    "2027 Standard": {"deductible": 640.0, "oop": 2400.0, "coins": 25, "imfg": 10, "cmfg": 20, "freins": 20},
    "Custom": {"deductible": 600.0, "oop": 2000.0, "coins": 25, "imfg": 10, "cmfg": 20, "freins": 20}
}

st.subheader("⚙️ Scenario Comparison Setup")
col_cfg_left, col_cfg_right = st.columns(2)

with col_cfg_left:
    st.markdown("### Left Scenario")
    preset_a = st.selectbox("Preset Year (Left)", options=list(presets.keys()), index=0, key="preset_a")
    
    # Initialize session state for sliders if not set or preset changed
    p_def = presets[preset_a] if preset_a != "Custom" else presets["2025 Standard"]
    
    ded_a = st.slider("Deductible ($) - Left", 0.0, 2000.0, float(p_def["deductible"]), 25.0, key="ded_a")
    oop_a = st.slider("OOP Limit ($) - Left", 500.0, 5000.0, float(p_def["oop"]), 50.0, key="oop_a")
    coins_a = st.slider("Coinsurance (%) - Left", 0, 50, int(p_def["coins"]), 1, key="coins_a") / 100
    imfg_a = st.slider("Initial Mfg Discount (%) - Left", 0, 50, int(p_def["imfg"]), 1, key="imfg_a") / 100
    cmfg_a = st.slider("Catastrophic Mfg Discount (%) - Left", 0, 50, int(p_def["cmfg"]), 1, key="cmfg_a") / 100
    freins_a = st.slider("Federal Reinsurance (%) - Left", 0, 50, int(p_def["freins"]), 1, key="freins_a") / 100

with col_cfg_right:
    st.markdown("### Right Scenario")
    preset_b = st.selectbox("Preset Year (Right)", options=list(presets.keys()), index=1, key="preset_b")
    
    p_def_b = presets[preset_b] if preset_b != "Custom" else presets["2026 Standard"]
    
    ded_b = st.slider("Deductible ($) - Right", 0.0, 2000.0, float(p_def_b["deductible"]), 25.0, key="ded_b")
    oop_b = st.slider("OOP Limit ($) - Right", 500.0, 5000.0, float(p_def_b["oop"]), 50.0, key="oop_b")
    coins_b = st.slider("Coinsurance (%) - Right", 0, 50, int(p_def_b["coins"]), 1, key="coins_b") / 100
    imfg_b = st.slider("Initial Mfg Discount (%) - Right", 0, 50, int(p_def_b["imfg"]), 1, key="imfg_b") / 100
    cmfg_b = st.slider("Catastrophic Mfg Discount (%) - Right", 0, 50, int(p_def_b["cmfg"]), 1, key="cmfg_b") / 100
    freins_b = st.slider("Federal Reinsurance (%) - Right", 0, 50, int(p_def_b["freins"]), 1, key="freins_b") / 100

st.divider()

# ---------------------------------------------------------------------
# 6. Run Simulations
# ---------------------------------------------------------------------
try:
    results_a = simulate_part_d(
        raw_df, deductible=ded_a, catastrophic_limit=oop_a,
        beneficiary_coinsurance=coins_a, plan_coinsurance=1 - coins_a,
        initial_manufacturer_discount_rate=imfg_a,
        catastrophic_manufacturer_discount_rate=cmfg_a,
        federal_reinsurance_rate=freins_a
    )
    results_b = simulate_part_d(
        raw_df, deductible=ded_b, catastrophic_limit=oop_b,
        beneficiary_coinsurance=coins_b, plan_coinsurance=1 - coins_b,
        initial_manufacturer_discount_rate=imfg_b,
        catastrophic_manufacturer_discount_rate=cmfg_b,
        federal_reinsurance_rate=freins_b
    )
except Exception as exc:
    st.error(f"Simulation execution error: {exc}")
    st.stop()

members_a = make_member_summary(results_a)
members_b = make_member_summary(results_b)

# ---------------------------------------------------------------------
# 7. Side-by-Side Financial Responsibility & Stacked Bar Charts for Coverage Phases
# ---------------------------------------------------------------------
st.subheader("📊 Financial Responsibility & Coverage Phase Spending")

col_res_l, col_res_r = st.columns(2)

with col_res_l:
    st.markdown(f"### Left Side ({preset_a})")
    plan_a = results_a["plan_liability"].sum()
    bene_a = results_a["beneficiary_payment"].sum()
    mfg_a = results_a["initial_manufacturer_discount"].sum() + results_a["catastrophic_manufacturer_discount"].sum()
    fed_a = results_a["federal_reinsurance"].sum()
    
    st.metric("Total Plan Liability", money(plan_a))
    st.metric("Total Member Out-of-Pocket", money(bene_a))
    
    df_a = pd.DataFrame({
        "Responsible Party": ["Beneficiary", "Plan Liability", "Manufacturer", "Federal Reinsurance"],
        "Amount": [bene_a, plan_a, mfg_a, fed_a],
        "Share": [bene_a/total_gross, plan_a/total_gross, mfg_a/total_gross, fed_a/total_gross]
    })
    st.dataframe(df_a.style.format({"Amount": "${:,.0f}", "Share": "{:.1%}"}), hide_index=True, use_container_width=True)

with col_res_r:
    st.markdown(f"### Right Side ({preset_b})")
    plan_b = results_b["plan_liability"].sum()
    bene_b = results_b["beneficiary_payment"].sum()
    mfg_b = results_b["initial_manufacturer_discount"].sum() + results_b["catastrophic_manufacturer_discount"].sum()
    fed_b = results_b["federal_reinsurance"].sum()
    
    st.metric("Total Plan Liability", money(plan_b), delta=f"{money(plan_b - plan_a)} vs Left", delta_color="inverse")
    st.metric("Total Member Out-of-Pocket", money(bene_b), delta=f"{money(bene_b - bene_a)} vs Left", delta_color="inverse")
    
    df_b = pd.DataFrame({
        "Responsible Party": ["Beneficiary", "Plan Liability", "Manufacturer", "Federal Reinsurance"],
        "Amount": [bene_b, plan_b, mfg_b, fed_b],
        "Share": [bene_b/total_gross, plan_b/total_gross, mfg_b/total_gross, fed_b/total_gross]
    })
    st.dataframe(df_b.style.format({"Amount": "${:,.0f}", "Share": "{:.1%}"}), hide_index=True, use_container_width=True)

st.markdown("#### Spending by Coverage Phase (Stacked Comparison)")
stage_chart_df = pd.DataFrame({
    f"Left - {preset_a}": [
        results_a["deductible_cost"].sum(),
        results_a["initial_cost"].sum(),
        results_a["catastrophic_cost"].sum()
    ],
    f"Right - {preset_b}": [
        results_b["deductible_cost"].sum(),
        results_b["initial_cost"].sum(),
        results_b["catastrophic_cost"].sum()
    ]
}, index=["Deductible Phase", "Initial Coverage Phase", "Catastrophic Phase"])

st.bar_chart(stage_chart_df)

st.divider()

# ---------------------------------------------------------------------
# 8. Stage Member Counts & Cost Drivers (Drugs & Individuals)
# ---------------------------------------------------------------------
st.subheader("👥 Beneficiary Distribution Across Benefit Stages")
cat_count_a = int((members_a["catastrophic_cost"] > 0).sum())
cat_count_b = int((members_b["catastrophic_cost"] > 0).sum())

stage_count_df = pd.DataFrame({
    "Scenario": [f"Left ({preset_a})", f"Right ({preset_b})"],
    "Total Patients": [n_patients, n_patients],
    "Reached Catastrophic": [cat_count_a, cat_count_b],
    "Catastrophic Share": [cat_count_a / n_patients if n_patients else 0, cat_count_b / n_patients if n_patients else 0]
})
st.dataframe(stage_count_df.style.format({"Catastrophic Share": "{:.1%}"}), hide_index=True, use_container_width=True)

st.divider()

driver_col1, driver_col2 = st.columns(2)

with driver_col1:
    st.markdown("### 💊 Highest Cost Drugs (Gross Spending)")
    if "drug_name" in results_a.columns:
        top_drugs = results_a.groupby("drug_name").agg(
            gross_cost=("total_drug_cost", "sum"),
            plan_liability=("plan_liability", "sum"),
            prescriptions=("pharmacy_order_id", "nunique")
        ).sort_values("gross_cost", ascending=False).head(10).reset_index()
        
        st.dataframe(top_drugs.style.format({"gross_cost": "${:,.0f}", "plan_liability": "${:,.0f}"}), hide_index=True, use_container_width=True)
    else:
        st.info("Dataset does not contain a `drug_name` column.")

with driver_col2:
    st.markdown("### 👤 Highest Cost Individuals (Left Scenario)")
    top_patients = members_a.sort_values("plan_liability", ascending=False).head(10)
    st.dataframe(top_patients.style.format({
        "gross_drug_cost": "${:,.0f}",
        "deductible_cost": "${:,.0f}",
        "initial_cost": "${:,.0f}",
        "catastrophic_cost": "${:,.0f}",
        "beneficiary_payment": "${:,.0f}",
        "plan_liability": "${:,.0f}"
    }), hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# 9. Patient Inspector
# ---------------------------------------------------------------------
st.subheader("🔍 Patient Transaction & Coverage Inspector")
patient_ids = sorted(raw_df["patient_id"].dropna().unique().tolist(), key=lambda x: str(x))
selected_patient = st.selectbox("Select Patient ID", options=patient_ids, format_func=lambda x: str(x))

p_claims_a = results_a[results_a["patient_id"] == selected_patient].copy()
p_summary_a = members_a[members_a["patient_id"] == selected_patient].iloc[0]

pi1, pi2, pi3, pi4 = st.columns(4)
pi1.metric("Patient Gross Cost", money(p_summary_a["gross_drug_cost"]))
pi2.metric("Member OOP (Left)", money(p_summary_a["beneficiary_payment"]))
pi3.metric("Plan Liability (Left)", money(p_summary_a["plan_liability"]))
pi4.metric("Prescription Count", f"{int(p_summary_a['prescription_count']):,}")

st.markdown(f"**Transaction-Level Benefit Allocation for Patient `{selected_patient}` (Left Scenario)**")
patient_display_cols = [c for c in ["encounter_id", "pharmacy_order_id", "drug_name", "total_drug_cost", "deductible_cost", "initial_cost", "catastrophic_cost", "beneficiary_payment", "plan_liability"] if c in p_claims_a.columns]

st.dataframe(p_claims_a[patient_display_cols].style.format({
    c: "${:,.2f}" for c in ["total_drug_cost", "deductible_cost", "initial_cost", "catastrophic_cost", "beneficiary_payment", "plan_liability"] if c in patient_display_cols
}), hide_index=True, use_container_width=True)

st.caption("Dashboard fully synchronized with GitHub source data and dual preset/custom scenario sliders.")
