import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Part D Side-by-Side Comparison",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------------------
# Core simulation engine
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
    if not (0 <= beneficiary_coinsurance <= 1 and 0 <= plan_coinsurance <= 1):
        raise ValueError("Coinsurance rates must be between 0 and 1.")
    if abs(beneficiary_coinsurance + plan_coinsurance - 1) > 1e-9:
        raise ValueError("Initial coinsurance percentages must sum to 100%.")

    required = ["patient_id", "encounter_id", "pharmacy_order_id", "total_drug_cost"]
    missing = [c for c in required if c not in result.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result["total_drug_cost"] = pd.to_numeric(result["total_drug_cost"], errors="raise")
    if result["total_drug_cost"].isna().any():
        raise ValueError("total_drug_cost contains missing values.")
    if (result["total_drug_cost"] < 0).any():
        raise ValueError("total_drug_cost cannot contain negative values.")
    if result["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values.")

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

def money(value):
    return f"${value:,.0f}"

def pct(value):
    return f"{value:.1%}"

# ---------------------------------------------------------------------
# Sidebar & File Upload
# ---------------------------------------------------------------------
st.title("💊 Part D Cost & Responsibility Comparison")
st.caption("Compare two benefit scenarios side-by-side using year presets or custom parameters.")

with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload prescription CSV", type=["csv"])
    st.caption("Required: `patient_id`, `encounter_id`, `pharmacy_order_id`, `total_drug_cost`")

if uploaded_file is None:
    st.info("👈 Please upload your prescription claims CSV in the sidebar to begin.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

required_cols = ["patient_id", "encounter_id", "pharmacy_order_id", "total_drug_cost"]
if any(c not in raw_df.columns for c in required_cols):
    st.error("Missing one or more required columns in CSV.")
    st.stop()

# ---------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------
presets = {
    "2025 Standard": {
        "deductible": 590.0, 
        "catastrophic_limit": 2000.0, 
        "beneficiary_coinsurance": 0.25, 
        "initial_mfg": 0.10, 
        "cat_mfg": 0.20, 
        "federal_reinsurance": 0.20
    },
    "2026 Standard": {
        "deductible": 615.0, 
        "catastrophic_limit": 2100.0, 
        "beneficiary_coinsurance": 0.25, 
        "initial_mfg": 0.10, 
        "cat_mfg": 0.20, 
        "federal_reinsurance": 0.20
    },
    "2027 Standard": {
        "deductible": 640.0, 
        "catastrophic_limit": 2400.0, 
        "beneficiary_coinsurance": 0.25, 
        "initial_mfg": 0.10, 
        "cat_mfg": 0.20, 
        "federal_reinsurance": 0.20
    },
    "Custom": None
}

st.divider()

# ---------------------------------------------------------------------
# Two-Column Selector (Left vs Right Dropdowns & Sliders)
# ---------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Left Side Configuration")
    preset_a = st.selectbox("Choose Left Scenario", options=list(presets.keys()), index=0)
    
    if preset_a != "Custom":
        p_a = presets[preset_a]
        ded_a, cap_a, coins_a = p_a["deductible"], p_a["catastrophic_limit"], p_a["beneficiary_coinsurance"]
        imfg_a, cmfg_a, freins_a = p_a["initial_mfg"], p_a["cat_mfg"], p_a["federal_reinsurance"]
        st.markdown(f"* **Deductible:** `${ded_a:,.0f}`\n* **OOP Cap:** `${cap_a:,.0f}`\n* **Coinsurance:** `{coins_a:.0%}`")
    else:
        ded_a = st.number_input("Deductible ($) - Left", min_value=0.0, value=590.0, step=25.0)
        cap_a = st.number_input("OOP Threshold ($) - Left", min_value=1.0, value=2000.0, step=50.0)
        coins_a = st.slider("Beneficiary Coinsurance - Left (%)", 0, 100, 25) / 100
        imfg_a = st.slider("Initial Mfg Discount - Left (%)", 0, 100, 10) / 100
        cmfg_a = st.slider("Catastrophic Mfg Discount - Left (%)", 0, 100, 20) / 100
        freins_a = st.slider("Federal Reinsurance - Left (%)", 0, 100, 20) / 100

with col_right:
    st.subheader("Right Side Configuration")
    preset_b = st.selectbox("Choose Right Scenario", options=list(presets.keys()), index=1)
    
    if preset_b != "Custom":
        p_b = presets[preset_b]
        ded_b, cap_b, coins_b = p_b["deductible"], p_b["catastrophic_limit"], p_b["beneficiary_coinsurance"]
        imfg_b, cmfg_b, freins_b = p_b["initial_mfg"], p_b["cat_mfg"], p_b["federal_reinsurance"]
        st.markdown(f"* **Deductible:** `${ded_b:,.0f}`\n* **OOP Cap:** `${cap_b:,.0f}`\n* **Coinsurance:** `{coins_b:.0%}`")
    else:
        ded_b = st.number_input("Deductible ($) - Right", min_value=0.0, value=615.0, step=25.0)
        cap_b = st.number_input("OOP Threshold ($) - Right", min_value=1.0, value=2100.0, step=50.0)
        coins_b = st.slider("Beneficiary Coinsurance - Right (%)", 0, 100, 25) / 100
        imfg_b = st.slider("Initial Mfg Discount - Right (%)", 0, 100, 10) / 100
        cmfg_b = st.slider("Catastrophic Mfg Discount - Right (%)", 0, 100, 20) / 100
        freins_b = st.slider("Federal Reinsurance - Right (%)", 0, 100, 20) / 100

st.divider()

# ---------------------------------------------------------------------
# Run Simulations
# ---------------------------------------------------------------------
try:
    results_a = simulate_part_d(
        raw_df, deductible=ded_a, catastrophic_limit=cap_a,
        beneficiary_coinsurance=coins_a, plan_coinsurance=1 - coins_a,
        initial_manufacturer_discount_rate=imfg_a,
        catastrophic_manufacturer_discount_rate=cmfg_a,
        federal_reinsurance_rate=freins_a
    )
    results_b = simulate_part_d(
        raw_df, deductible=ded_b, catastrophic_limit=cap_b,
        beneficiary_coinsurance=coins_b, plan_coinsurance=1 - coins_b,
        initial_manufacturer_discount_rate=imfg_b,
        catastrophic_manufacturer_discount_rate=cmfg_b,
        federal_reinsurance_rate=freins_b
    )
except Exception as exc:
    st.error(f"Simulation error: {exc}")
    st.stop()

gross = results_a["total_drug_cost"].sum()

# Extract dollar responsibility metrics
bene_a = results_a["beneficiary_payment"].sum()
plan_a = results_a["plan_liability"].sum()
mfg_a = results_a["initial_manufacturer_discount"].sum() + results_a["catastrophic_manufacturer_discount"].sum()
fed_a = results_a["federal_reinsurance"].sum()

bene_b = results_b["beneficiary_payment"].sum()
plan_b = results_b["plan_liability"].sum()
mfg_b = results_b["initial_manufacturer_discount"].sum() + results_b["catastrophic_manufacturer_discount"].sum()
fed_b = results_b["federal_reinsurance"].sum()

# ---------------------------------------------------------------------
# Side-by-Side Comparison Display
# ---------------------------------------------------------------------
st.subheader("📊 Financial Responsibility Comparison")

res_col_left, res_col_right = st.columns(2)

with res_col_left:
    st.markdown(f"### Left Side: {preset_a}")
    df_a = pd.DataFrame({
        "Responsible Party": ["Beneficiary (OOP)", "Plan Liability", "Manufacturer Discounts", "Federal Reinsurance"],
        "Dollar Amount": [bene_a, plan_a, mfg_a, fed_a],
        "Share (%)": [bene_a/gross, plan_a/gross, mfg_a/gross, fed_a/gross] if gross else [0,0,0,0]
    })
    st.dataframe(
        df_a.style.format({"Dollar Amount": "${:,.0f}", "Share (%)": "{:.1%}"}),
        hide_index=True, use_container_width=True
    )

with res_col_right:
    st.markdown(f"### Right Side: {preset_b}")
    df_b = pd.DataFrame({
        "Responsible Party": ["Beneficiary (OOP)", "Plan Liability", "Manufacturer Discounts", "Federal Reinsurance"],
        "Dollar Amount": [bene_b, plan_b, mfg_b, fed_b],
        "Share (%)": [bene_b/gross, plan_b/gross, mfg_b/gross, fed_b/gross] if gross else [0,0,0,0]
    })
    st.dataframe(
        df_b.style.format({"Dollar Amount": "${:,.0f}", "Share (%)": "{:.1%}"}),
        hide_index=True, use_container_width=True
    )

st.divider()

st.subheader("📈 Side-by-Side Funding Chart")
comparison_chart_df = pd.DataFrame({
    f"Left ({preset_a})": [bene_a, plan_a, mfg_a, fed_a],
    f"Right ({preset_b})": [bene_b, plan_b, mfg_b, fed_b],
}, index=["Beneficiary", "Plan", "Manufacturer", "Federal"])

st.bar_chart(comparison_chart_df)
