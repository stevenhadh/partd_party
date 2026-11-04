import numpy as np
import pandas as pd
import streamlit as st

# --- 1. CORE SIMULATION FUNCTION ---


def simulate_part_d(
    df,
    deductible=615,
    catastrophic_limit=2100,
    beneficiary_coinsurance=0.25,
    plan_coinsurance=0.75,
    initial_manufacturer_discount_rate=0.10,
    catastrophic_manufacturer_discount_rate=0.20,
    federal_reinsurance_rate=0.20,
):
    """Simulate Part D benefit stages with phase-specific controls."""
    result = df.copy()

    # Validation
    if deductible < 0 or catastrophic_limit <= 0:
        raise ValueError("Deductible and catastrophic limit must be valid.")
    if deductible > catastrophic_limit:
        raise ValueError("Deductible cannot exceed catastrophic limit.")
    if not (0 <= beneficiary_coinsurance <= 1 and 0 <= plan_coinsurance <= 1):
        raise ValueError("Coinsurance rates must be between 0 and 1.")
    if abs(beneficiary_coinsurance + plan_coinsurance - 1) > 1e-9:
        raise ValueError("Initial coinsurance percentages must sum to 100%.")

    rates = [
        initial_manufacturer_discount_rate,
        catastrophic_manufacturer_discount_rate,
        federal_reinsurance_rate,
    ]
    if any(not (0 <= r <= 1) for r in rates):
        raise ValueError("Discount and reinsurance rates must be between 0 and 1.")

    required = [
        "patient_id",
        "encounter_id",
        "pharmacy_order_id",
        "total_drug_cost",
    ]
    missing = [c for c in required if c not in result.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in your repository CSV: {missing}"
        )

    result["total_drug_cost"] = pd.to_numeric(
        result["total_drug_cost"], errors="raise"
    )

    result = result.sort_values(
        ["patient_id", "encounter_id", "pharmacy_order_id"]
    ).copy()

    # Initialize columns
    result["deductible_cost"] = 0.0
    result["initial_cost"] = 0.0
    result["catastrophic_cost"] = 0.0
    result["beneficiary_payment"] = 0.0
    result["initial_manufacturer_discount"] = 0.0
    result["catastrophic_manufacturer_discount"] = 0.0
    result["federal_reinsurance"] = 0.0
    result["plan_liability"] = 0.0

    for patient_id, indices in result.groupby("patient_id").groups.items():
        cumulative_deductible = 0.0
        cumulative_beneficiary_payment = 0.0

        for idx in indices:
            drug_cost = result.at[idx, "total_drug_cost"]
            remaining = drug_cost

            # 1. Deductible Phase
            deductible_remaining = max(deductible - cumulative_deductible, 0)
            deductible_cost = min(remaining, deductible_remaining)
            cumulative_deductible += deductible_cost
            remaining -= deductible_cost
            beneficiary_payment = deductible_cost

            # 2. Initial Coverage Phase
            oop_remaining = max(
                catastrophic_limit - cumulative_beneficiary_payment, 0
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

            # 3. Catastrophic Phase
            catastrophic_cost = remaining
            catastrophic_mfg_discount = 0.0
            federal_reinsurance = 0.0
            catastrophic_plan = 0.0

            if catastrophic_cost > 0:
                catastrophic_mfg_discount = (
                    catastrophic_cost * catastrophic_manufacturer_discount_rate
                )
                federal_reinsurance = (
                    catastrophic_cost * federal_reinsurance_rate
                )
                covered_portions = (
                    catastrophic_mfg_discount + federal_reinsurance
                )
                catastrophic_plan = max(
                    catastrophic_cost - covered_portions, 0.0
                )

            plan_liability = initial_plan + catastrophic_plan

            result.at[idx, "deductible_cost"] = deductible_cost
            result.at[idx, "initial_cost"] = initial_cost
            result.at[idx, "catastrophic_cost"] = catastrophic_cost
            result.at[idx, "beneficiary_payment"] = beneficiary_payment
            result.at[idx, "initial_manufacturer_discount"] = (
                initial_mfg_discount
            )
            result.at[idx, "catastrophic_manufacturer_discount"] = (
                catastrophic_mfg_discount
            )
            result.at[idx, "federal_reinsurance"] = federal_reinsurance
            result.at[idx, "plan_liability"] = plan_liability

    return result


# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    file_path = "pharmacy_prescriptions.csv"
    df = pd.read_csv(file_path)
    return df


# --- 3. STREAMLIT USER INTERFACE ---
st.set_page_config(
    page_title="Medicare Part D Policy Simulator", layout="wide"
)

st.title("💊 Medicare Part D Portfolio Impact Simulator")
st.markdown(
    "Analyzing hardcoded portfolio data from the repository. Compare baseline structures (**2026 vs. 2027**) or tune custom plan variables."
)

# Load dataset safely with error handling
try:
    df_raw = load_data()
except Exception as e:
    st.error(
        f"Could not find or load the dataset. Error: {e}"
    )
    st.stop()

# Sidebar Configuration
st.sidebar.header("⚙️ Simulation Configuration")
sim_mode = st.sidebar.radio(
    "Configuration Mode",
    ["Baseline Comparison (2026 vs 2027)", "Custom Policy Sandbox"],
)

if sim_mode == "Baseline Comparison (2026 vs 2027)":
    p_2026 = {
        "deductible": 615.0,
        "catastrophic_limit": 2100.0,
        "beneficiary_coinsurance": 0.25,
        "plan_coinsurance": 0.75,
        "initial_manufacturer_discount_rate": 0.10,
        "catastrophic_manufacturer_discount_rate": 0.20,
        "federal_reinsurance_rate": 0.20,
    }
    p_2027 = {
        "deductible": 700.0,
        "catastrophic_limit": 2400.0,
        "beneficiary_coinsurance": 0.25,
        "plan_coinsurance": 0.75,
        "initial_manufacturer_discount_rate": 0.10,
        "catastrophic_manufacturer_discount_rate": 0.20,
        "federal_reinsurance_rate": 0.20,
    }

    res_a = simulate_part_d(df_raw, **p_2026)
    res_b = simulate_part_d(df_raw, **p_2027)
    label_a, label_b = "2026 Baseline", "2027 Baseline"

else:
    st.sidebar.subheader("Custom Plan Builder")
    deductible = st.sidebar.slider(
        "Deductible ($)", 0, 1500, 650, step=25
    )
    catastrophic_limit = st.sidebar.slider(
        "Out-of-Pocket Cap ($)", 1000, 5000, 2250, step=50
    )
    beneficiary_coinsurance = st.sidebar.slider(
        "Beneficiary Coinsurance (Initial)", 0.0, 0.5, 0.25, step=0.05
    )
    plan_coinsurance = round(1.0 - beneficiary_coinsurance, 2)
    st.sidebar.text(
        f"Derived Plan Initial Coinsurance: {plan_coinsurance * 100:.0f}%"
    )

    init_mfg = st.sidebar.slider(
        "Initial Manufacturer Discount Rate", 0.0, 0.3, 0.10, step=0.01
    )
    cat_mfg = st.sidebar.slider(
        "Catastrophic Manufacturer Discount Rate", 0.0, 0.4, 0.20, step=0.01
    )
    fed_rein = st.sidebar.slider(
        "Federal Reinsurance Rate", 0.0, 0.5, 0.20, step=0.01
    )

    p_2026 = {
        "deductible": 615.0,
        "catastrophic_limit": 2100.0,
        "beneficiary_coinsurance": 0.25,
        "plan_coinsurance": 0.75,
        "initial_manufacturer_discount_rate": 0.10,
        "catastrophic_manufacturer_discount_rate": 0.20,
        "federal_reinsurance_rate": 0.20,
    }
    p_custom = {
        "deductible": deductible,
        "catastrophic_limit": catastrophic_limit,
        "beneficiary_coinsurance": beneficiary_coinsurance,
        "plan_coinsurance": plan_coinsurance,
        "initial_manufacturer_discount_rate": init_mfg,
        "catastrophic_manufacturer_discount_rate": cat_mfg,
        "federal_reinsurance_rate": fed_rein,
    }

    res_a = simulate_part_d(df_raw, **p_2026)
    res_b = simulate_part_d(df_raw, **p_custom)
    label_a, label_b = "2026 Baseline", "Custom Sandbox Plan"

# --- 4. AGGREGATE METRICS & COMPARISON ---


def summarize_results(df):
    return {
        "Total Drug Cost": df["total_drug_cost"].sum(),
        "Beneficiary Out-of-Pocket": df["beneficiary_payment"].sum(),
        "Plan Liability": df["plan_liability"].sum(),
        "Initial Mfg Discounts": df["initial_manufacturer_discount"].sum(),
        "Catastrophic Mfg Discounts": df[
            "catastrophic_manufacturer_discount"
        ].sum(),
        "Federal Reinsurance": df["federal_reinsurance"].sum(),
    }


sum_a = summarize_results(res_a)
sum_b = summarize_results(res_b)

st.header(
    f"📊 Portfolio Financial Impact: {label_a} vs. {label_b}"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Portfolio Drug Spend",
        value=f"${sum_b['Total Drug Cost']:,.2f}",
        delta=f"${sum_b['Total Drug Cost'] - sum_a['Total Drug Cost']:,.2f}",
    )
with col2:
    st.metric(
        label="Beneficiary Exposure (OOP)",
        value=f"${sum_b['Beneficiary Out-of-Pocket']:,.2f}",
        delta=f"${sum_b['Beneficiary Out-of-Pocket'] - sum_a['Beneficiary Out-of-Pocket']:,.2f}",
        delta_inverse=True,
    )
with col3:
    st.metric(
        label="Total Plan Liability",
        value=f"${sum_b['Plan Liability']:,.2f}",
        delta=f"${sum_b['Plan Liability'] - sum_a['Plan Liability']:,.2f}",
    )
with col4:
    st.metric(
        label="Federal Reinsurance",
        value=f"${sum_b['Federal Reinsurance']:,.2f}",
        delta=f"${sum_b['Federal Reinsurance'] - sum_a['Federal Reinsurance']:,.2f}",
    )

st.markdown("---")

# Comparative Breakdown Table
st.subheader("📋 Financial Flow Summary Table")
comparison_df = pd.DataFrame(
    {
        f"Metric ({label_a})": [
            f"${v:,.2f}" for v in sum_a.values()
        ],
        f"Metric ({label_b})": [
            f"${v:,.2f}" for v in sum_b.values()
        ],
        "Absolute Difference ($)": [
            f"${v_b - v_a:,.2f}" for v_a, v_b in zip(sum_a.values(), sum_b.values())
        ],
    },
    index=list(sum_a.keys()),
)
st.dataframe(comparison_df, use_container_width=True)

# Visualizing Stakeholder Burden Share
st.subheader("📈 Stakeholder Burden Comparison")
breakdown_chart_data = pd.DataFrame(
    {
        label_a: [
            sum_a["Beneficiary Out-of-Pocket"],
            sum_a["Plan Liability"],
            sum_a["Initial Mfg Discounts"]
            + sum_a["Catastrophic Mfg Discounts"],
            sum_a["Federal Reinsurance"],
        ],
        label_b: [
            sum_b["Beneficiary Out-of-Pocket"],
            sum_b["Plan Liability"],
            sum_b["Initial Mfg Discounts"]
            + sum_b["Catastrophic Mfg Discounts"],
            sum_b["Federal Reinsurance"],
        ],
    },
    index=[
        "Beneficiary (OOP)",
        "Plan Liability",
        "Manufacturer Discounts",
        "Federal Reinsurance",
    ],
)
st.bar_chart(breakdown_chart_data)

st.markdown("---")
st.subheader("🔍 Repository Dataset Simulation Inspector")
st.write(
    f"Showing simulated rows for **{label_b}** using your hardcoded repository file (preserving original columns like `drug_name`, `ndc_code`, etc.):"
)
st.dataframe(res_b, use_container_width=True)