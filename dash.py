import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Part D Benefit Comparison & Analysis Engine",
    page_icon="💊",
    layout="wide",
)

PRESETS = {
    "2025 Standard": {
        "deductible": 590.0,
        "oop": 2000.0,
        "coins": 25,
        "imfg": 10,
        "cmfg": 20,
        "freins": 20,
    },
    "2026 Standard": {
        "deductible": 615.0,
        "oop": 2100.0,
        "coins": 25,
        "imfg": 10,
        "cmfg": 20,
        "freins": 20,
    },
    "2027 Standard": {
        "deductible": 640.0,
        "oop": 2400.0,
        "coins": 25,
        "imfg": 10,
        "cmfg": 20,
        "freins": 20,
    },
}
PRESET_NAMES = list(PRESETS.keys())


# ---------------------------------------------------------------------
# 1. Data Loader (Cached)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Loading prescription data...")
def load_data(path: str = "pharmacy_prescriptions.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None


# ---------------------------------------------------------------------
# 2. Core Simulation Engine (Cached + NumPy-accelerated)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Running Part D benefit simulation...")
def simulate_part_d(
    df: pd.DataFrame,
    deductible: float = 590.0,
    catastrophic_limit: float = 2000.0,
    beneficiary_coinsurance: float = 0.25,
    plan_coinsurance: float = 0.75,
    initial_manufacturer_discount_rate: float = 0.10,
    catastrophic_manufacturer_discount_rate: float = 0.20,
    federal_reinsurance_rate: float = 0.20,
) -> pd.DataFrame:
    """Illustrative benefit-stage allocation model for Part D."""
    if deductible < 0 or catastrophic_limit <= 0:
        raise ValueError("Deductible and catastrophic limit must be valid.")
    if deductible > catastrophic_limit:
        raise ValueError("Deductible cannot exceed catastrophic limit.")

    required = ["patient_id", "encounter_id", "pharmacy_order_id", "total_drug_cost"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()
    result["total_drug_cost"] = pd.to_numeric(result["total_drug_cost"], errors="raise")
    result = result.sort_values(
        ["patient_id", "encounter_id", "pharmacy_order_id"],
        kind="stable",
    ).reset_index(drop=True)

    n = len(result)
    costs = result["total_drug_cost"].to_numpy(dtype=np.float64)

    deductible_cost_arr = np.zeros(n, dtype=np.float64)
    initial_cost_arr = np.zeros(n, dtype=np.float64)
    catastrophic_cost_arr = np.zeros(n, dtype=np.float64)
    beneficiary_payment_arr = np.zeros(n, dtype=np.float64)
    initial_mfg_arr = np.zeros(n, dtype=np.float64)
    catastrophic_mfg_arr = np.zeros(n, dtype=np.float64)
    manufacturer_burden_arr = np.zeros(n, dtype=np.float64)
    federal_reinsurance_arr = np.zeros(n, dtype=np.float64)
    plan_liability_arr = np.zeros(n, dtype=np.float64)

    # Group row positions by patient_id while preserving encounter/order sequence
    patient_groups = result.groupby("patient_id", sort=False).indices

    for _, idx_array in patient_groups.items():
        cumulative_deductible = 0.0
        cumulative_beneficiary_payment = 0.0

        for idx in idx_array:
            remaining = float(costs[idx])

            # 1. Deductible stage: beneficiary pays 100% until deductible is met
            deductible_remaining = max(deductible - cumulative_deductible, 0.0)
            ded_cost = min(remaining, deductible_remaining)
            cumulative_deductible += ded_cost
            remaining -= ded_cost
            bene_pay = ded_cost

            # 2. Initial coverage stage: begins once deductible is satisfied and
            # remaining gross cost > 0, up to the beneficiary OOP / catastrophic threshold
            oop_remaining = max(
                catastrophic_limit - cumulative_beneficiary_payment - ded_cost,
                0.0,
            )
            if remaining > 0.0 and oop_remaining > 0.0:
                if beneficiary_coinsurance > 0.0:
                    initial_capacity = oop_remaining / beneficiary_coinsurance
                else:
                    initial_capacity = float("inf")
                init_cost = min(remaining, initial_capacity)
            else:
                init_cost = 0.0

            init_beneficiary = init_cost * beneficiary_coinsurance
            init_plan_gross = init_cost * plan_coinsurance
            init_mfg = init_cost * initial_manufacturer_discount_rate
            init_plan = max(init_plan_gross - init_mfg, 0.0)

            bene_pay += init_beneficiary
            cumulative_beneficiary_payment += bene_pay
            remaining -= init_cost

            # 3. Catastrophic stage: any cost exceeding initial capacity
            cat_cost = max(remaining, 0.0)
            cat_mfg = cat_cost * catastrophic_manufacturer_discount_rate
            fed_reins = cat_cost * federal_reinsurance_rate
            cat_plan = max(cat_cost - cat_mfg - fed_reins, 0.0)

            mfg_burden = init_mfg + cat_mfg
            plan_liab = init_plan + cat_plan

            deductible_cost_arr[idx] = ded_cost
            initial_cost_arr[idx] = init_cost
            catastrophic_cost_arr[idx] = cat_cost
            beneficiary_payment_arr[idx] = bene_pay
            initial_mfg_arr[idx] = init_mfg
            catastrophic_mfg_arr[idx] = cat_mfg
            manufacturer_burden_arr[idx] = mfg_burden
            federal_reinsurance_arr[idx] = fed_reins
            plan_liability_arr[idx] = plan_liab

    result["deductible_cost"] = deductible_cost_arr
    result["initial_cost"] = initial_cost_arr
    result["catastrophic_cost"] = catastrophic_cost_arr
    result["beneficiary_payment"] = beneficiary_payment_arr
    result["initial_manufacturer_discount"] = initial_mfg_arr
    result["catastrophic_manufacturer_discount"] = catastrophic_mfg_arr
    result["manufacturer_burden"] = manufacturer_burden_arr
    result["federal_reinsurance"] = federal_reinsurance_arr
    result["plan_liability"] = plan_liability_arr

    return result


@st.cache_data(show_spinner=False)
def make_member_summary(results: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "gross_drug_cost": ("total_drug_cost", "sum"),
        "deductible_cost": ("deductible_cost", "sum"),
        "initial_cost": ("initial_cost", "sum"),
        "catastrophic_cost": ("catastrophic_cost", "sum"),
        "beneficiary_payment": ("beneficiary_payment", "sum"),
        "manufacturer_burden": ("manufacturer_burden", "sum"),
        "federal_reinsurance": ("federal_reinsurance", "sum"),
        "plan_liability": ("plan_liability", "sum"),
        "prescription_count": ("pharmacy_order_id", "nunique"),
    }
    if "drug_name" in results.columns:
        agg["unique_drugs"] = ("drug_name", "nunique")
    return results.groupby("patient_id", as_index=False).agg(**agg)


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


# ---------------------------------------------------------------------
# 3. Scenario State Callbacks (Presets selectable; sliders switch to Custom)
# ---------------------------------------------------------------------
def init_side_state(side: str, default_preset: str):
    mode_key = f"mode_{side}"
    select_key = f"preset_select_{side}"
    vals = PRESETS[default_preset]

    if mode_key not in st.session_state:
        st.session_state[mode_key] = default_preset
    if select_key not in st.session_state:
        st.session_state[select_key] = default_preset

    defaults = {
        f"ded_{side}": float(vals["deductible"]),
        f"oop_{side}": float(vals["oop"]),
        f"coins_{side}": int(vals["coins"]),
        f"imfg_{side}": int(vals["imfg"]),
        f"cmfg_{side}": int(vals["cmfg"]),
        f"freins_{side}": int(vals["freins"]),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def on_preset_change(side: str):
    chosen = st.session_state[f"preset_select_{side}"]
    st.session_state[f"mode_{side}"] = chosen
    vals = PRESETS[chosen]
    st.session_state[f"ded_{side}"] = float(vals["deductible"])
    st.session_state[f"oop_{side}"] = float(vals["oop"])
    st.session_state[f"coins_{side}"] = int(vals["coins"])
    st.session_state[f"imfg_{side}"] = int(vals["imfg"])
    st.session_state[f"cmfg_{side}"] = int(vals["cmfg"])
    st.session_state[f"freins_{side}"] = int(vals["freins"])


def on_slider_change(side: str):
    st.session_state[f"mode_{side}"] = "Custom"


# ---------------------------------------------------------------------
# 4. Sidebar Configuration & Data Load
# ---------------------------------------------------------------------
st.title("Part D Benefit Comparison & Analytics Dashboard")

with st.sidebar:
    st.header("Data Source (GitHub)")
    st.divider()
    st.header("Population Filters")

raw_df = load_data()

if raw_df is None:
    st.stop()

required_cols = ["patient_id", "encounter_id", "pharmacy_order_id", "total_drug_cost"]
if any(c not in raw_df.columns for c in required_cols):
    st.error(f"Dataset is missing required columns. Ensure columns include: {required_cols}")
    st.stop()

# ---------------------------------------------------------------------
# 5. Global KPIs & Data Summary (Top of Page)
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
# 6. Side-by-Side Scenario Configuration
# ---------------------------------------------------------------------
init_side_state("a", "2025 Standard")
init_side_state("b", "2026 Standard")

st.subheader("⚙️ Scenario Comparison Setup")
col_cfg_left, col_cfg_right = st.columns(2)

with col_cfg_left:
    st.markdown("### Left Scenario")
    st.selectbox(
        "Preset Year (Left)",
        options=PRESET_NAMES,
        key="preset_select_a",
        on_change=on_preset_change,
        args=("a",),
    )
    active_label_a = st.session_state["mode_a"]
    st.caption(f"Active configuration: **{active_label_a}**")

    ded_a = st.slider(
        "Deductible ($) - Left", 0.0, 2000.0, step=25.0,
        key="ded_a", on_change=on_slider_change, args=("a",)
    )
    oop_a = st.slider(
        "OOP Limit ($) - Left", 500.0, 5000.0, step=50.0,
        key="oop_a", on_change=on_slider_change, args=("a",)
    )
    coins_a = st.slider(
        "Coinsurance (%) - Left", 0, 50, step=1,
        key="coins_a", on_change=on_slider_change, args=("a",)
    ) / 100.0
    imfg_a = st.slider(
        "Initial Mfg Discount (%) - Left", 0, 50, step=1,
        key="imfg_a", on_change=on_slider_change, args=("a",)
    ) / 100.0
    cmfg_a = st.slider(
        "Catastrophic Mfg Discount (%) - Left", 0, 50, step=1,
        key="cmfg_a", on_change=on_slider_change, args=("a",)
    ) / 100.0
    freins_a = st.slider(
        "Federal Reinsurance (%) - Left", 0, 50, step=1,
        key="freins_a", on_change=on_slider_change, args=("a",)
    ) / 100.0

with col_cfg_right:
    st.markdown("### Right Scenario")
    st.selectbox(
        "Preset Year (Right)",
        options=PRESET_NAMES,
        key="preset_select_b",
        on_change=on_preset_change,
        args=("b",),
    )
    active_label_b = st.session_state["mode_b"]
    st.caption(f"Active configuration: **{active_label_b}**")

    ded_b = st.slider(
        "Deductible ($) - Right", 0.0, 2000.0, step=25.0,
        key="ded_b", on_change=on_slider_change, args=("b",)
    )
    oop_b = st.slider(
        "OOP Limit ($) - Right", 500.0, 5000.0, step=50.0,
        key="oop_b", on_change=on_slider_change, args=("b",)
    )
    coins_b = st.slider(
        "Coinsurance (%) - Right", 0, 50, step=1,
        key="coins_b", on_change=on_slider_change, args=("b",)
    ) / 100.0
    imfg_b = st.slider(
        "Initial Mfg Discount (%) - Right", 0, 50, step=1,
        key="imfg_b", on_change=on_slider_change, args=("b",)
    ) / 100.0
    cmfg_b = st.slider(
        "Catastrophic Mfg Discount (%) - Right", 0, 50, step=1,
        key="cmfg_b", on_change=on_slider_change, args=("b",)
    ) / 100.0
    freins_b = st.slider(
        "Federal Reinsurance (%) - Right", 0, 50, step=1,
        key="freins_b", on_change=on_slider_change, args=("b",)
    ) / 100.0

st.divider()

# ---------------------------------------------------------------------
# 7. Run Simulations (Cached by input parameters)
# ---------------------------------------------------------------------
try:
    results_a = simulate_part_d(
        raw_df,
        deductible=ded_a,
        catastrophic_limit=oop_a,
        beneficiary_coinsurance=coins_a,
        plan_coinsurance=1.0 - coins_a,
        initial_manufacturer_discount_rate=imfg_a,
        catastrophic_manufacturer_discount_rate=cmfg_a,
        federal_reinsurance_rate=freins_a,
    )
    results_b = simulate_part_d(
        raw_df,
        deductible=ded_b,
        catastrophic_limit=oop_b,
        beneficiary_coinsurance=coins_b,
        plan_coinsurance=1.0 - coins_b,
        initial_manufacturer_discount_rate=imfg_b,
        catastrophic_manufacturer_discount_rate=cmfg_b,
        federal_reinsurance_rate=freins_b,
    )
except Exception as exc:
    st.error(f"Simulation execution error: {exc}")
    st.stop()

members_a = make_member_summary(results_a)
members_b = make_member_summary(results_b)

# ---------------------------------------------------------------------
# 8. Side-by-Side Financial Responsibility & Stacked Bar Charts
# ---------------------------------------------------------------------
st.subheader("📊 Financial Responsibility & Coverage Phase Spending")

col_res_l, col_res_r = st.columns(2)

with col_res_l:
    st.markdown(f"### Left Side ({active_label_a})")
    plan_a = results_a["plan_liability"].sum()
    bene_a = results_a["beneficiary_payment"].sum()
    mfg_a = results_a["manufacturer_burden"].sum()
    fed_a = results_a["federal_reinsurance"].sum()

    st.metric("Total Plan Liability", money(plan_a))
    st.metric("Total Member Out-of-Pocket", money(bene_a))

    df_a = pd.DataFrame({
        "Responsible Party": ["Beneficiary", "Plan Liability", "Manufacturer", "Federal Reinsurance"],
        "Amount": [bene_a, plan_a, mfg_a, fed_a],
        "Share": [
            bene_a / total_gross if total_gross else 0,
            plan_a / total_gross if total_gross else 0,
            mfg_a / total_gross if total_gross else 0,
            fed_a / total_gross if total_gross else 0,
        ],
    })
    st.dataframe(
        df_a.style.format({"Amount": "${:,.0f}", "Share": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )

with col_res_r:
    st.markdown(f"### Right Side ({active_label_b})")
    plan_b = results_b["plan_liability"].sum()
    bene_b = results_b["beneficiary_payment"].sum()
    mfg_b = results_b["manufacturer_burden"].sum()
    fed_b = results_b["federal_reinsurance"].sum()

    st.metric(
        "Total Plan Liability",
        money(plan_b),
        delta=f"{money(plan_b - plan_a)} vs Left",
        delta_color="inverse",
    )
    st.metric(
        "Total Member Out-of-Pocket",
        money(bene_b),
        delta=f"{money(bene_b - bene_a)} vs Left",
        delta_color="inverse",
    )

    df_b = pd.DataFrame({
        "Responsible Party": ["Beneficiary", "Plan Liability", "Manufacturer", "Federal Reinsurance"],
        "Amount": [bene_b, plan_b, mfg_b, fed_b],
        "Share": [
            bene_b / total_gross if total_gross else 0,
            plan_b / total_gross if total_gross else 0,
            mfg_b / total_gross if total_gross else 0,
            fed_b / total_gross if total_gross else 0,
        ],
    })
    st.dataframe(
        df_b.style.format({"Amount": "${:,.0f}", "Share": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )

st.markdown("#### Spending by Coverage Phase (Stacked Comparison)")
stage_chart_df = pd.DataFrame(
    {
        f"Left - {active_label_a}": [
            results_a["deductible_cost"].sum(),
            results_a["initial_cost"].sum(),
            results_a["catastrophic_cost"].sum(),
        ],
        f"Right - {active_label_b}": [
            results_b["deductible_cost"].sum(),
            results_b["initial_cost"].sum(),
            results_b["catastrophic_cost"].sum(),
        ],
    },
    index=["Deductible Phase", "Initial Coverage Phase", "Catastrophic Phase"],
)

st.bar_chart(stage_chart_df)

st.divider()

# ---------------------------------------------------------------------
# 9. Stage Member Counts & Cost Drivers (Drugs & Individuals)
# ---------------------------------------------------------------------
st.subheader("👥 Beneficiary Distribution Across Benefit Stages")
cat_count_a = int((members_a["catastrophic_cost"] > 0).sum())
cat_count_b = int((members_b["catastrophic_cost"] > 0).sum())

stage_count_df = pd.DataFrame({
    "Scenario": [f"Left ({active_label_a})", f"Right ({active_label_b})"],
    "Total Patients": [n_patients, n_patients],
    "Reached Catastrophic": [cat_count_a, cat_count_b],
    "Catastrophic Share": [
        cat_count_a / n_patients if n_patients else 0,
        cat_count_b / n_patients if n_patients else 0,
    ],
})
st.dataframe(
    stage_count_df.style.format({"Catastrophic Share": "{:.1%}"}),
    hide_index=True,
    use_container_width=True,
)

st.divider()

driver_col1, driver_col2 = st.columns(2)

with driver_col1:
    st.markdown("### 💊 Highest Cost Drugs (Gross Spending)")
    if "drug_name" in results_a.columns:
        top_drugs = (
            results_a.groupby("drug_name", as_index=False)
            .agg(
                gross_cost=("total_drug_cost", "sum"),
                plan_liability=("plan_liability", "sum"),
                prescriptions=("pharmacy_order_id", "nunique"),
            )
            .sort_values("gross_cost", ascending=False)
            .head(10)
        )
        st.dataframe(
            top_drugs.style.format({"gross_cost": "${:,.0f}", "plan_liability": "${:,.0f}"}),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Dataset does not contain a `drug_name` column.")

with driver_col2:
    st.markdown("### 👤 Highest Cost Individuals (Left Scenario)")
    top_patients = members_a.sort_values("plan_liability", ascending=False).head(10)
    st.dataframe(
        top_patients.style.format({
            "gross_drug_cost": "${:,.0f}",
            "deductible_cost": "${:,.0f}",
            "initial_cost": "${:,.0f}",
            "catastrophic_cost": "${:,.0f}",
            "beneficiary_payment": "${:,.0f}",
            "manufacturer_burden": "${:,.0f}",
            "federal_reinsurance": "${:,.0f}",
            "plan_liability": "${:,.0f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# ---------------------------------------------------------------------
# 10. Patient Inspector (with Manufacturer & Federal Reinsurance Burden)
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

pi5, pi6, _, _ = st.columns(4)
pi5.metric("Manufacturer Burden (Left)", money(p_summary_a["manufacturer_burden"]))
pi6.metric("Federal Reinsurance (Left)", money(p_summary_a["federal_reinsurance"]))

st.markdown(f"**Transaction-Level Benefit Allocation for Patient `{selected_patient}` (Left Scenario)**")
candidate_cols = [
    "encounter_id",
    "pharmacy_order_id",
    "drug_name",
    "total_drug_cost",
    "deductible_cost",
    "initial_cost",
    "catastrophic_cost",
    "beneficiary_payment",
    "initial_manufacturer_discount",
    "catastrophic_manufacturer_discount",
    "manufacturer_burden",
    "federal_reinsurance",
    "plan_liability",
]
patient_display_cols = [c for c in candidate_cols if c in p_claims_a.columns]

currency_cols = [
    c for c in [
        "total_drug_cost",
        "deductible_cost",
        "initial_cost",
        "catastrophic_cost",
        "beneficiary_payment",
        "initial_manufacturer_discount",
        "catastrophic_manufacturer_discount",
        "manufacturer_burden",
        "federal_reinsurance",
        "plan_liability",
    ]
    if c in patient_display_cols
]

st.dataframe(
    p_claims_a[patient_display_cols].style.format({c: "${:,.2f}" for c in currency_cols}),
    hide_index=True,
    use_container_width=True,
)

st.caption("Dashboard fully synchronized with GitHub source data and dual preset/custom scenario sliders.")
