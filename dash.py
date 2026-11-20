import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Part D Benefit Design Simulation",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* 1. Shift the React Aria slider track & thumb from red to blue */
    .st-key-ded_b [role="group"][data-rac],
    .st-key-oop_b [role="group"][data-rac],
    .st-key-coins_b [role="group"][data-rac],
    .st-key-imfg_b [role="group"][data-rac],
    .st-key-cmfg_b [role="group"][data-rac],
    .st-key-freins_b [role="group"][data-rac],
    .st-key-price_chg_b [role="group"][data-rac] {
        filter: hue-rotate(215deg) saturate(1.25) !important;
    }

    /* 2. Keep the floating thumb value & min/max tick labels crisp #1E88E5 / normal color */
    .st-key-ded_b [data-testid="stSliderThumbValue"],
    .st-key-oop_b [data-testid="stSliderThumbValue"],
    .st-key-coins_b [data-testid="stSliderThumbValue"],
    .st-key-imfg_b [data-testid="stSliderThumbValue"],
    .st-key-cmfg_b [data-testid="stSliderThumbValue"],
    .st-key-freins_b [data-testid="stSliderThumbValue"],
    .st-key-price_chg_b [data-testid="stSliderThumbValue"] {
        filter: hue-rotate(-215deg) !important;
        color: #1E88E5 !important;
    }

    .st-key-ded_b [data-testid="stSliderTickBar"],
    .st-key-oop_b [data-testid="stSliderTickBar"],
    .st-key-coins_b [data-testid="stSliderTickBar"],
    .st-key-imfg_b [data-testid="stSliderTickBar"],
    .st-key-cmfg_b [data-testid="stSliderTickBar"],
    .st-key-freins_b [data-testid="stSliderTickBar"],
    .st-key-price_chg_b [data-testid="stSliderTickBar"] {
        filter: hue-rotate(-215deg) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PRESETS = {
    "2025 Standard": {
        "deductible": 590.0,
        "oop": 2000.0,
        "coins": 25,
        "imfg": 10,
        "cmfg": 20,
        "freins": 20,
        "price_chg": 0,
    },
    "2026 Standard": {
        "deductible": 615.0,
        "oop": 2100.0,
        "coins": 25,
        "imfg": 10,
        "cmfg": 20,
        "freins": 20,
        "price_chg": 0,
    },
    "2027 Standard": {
        "deductible": 640.0,
        "oop": 2400.0,
        "coins": 25,
        "imfg": 10,
        "cmfg": 20,
        "freins": 20,
        "price_chg": 0,
    },
}
PRESET_NAMES = list(PRESETS.keys())
SELECT_OPTIONS = PRESET_NAMES + ["Custom"]


# ---------------------------------------------------------------------
# 1. Load Data
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
# 2. Core Simulation Engine
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
    drug_price_change_rate: float = 0.0,
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
    base_costs = pd.to_numeric(result["total_drug_cost"], errors="raise").to_numpy(dtype=np.float64)
    adjusted_costs = np.maximum(base_costs * (1.0 + float(drug_price_change_rate)), 0.0)
    result["total_drug_cost"] = adjusted_costs

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
            # remaining cost > 0, up to the beneficiary OOP / catastrophic threshold
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


def style_right_vs_left(right_df: pd.DataFrame, left_df: pd.DataFrame):
    """Highlight 'Amount' green if Right > Left, and red if Right < Left."""
    left_map = dict(zip(left_df["Responsible Party"], left_df["Amount"]))

    def _highlight_row(row):
        styles = [""] * len(row)
        party = row["Responsible Party"]
        right_val = row["Amount"]
        left_val = left_map.get(party, right_val)
        if "Amount" in row.index:
            amt_idx = list(row.index).index("Amount")
            if right_val > left_val:
                styles[amt_idx] = "background-color: #d1fae5; color: #065f46; font-weight: 500;"
            elif right_val < left_val:
                styles[amt_idx] = "background-color: #fee2e2; color: #991b1b; font-weight: 500;"
        return styles

    return right_df.style.apply(_highlight_row, axis=1).format({
        "Amount": "${:,.0f}",
        "Share": "{:.1%}",
    })


# ---------------------------------------------------------------------
# 3. Scenario States
# ---------------------------------------------------------------------
def _apply_preset_to_state(side: str, preset_name: str):
    vals = PRESETS[preset_name]
    st.session_state[f"last_preset_{side}"] = preset_name
    st.session_state[f"preset_select_{side}"] = preset_name
    st.session_state[f"ded_{side}"] = float(vals["deductible"])
    st.session_state[f"oop_{side}"] = float(vals["oop"])
    st.session_state[f"coins_{side}"] = int(vals["coins"])
    st.session_state[f"imfg_{side}"] = int(vals["imfg"])
    st.session_state[f"cmfg_{side}"] = int(vals["cmfg"])
    st.session_state[f"freins_{side}"] = int(vals["freins"])
    st.session_state[f"price_chg_{side}"] = int(vals["price_chg"])


def init_side_state(side: str, default_preset: str):
    last_key = f"last_preset_{side}"
    select_key = f"preset_select_{side}"
    blocked_key = f"custom_blocked_{side}"
    vals = PRESETS[default_preset]

    if last_key not in st.session_state:
        st.session_state[last_key] = default_preset
    if select_key not in st.session_state:
        st.session_state[select_key] = default_preset
    if blocked_key not in st.session_state:
        st.session_state[blocked_key] = False

    defaults = {
        f"ded_{side}": float(vals["deductible"]),
        f"oop_{side}": float(vals["oop"]),
        f"coins_{side}": int(vals["coins"]),
        f"imfg_{side}": int(vals["imfg"]),
        f"cmfg_{side}": int(vals["cmfg"]),
        f"freins_{side}": int(vals["freins"]),
        f"price_chg_{side}": int(vals["price_chg"]),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def on_preset_change(side: str):
    chosen = st.session_state[f"preset_select_{side}"]
    if chosen == "Custom":
        # Snap back to the last valid preset so users cannot manually select Custom
        fallback = st.session_state.get(f"last_preset_{side}", PRESET_NAMES[0])
        st.session_state[f"preset_select_{side}"] = fallback
        st.session_state[f"custom_blocked_{side}"] = True
    else:
        st.session_state[f"custom_blocked_{side}"] = False
        _apply_preset_to_state(side, chosen)


def on_slider_change(side: str):
    st.session_state[f"custom_blocked_{side}"] = False
    st.session_state[f"preset_select_{side}"] = "Custom"


# ---------------------------------------------------------------------
# 4. Config
# ---------------------------------------------------------------------
st.title("Part D Benefit Comparison Dashboard")

raw_df = load_data()

if raw_df is None:
    st.stop()

required_cols = ["patient_id", "encounter_id", "pharmacy_order_id", "total_drug_cost"]
if any(c not in raw_df.columns for c in required_cols):
    st.error(f"Dataset is missing required columns. Ensure columns include: {required_cols}")
    st.stop()

# ---------------------------------------------------------------------
# 5. Data Summary
# ---------------------------------------------------------------------
n_patients = raw_df["patient_id"].nunique()
n_rx = raw_df["pharmacy_order_id"].nunique()
base_total_gross = raw_df["total_drug_cost"].sum()

st.subheader("📊\u00A0\u00A0Dataset Overview")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Patients", f"{n_patients:,}")
kpi2.metric("Total Prescriptions", f"{n_rx:,}")
kpi3.metric("Base Gross Drug Spending", money(base_total_gross))
kpi4.metric("Base Avg Cost per Patient", money(base_total_gross / n_patients if n_patients else 0))

st.space("small")
st.caption("Synthetic dataset from Kaggle by DeepLumiere: https://www.kaggle.com/datasets/deeplumiere/hospital-management-data")
st.divider()

# ---------------------------------------------------------------------
# 6. Side-by-side Scenarios
# ---------------------------------------------------------------------
init_side_state("a", "2026 Standard")
init_side_state("b", "2027 Standard")

st.subheader("⚙️\u00A0\u00A0Scenario Settings")
col_cfg_left, col_cfg_right = st.columns(2)

with col_cfg_left:
    st.markdown("### Left Scenario (Base)")
    st.selectbox(
        "Select a year",
        options=SELECT_OPTIONS,
        key="preset_select_a",
        on_change=on_preset_change,
        args=("a",),
    )
    if st.session_state.get("custom_blocked_a"):
        st.caption("Custom is set automatically when adjusting sliders.")
    active_label_a = st.session_state["preset_select_a"]

    ded_a = st.slider(
        "Deductible ($)", 0.0, 2000.0, step=5.0,
        key="ded_a", on_change=on_slider_change, args=("a",)
    )
    oop_a = st.slider(
        "OOP Limit ($)", 500.0, 5000.0, step=50.0,
        key="oop_a", on_change=on_slider_change, args=("a",)
    )
    coins_a = st.slider(
        "Coinsurance (%)", 0, 50, step=1,
        key="coins_a", on_change=on_slider_change, args=("a",)
    ) / 100.0
    imfg_a = st.slider(
        "Initial Phase Manufacturer Discount (%) - Left", 0, 50, step=1,
        key="imfg_a", on_change=on_slider_change, args=("a",)
    ) / 100.0
    cmfg_a = st.slider(
        "Catastrophic Phase Manufacturer Discount (%) - Left", 0, 50, step=1,
        key="cmfg_a", on_change=on_slider_change, args=("a",)
    ) / 100.0
    freins_a = st.slider(
        "Federal Reinsurance (%)", 0, 50, step=1,
        key="freins_a", on_change=on_slider_change, args=("a",)
    ) / 100.0
    price_chg_a = st.slider(
        "Drug Price Change (%)", -200, 200, step=1,
        key="price_chg_a", on_change=on_slider_change, args=("a",)
    ) / 100.0

with col_cfg_right:
    st.markdown("### Right Scenario")
    st.selectbox(
        "Select a year",
        options=SELECT_OPTIONS,
        key="preset_select_b",
        on_change=on_preset_change,
        args=("b",),
    )
    if st.session_state.get("custom_blocked_b"):
        st.caption("Custom is set automatically when adjusting sliders.")
    active_label_b = st.session_state["preset_select_b"]

    ded_b = st.slider(
        "Deductible ($)", 0.0, 2000.0, step=5.0,
        key="ded_b", on_change=on_slider_change, args=("b",)
    )
    oop_b = st.slider(
        "OOP Limit ($)", 500.0, 5000.0, step=50.0,
        key="oop_b", on_change=on_slider_change, args=("b",)
    )
    coins_b = st.slider(
        "Coinsurance (%)", 0, 50, step=1,
        key="coins_b", on_change=on_slider_change, args=("b",)
    ) / 100.0
    imfg_b = st.slider(
        "Initial Phase Manufacturer Discount (%) - Right", 0, 50, step=1,
        key="imfg_b", on_change=on_slider_change, args=("b",)
    ) / 100.0
    cmfg_b = st.slider(
        "Catastrophic Phase Manufacturer Discount (%) - Right", 0, 50, step=1,
        key="cmfg_b", on_change=on_slider_change, args=("b",)
    ) / 100.0
    freins_b = st.slider(
        "Federal Reinsurance (%)", 0, 50, step=1,
        key="freins_b", on_change=on_slider_change, args=("b",)
    ) / 100.0
    price_chg_b = st.slider(
        "Drug Price Change (%)", -200, 200, step=1,
        key="price_chg_b", on_change=on_slider_change, args=("b",)
    ) / 100.0

st.divider()

# ---------------------------------------------------------------------
# 7. Number crunch
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
        drug_price_change_rate=price_chg_a,
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
        drug_price_change_rate=price_chg_b,
    )
except Exception as exc:
    st.error(f"Simulation execution error: {exc}")
    st.stop()

members_a = make_member_summary(results_a)
members_b = make_member_summary(results_b)

gross_a = results_a["total_drug_cost"].sum()
gross_b = results_b["total_drug_cost"].sum()

# ---------------------------------------------------------------------
# 8. Financial Responsibility
# ---------------------------------------------------------------------
st.subheader("💰  Financial Responsibility & Catastrophic Distribution")

col_res_l, col_res_r = st.columns(2)

with col_res_l:
    st.markdown(f"### Left Side ({active_label_a})")
    plan_a = results_a["plan_liability"].sum()
    bene_a = results_a["beneficiary_payment"].sum()
    mfg_a = results_a["manufacturer_burden"].sum()
    fed_a = results_a["federal_reinsurance"].sum()

    cat_count_a = int((members_a["catastrophic_cost"] > 0).sum())
    st.metric("Total Members Reaching Catastrophic", cat_count_a)
    st.metric("Catastrophic Share", f"{(cat_count_a / n_patients if n_patients else 0):.1%}")
    
    st.divider()

    df_a = pd.DataFrame({
        "Responsible Party": ["Beneficiary", "Plan Liability", "Manufacturer", "Federal Reinsurance"],
        "Amount": [bene_a, plan_a, mfg_a, fed_a],
        "Share": [
            bene_a / gross_a if gross_a else 0,
            plan_a / gross_a if gross_a else 0,
            mfg_a / gross_a if gross_a else 0,
            fed_a / gross_a if gross_a else 0,
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
    
    cat_count_b = int((members_b["catastrophic_cost"] > 0).sum())
    st.metric("Total Members Reaching Catastrophic", cat_count_b)
    st.metric("Catastrophic Share", f"{(cat_count_b / n_patients if n_patients else 0):.1%}")
    st.divider()

    df_b = pd.DataFrame({
        "Responsible Party": ["Beneficiary", "Plan Liability", "Manufacturer", "Federal Reinsurance"],
        "Amount": [bene_b, plan_b, mfg_b, fed_b],
        "Share": [
            bene_b / gross_b if gross_b else 0,
            plan_b / gross_b if gross_b else 0,
            mfg_b / gross_b if gross_b else 0,
            fed_b / gross_b if gross_b else 0,
        ],
    })
    st.dataframe(
        style_right_vs_left(df_b, df_a),
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# ---------------------------------------------------------------------
# 9. Cost Drivers (Drugs & Individuals)
# ---------------------------------------------------------------------

driver_col1, driver_col2 = st.columns(2)

with driver_col1:
    st.markdown("### 💊\u00A0\u00A0Highest Cost Drugs (Gross Spending)")
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
    st.markdown("### 👤\u00A0\u00A0Highest Cost Individuals (Under Left Scenario)")
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
st.subheader("🔍\u00A0\u00A0Individual Patient Transaction & Coverage Inspector")
patient_ids = sorted(raw_df["patient_id"].dropna().unique().tolist(), key=lambda x: str(x))
selected_patient = st.selectbox("Select Patient ID", options=patient_ids, format_func=lambda x: str(x))

p_claims_a = results_a[results_a["patient_id"] == selected_patient].copy()
p_summary_a = members_a[members_a["patient_id"] == selected_patient].iloc[0]

pi1, pi2, pi3, pi4 = st.columns(4)
pi1.metric("Patient Gross Cost", money(p_summary_a["gross_drug_cost"]))
pi2.metric("Member OOP", money(p_summary_a["beneficiary_payment"]))
pi3.metric("Plan Liability", money(p_summary_a["plan_liability"]))
pi4.metric("Prescription Count", f"{int(p_summary_a['prescription_count']):,}")

pi5, pi6, _, _ = st.columns(4)
pi5.metric("Manufacturer Burden", money(p_summary_a["manufacturer_burden"]))
pi6.metric("Federal Reinsurance", money(p_summary_a["federal_reinsurance"]))

st.markdown(f"**Transaction-Level Analysis for `{selected_patient}` (Under Left Scenario)**")
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
