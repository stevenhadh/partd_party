import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Prescription Drug Cost & Benefit Analysis",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------
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
    """Illustrative benefit-stage allocation model; not a CMS adjudicator."""
    result = df.copy()

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
        raise ValueError("patient_id contains missing values. Please clean these rows.")

    # Reset index so group indices are unique and safe for .at assignments.
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

            # 1. Deductible: beneficiary pays 100%.
            deductible_remaining = max(deductible - cumulative_deductible, 0)
            deductible_cost = min(remaining, deductible_remaining)
            cumulative_deductible += deductible_cost
            remaining -= deductible_cost
            beneficiary_payment = deductible_cost

            # 2. Initial coverage. Deductible payments on this same claim
            # are included when determining remaining OOP capacity.
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

            # 3. Catastrophic coverage.
            catastrophic_cost = remaining
            catastrophic_mfg_discount = (
                catastrophic_cost * catastrophic_manufacturer_discount_rate
            )
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def money(value):
    return f"${value:,.0f}"


def pct(value):
    return f"{value:.1%}"


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
    members = results.groupby("patient_id").agg(**agg).reset_index()
    return members


def safe_download(df, filename, label):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Header and data upload
# ---------------------------------------------------------------------
st.title("Prescription Drug Cost & Benefit Analysis")
st.caption(
    "Explore medication spending, high-cost patients, and an adjustable "
    "benefit-allocation scenario."
)

with st.sidebar:
    st.header("1. Data")
    uploaded_file = st.file_uploader("Upload prescription CSV", type=["csv"])
    st.caption(
        "Required columns: patient_id, encounter_id, pharmacy_order_id, "
        "total_drug_cost. Drug name and therapeutic class are recommended."
    )

if uploaded_file is None:
    st.info("Upload a CSV file in the sidebar to begin.")
    st.markdown(
        """
        **Expected input fields**

        | Field | Required? | Use |
        |---|---|---|
        | `patient_id` | Yes | Groups prescriptions by patient |
        | `encounter_id` | Yes | Ordering proxy for prescriptions |
        | `pharmacy_order_id` | Yes | Deterministic tie-breaker |
        | `total_drug_cost` | Yes | Cost allocated by the model |
        | `drug_name` | Recommended | Drug-level analysis |
        | `therapeutic_class` | Recommended | Drug-class analysis |
        | `days_supply` | Optional | Additional utilization analysis |
        """
    )
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

required_cols = ["patient_id", "encounter_id", "pharmacy_order_id", "total_drug_cost"]
missing_cols = [c for c in required_cols if c not in raw_df.columns]
if missing_cols:
    st.error(f"Your file is missing required columns: {', '.join(missing_cols)}")
    st.stop()

if raw_df.empty:
    st.warning("The uploaded file contains no rows.")
    st.stop()

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("2. Population filters")

    filtered_df = raw_df.copy()
    if "therapeutic_class" in filtered_df.columns:
        classes = sorted(
            filtered_df["therapeutic_class"].dropna().astype(str).unique().tolist()
        )
        excluded_classes = st.multiselect(
            "Exclude therapeutic classes",
            options=classes,
            help="Excluded classes are removed before the simulation is rerun.",
        )
        if excluded_classes:
            filtered_df = filtered_df[
                ~filtered_df["therapeutic_class"].astype(str).isin(excluded_classes)
            ].copy()
    else:
        excluded_classes = []
        st.caption("No therapeutic_class column found; class filtering is unavailable.")

    if "drug_name" in filtered_df.columns:
        drug_options = sorted(filtered_df["drug_name"].dropna().astype(str).unique())
        selected_drugs = st.multiselect(
            "Limit to specific drugs (optional)",
            options=drug_options,
            help="Leave blank to include all remaining drugs.",
        )
        if selected_drugs:
            filtered_df = filtered_df[
                filtered_df["drug_name"].astype(str).isin(selected_drugs)
            ].copy()

    st.header("3. Benefit assumptions")
    st.caption("Illustrative controls; confirm model rules before interpreting as Part D.")

    deductible = st.number_input("Deductible ($)", min_value=0.0, value=615.0, step=25.0)
    catastrophic_limit = st.number_input(
        "Beneficiary out-of-pocket threshold ($)",
        min_value=1.0,
        value=2100.0,
        step=100.0,
    )
    beneficiary_coinsurance = st.slider(
        "Beneficiary initial coinsurance",
        min_value=0,
        max_value=100,
        value=25,
        step=1,
    ) / 100
    plan_coinsurance = 1 - beneficiary_coinsurance

    with st.expander("Additional manufacturer / federal assumptions"):
        initial_mfg = st.slider(
            "Initial manufacturer discount",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
        ) / 100
        catastrophic_mfg = st.slider(
            "Catastrophic manufacturer discount",
            min_value=0,
            max_value=100,
            value=20,
            step=1,
        ) / 100
        federal_reinsurance = st.slider(
            "Federal reinsurance",
            min_value=0,
            max_value=100,
            value=20,
            step=1,
        ) / 100

    st.caption("Plan initial coinsurance is calculated as the remainder.")

if filtered_df.empty:
    st.warning("No records remain after applying your filters.")
    st.stop()

if deductible > catastrophic_limit:
    st.error("The deductible cannot exceed the out-of-pocket threshold.")
    st.stop()

# ---------------------------------------------------------------------
# Run model
# ---------------------------------------------------------------------
try:
    results = simulate_part_d(
        filtered_df,
        deductible=deductible,
        catastrophic_limit=catastrophic_limit,
        beneficiary_coinsurance=beneficiary_coinsurance,
        plan_coinsurance=plan_coinsurance,
        initial_manufacturer_discount_rate=initial_mfg,
        catastrophic_manufacturer_discount_rate=catastrophic_mfg,
        federal_reinsurance_rate=federal_reinsurance,
    )
except Exception as exc:
    st.error(f"Simulation could not run: {exc}")
    st.stop()

members = make_member_summary(results)

gross = results["total_drug_cost"].sum()
beneficiary = results["beneficiary_payment"].sum()
plan = results["plan_liability"].sum()
initial_mfg_total = results["initial_manufacturer_discount"].sum()
cat_mfg_total = results["catastrophic_manufacturer_discount"].sum()
federal_total = results["federal_reinsurance"].sum()
deductible_total = results["deductible_cost"].sum()
initial_total = results["initial_cost"].sum()
catastrophic_total = results["catastrophic_cost"].sum()
n_members = results["patient_id"].nunique()
n_rx = results["pharmacy_order_id"].nunique()
n_cat_members = int((members["catastrophic_cost"] > 0).sum())

# ---------------------------------------------------------------------
# Population overview
# ---------------------------------------------------------------------
st.subheader("Population overview")
st.caption(
    f"Showing {len(results):,} prescription records for {n_members:,} patients "
    f"after filters. {len(raw_df) - len(filtered_df):,} records excluded."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Patients", f"{n_members:,}")
k2.metric("Prescriptions", f"{n_rx:,}")
k3.metric("Gross drug cost", money(gross))
k4.metric("Plan liability", money(plan))
k5.metric("Reached catastrophic", f"{n_cat_members:,}", f"{pct(n_cat_members / n_members) if n_members else '0.0%'} of patients")

st.divider()

# ---------------------------------------------------------------------
# Financial results
# ---------------------------------------------------------------------
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Cost allocation by benefit stage")
    stage_df = pd.DataFrame(
        {
            "Benefit stage": ["Deductible", "Initial coverage", "Catastrophic"],
            "Gross cost": [deductible_total, initial_total, catastrophic_total],
        }
    ).set_index("Benefit stage")
    st.bar_chart(stage_df, horizontal=True)
    stage_display = pd.DataFrame(
        {
            "Benefit stage": ["Deductible", "Initial coverage", "Catastrophic"],
            "Gross cost": [deductible_total, initial_total, catastrophic_total],
            "Share of gross cost": [
                deductible_total / gross if gross else 0,
                initial_total / gross if gross else 0,
                catastrophic_total / gross if gross else 0,
            ],
        }
    )
    stage_display["Gross cost"] = stage_display["Gross cost"].map(lambda x: f"${x:,.2f}")
    stage_display["Share of gross cost"] = stage_display["Share of gross cost"].map(pct)
    st.dataframe(stage_display, hide_index=True, use_container_width=True)

with right:
    st.subheader("Who funds the modeled drug cost?")
    funding_df = pd.DataFrame(
        {
            "Funding source": [
                "Beneficiary",
                "Plan",
                "Manufacturer discounts",
                "Federal reinsurance",
            ],
            "Amount": [
                beneficiary,
                plan,
                initial_mfg_total + cat_mfg_total,
                federal_total,
            ],
        }
    ).set_index("Funding source")
    st.bar_chart(funding_df, horizontal=True)
    st.metric("Beneficiary share", pct(beneficiary / gross) if gross else "0.0%")
    st.metric("Plan share", pct(plan / gross) if gross else "0.0%")

# Reconciliation check: gross cost should equal all modeled funding sources.
funding_total = beneficiary + plan + initial_mfg_total + cat_mfg_total + federal_total
difference = gross - funding_total
if abs(difference) < 0.01:
    st.success("Allocation check passed: modeled funding sources reconcile to gross drug cost.")
else:
    st.warning(
        f"Allocation difference: {money(difference)}. Review benefit assumptions and allocation logic."
    )

st.divider()

# ---------------------------------------------------------------------
# Drug and therapeutic-class analysis
# ---------------------------------------------------------------------
st.subheader("Medication cost drivers")
drug_col, class_col = st.columns(2)

with drug_col:
    st.markdown("**Highest-cost drugs**")
    if "drug_name" in results.columns:
        top_drugs = (
            results.groupby("drug_name", dropna=False)
            .agg(
                gross_drug_cost=("total_drug_cost", "sum"),
                plan_liability=("plan_liability", "sum"),
                prescription_count=("pharmacy_order_id", "nunique"),
            )
            .sort_values("gross_drug_cost", ascending=False)
            .head(10)
            .reset_index()
        )
        st.dataframe(
            top_drugs.style.format(
                {
                    "gross_drug_cost": "${:,.0f}",
                    "plan_liability": "${:,.0f}",
                    "prescription_count": "{:,.0f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Add a drug_name column to see drug-level results.")

with class_col:
    st.markdown("**Spending by therapeutic class**")
    if "therapeutic_class" in results.columns:
        class_summary = (
            results.groupby("therapeutic_class", dropna=False)
            .agg(
                gross_drug_cost=("total_drug_cost", "sum"),
                plan_liability=("plan_liability", "sum"),
                prescription_count=("pharmacy_order_id", "nunique"),
            )
            .sort_values("gross_drug_cost", ascending=False)
            .head(12)
            .reset_index()
        )
        chart_class = class_summary.set_index("therapeutic_class")[["gross_drug_cost"]]
        st.bar_chart(chart_class, horizontal=True)
        st.dataframe(
            class_summary.style.format(
                {
                    "gross_drug_cost": "${:,.0f}",
                    "plan_liability": "${:,.0f}",
                    "prescription_count": "{:,.0f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Add a therapeutic_class column to see class-level results.")

st.divider()

# ---------------------------------------------------------------------
# High-cost patient analysis
# ---------------------------------------------------------------------
st.subheader("High-cost patient analysis")
high_left, high_right = st.columns([1.2, 1])

with high_left:
    st.markdown("**Highest plan-liability patients**")
    high_cost = members.sort_values("plan_liability", ascending=False).head(10)
    st.dataframe(
        high_cost.style.format(
            {
                "gross_drug_cost": "${:,.0f}",
                "deductible_cost": "${:,.0f}",
                "initial_cost": "${:,.0f}",
                "catastrophic_cost": "${:,.0f}",
                "beneficiary_payment": "${:,.0f}",
                "plan_liability": "${:,.0f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with high_right:
    st.markdown("**Cost concentration**")
    sorted_members = members.sort_values("plan_liability", ascending=False)
    total_plan = sorted_members["plan_liability"].sum()
    concentration_rows = []
    for share in [0.01, 0.05, 0.10]:
        count = max(1, int(np.ceil(len(sorted_members) * share))) if len(sorted_members) else 0
        top_amount = sorted_members.head(count)["plan_liability"].sum()
        concentration_rows.append(
            {
                "Highest-cost group": f"Top {share:.0%}",
                "Patients": count,
                "Share of plan liability": top_amount / total_plan if total_plan else 0,
            }
        )
    concentration = pd.DataFrame(concentration_rows)
    st.dataframe(
        concentration.style.format({"Share of plan liability": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.metric(
        "Patients reaching catastrophic",
        f"{n_cat_members:,}",
        f"{n_cat_members / n_members:.1%} of patients" if n_members else "0%",
    )

st.divider()

# ---------------------------------------------------------------------
# Individual patient explorer
# ---------------------------------------------------------------------
st.subheader("Individual patient explorer")
patient_ids = sorted(results["patient_id"].dropna().unique().tolist(), key=lambda x: str(x))
selected_patient = st.selectbox(
    "Select a patient",
    options=patient_ids,
    format_func=lambda x: str(x),
)

patient_claims = results[results["patient_id"] == selected_patient].copy()
patient_summary = members[members["patient_id"] == selected_patient].iloc[0]

p1, p2, p3, p4 = st.columns(4)
p1.metric("Gross drug cost", money(patient_summary["gross_drug_cost"]))
p2.metric("Beneficiary payment", money(patient_summary["beneficiary_payment"]))
p3.metric("Plan liability", money(patient_summary["plan_liability"]))
p4.metric("Prescription count", f"{int(patient_summary['prescription_count']):,}")

display_cols = [
    c for c in [
        "encounter_id", "pharmacy_order_id", "drug_name", "therapeutic_class",
        "days_supply", "total_drug_cost", "deductible_cost", "initial_cost",
        "catastrophic_cost", "beneficiary_payment", "plan_liability",
    ]
    if c in patient_claims.columns
]
st.markdown("**Prescription-level benefit allocation**")
st.dataframe(
    patient_claims[display_cols].style.format(
        {
            c: "${:,.2f}"
            for c in [
                "total_drug_cost", "deductible_cost", "initial_cost",
                "catastrophic_cost", "beneficiary_payment", "plan_liability",
            ]
            if c in display_cols
        }
    ),
    hide_index=True,
    use_container_width=True,
)

st.divider()

# ---------------------------------------------------------------------
# Downloads and assumptions
# ---------------------------------------------------------------------
st.subheader("Export results")
download1, download2 = st.columns(2)
with download1:
    safe_download(results, "prescription_level_simulation.csv", "Download prescription-level results")
with download2:
    safe_download(members, "patient_level_summary.csv", "Download patient summary")

with st.expander("Model assumptions and interpretation"):
    st.markdown(
        """
        - The model processes each patient's prescriptions in ascending
          `encounter_id` and then `pharmacy_order_id` order. These are ordering
          proxies, not verified service dates.
        - The model treats `total_drug_cost` as the gross cost subject to the
          benefit allocation. Confirm that this is an appropriate cost basis.
        - The deductible is assigned entirely to the beneficiary.
        - Initial coverage uses the selected beneficiary and plan coinsurance
          rates. The initial manufacturer discount is deducted from the plan's
          gross initial-stage share.
        - Catastrophic coverage allocates the selected manufacturer discount
          and federal reinsurance rates, with the remainder assigned to the plan.
        - This is an illustrative simplified model, not an official CMS Part D
          claims adjudication engine. Actual Part D treatment depends on the
          applicable benefit year, drug eligibility, negotiated prices, TrOOP
          rules, and other program-specific provisions.
        - If the uploaded data represents hospital medication orders, the
          Part D benefit simulation should not be interpreted as actual Part D
          liability. The utilization and cost analyses can still be useful.
        """
    )

st.caption(
    "For analytical demonstration only. Validate source data definitions and "
    "benefit rules before using results for pricing, reserving, or coverage decisions."
)
