# app.py
from __future__ import annotations
import sys
import warnings as _warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

sys.modules.setdefault("warnings", _warnings)

from data_loader import load_and_aggregate
from analytics import add_features, district_anomalies, recommendation_blocks

st.set_page_config(page_title="UIDAI Aadhaar Insights Dashboard", page_icon="🪪", layout="wide")
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"
BANNER_PATH = ASSETS_DIR / "aadhaar_banner.png"
LOGO_PATH = ASSETS_DIR / "aadhaar_logo.png"  # optional


# -------------------- Paths (edit if your CSVs are elsewhere) --------------------
BIOMETRIC_PATH = "data/final_aadhar_biometric_analysis.csv"
DEMOGRAPHIC_PATH = "data/final_aadhar_demographic_analysis.csv"
ENROLMENT_PATH  = "data/final_aadhar_enrolment_analysis.csv"

# -------------------- Cache data load --------------------
@st.cache_data(show_spinner=True)
def get_data():
    data = load_and_aggregate(
        biometric_path=BIOMETRIC_PATH,
        demographic_path=DEMOGRAPHIC_PATH,
        enrolment_path=ENROLMENT_PATH,
    )
    district_daily = add_features(data["district_daily"])
    state_daily = add_features(data["state_daily"])
    return district_daily, state_daily

district_daily, state_daily = get_data()

# -------------------- Sidebar Filters --------------------
st.sidebar.title("Filters")

min_date = min(district_daily["date"].min(), state_daily["date"].min())
max_date = max(district_daily["date"].max(), state_daily["date"].max())

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = min_date, max_date

states = ["All"] + sorted(state_daily["state"].astype(str).unique().tolist())
sel_state = st.sidebar.selectbox("State", states, index=0)

# district list depends on state
if sel_state == "All":
    districts = ["All"] + sorted(district_daily["district_final"].astype(str).unique().tolist())
else:
    districts = ["All"] + sorted(
        district_daily.loc[district_daily["state"].astype(str) == sel_state, "district_final"].astype(str).unique().tolist()
    )

sel_district = st.sidebar.selectbox("District", districts, index=0)

st.sidebar.markdown("---")
metric = st.sidebar.selectbox(
    "Primary metric for anomaly detection",
    ["enr_total", "demo_total", "bio_total"],
    index=0
)

# -------------------- Filtered Data --------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    if sel_state != "All":
        out = out[out["state"].astype(str) == sel_state]
    if "district_final" in out.columns and sel_district != "All":
        out = out[out["district_final"].astype(str) == sel_district]
    return out

f_state = apply_filters(state_daily)
f_district = apply_filters(district_daily)

# -------------------- Header --------------------
st.title("🪪 UIDAI Aadhaar Data-Driven Insights Dashboard")
st.caption(
    "Interactive analytics over Aadhaar enrolment, demographic proxy, and biometric activity datasets. "
    "Designed to support operational monitoring, anomaly detection, and policy-level decision-making."
)

# -------------------- KPI Row --------------------
def kpi_block(label: str, value: float, suffix: str = ""):
    
    st.metric(label, f"{value:,.0f}{suffix}")

k1, k2, k3, k4 = st.columns(4)

enr_sum = float(f_state["enr_total"].sum()) if "enr_total" in f_state else 0
demo_sum = float(f_state["demo_total"].sum()) if "demo_total" in f_state else 0
bio_sum = float(f_state["bio_total"].sum()) if "bio_total" in f_state else 0

with k1:
    kpi_block("Total Enrolment", enr_sum)
with k2:
    kpi_block("Total Demographic (proxy)", demo_sum)
with k3:
    kpi_block("Total Biometric Activity", bio_sum)
with k4:
    # Use overall ratios computed from sums (more stable than mean of ratios)
    enr_per_demo = (enr_sum / demo_sum) if demo_sum > 0 else np.nan
    st.metric("Enrolment / Demo (proxy)", f"{enr_per_demo:.3f}" if not np.isnan(enr_per_demo) else "NA")

st.markdown("---")

# -------------------- Tabs --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Trends", "District Deep-Dive", "Anomalies", "Recommendations"])

# ========== TAB 1: OVERVIEW ==========
with tab1:
    left, right = st.columns([1.2, 1])

    # Top states by enrolment in range
    top_states = (
        f_state.groupby("state", observed=True)[["enr_total", "demo_total", "bio_total"]]
        .sum()
        .reset_index()
        .sort_values("enr_total", ascending=False)
        .head(15)
    )

    with left:
        st.subheader("Top States by Enrolment (selected period)")
        fig = px.bar(top_states, x="state", y="enr_total", title="Enrolment by State")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Age Mix of Enrolment")
        age_cols = ["age_0_5", "age_5_17", "age_18_greater"]
        age_present = [c for c in age_cols if c in f_state.columns]
        if age_present:
            age_sum = f_state[age_present].sum().reset_index()
            age_sum.columns = ["age_bucket", "count"]
            fig2 = px.pie(age_sum, names="age_bucket", values="count", title="Enrolment Age Composition")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Age columns not found for enrolment in filtered data.")

    st.subheader("District leaderboard (high activity)")
    district_table = (
        f_district.groupby(["state", "district_final"], observed=True)[["enr_total", "demo_total", "bio_total"]]
        .sum()
        .reset_index()
        .sort_values("enr_total", ascending=False)
        .head(25)
    )
    st.dataframe(district_table, use_container_width=True)

# ========== TAB 2: TRENDS ==========
with tab2:
    st.subheader("Time Series Trends")

    ts = (
        f_state.groupby("date", observed=True)[["enr_total", "demo_total", "bio_total"]]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(ts, x="date", y="enr_total", title="Enrolment Trend")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.line(ts, x="date", y="bio_total", title="Biometric Activity Trend")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Enrolment vs Biometric (relationship)")
    scatter = (
        f_state.groupby("state", observed=True)[["enr_total", "bio_total", "demo_total"]]
        .sum()
        .reset_index()
    )
    fig = px.scatter(
        scatter, x="enr_total", y="bio_total", size="demo_total", hover_name="state",
        title="State-level: Enrolment vs Biometric (bubble size = demographic proxy)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== TAB 3: DISTRICT DEEP-DIVE ==========
with tab3:
    st.subheader("District Deep-Dive")
    st.caption("Use filters (State + District) to drill down. If District = All, this shows top districts.")

    dd = (
        f_district.groupby(["state", "district_final"], observed=True)[["enr_total", "demo_total", "bio_total"]]
        .sum()
        .reset_index()
    )

    # Rates from sums
    dd["enr_per_demo"] = np.where(dd["demo_total"] > 0, dd["enr_total"] / dd["demo_total"], np.nan)
    dd["bio_per_enr"]  = np.where(dd["enr_total"] > 0, dd["bio_total"] / dd["enr_total"], np.nan)

    c1, c2 = st.columns(2)
    with c1:
        top = dd.sort_values("enr_per_demo", ascending=False).head(20)
        fig = px.bar(top, x="district_final", y="enr_per_demo", color="state", title="Top Districts: Enrolment/Demo (proxy)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top = dd.sort_values("bio_per_enr", ascending=False).head(20)
        fig = px.bar(top, x="district_final", y="bio_per_enr", color="state", title="Top Districts: Biometric/Enrolment (proxy)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("District table (downloadable)")
    st.dataframe(dd.sort_values("enr_total", ascending=False), use_container_width=True)

# ========== TAB 4: ANOMALIES ==========
with tab4:
    st.subheader("Anomaly Detection (Outlier Monitoring)")
    st.caption("Uses z-score to surface unusual spikes. Default is within-state (more fair).")

    # For anomalies, we need district-level daily points (not just sums)
    points = f_district.copy()
    if points.empty:
        st.warning("No data for selected filters/date range.")
    else:
        # Within-state outliers by day
        anom = district_anomalies(points, metric=metric, group="state")

        st.write("Top positive outliers (potential spikes):")
        show_cols = ["date", "state", "district_final", metric, "z", "anomaly_flag"]
        st.dataframe(anom[show_cols].head(30), use_container_width=True)

        # Visualize: top 10 spikes over time
        top10 = anom.head(10).copy()
        fig = px.scatter(
            top10, x="date", y=metric, color="state", hover_name="district_final",
            size=np.clip(top10[metric], 1, None),
            title=f"Top 10 District-Day Spikes for {metric}"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "How to interpret: A high z-score means the district’s value is unusually large compared to other districts "
            "in the same state during the selected period. Use this to prioritize audits, capacity checks, or investigations."
        )

# ========== TAB 5: RECOMMENDATIONS ==========
with tab5:
    st.subheader("Auto Recommendations (Policy + Operations)")

    # Build a single summary row for selected state (or All)
    summary = f_state.groupby("state", observed=True)[["enr_total", "demo_total", "bio_total"]].sum().reset_index()

    if sel_state != "All" and not summary.empty:
        row = summary.iloc[0]
        # Add computed ratios from sums
        row = row.copy()
        row["enr_per_demo"] = (row["enr_total"] / row["demo_total"]) if row["demo_total"] > 0 else np.nan
        row["bio_per_enr"] = (row["bio_total"] / row["enr_total"]) if row["enr_total"] > 0 else np.nan

        st.markdown(f"### Recommendations for **{sel_state}**")
        for r in recommendation_blocks(row):
            st.write(f"✅ {r}")

    else:
        st.markdown("### National-level Recommendations (selected period)")
        total = summary[["enr_total", "demo_total", "bio_total"]].sum()
        enr_per_demo = (total["enr_total"] / total["demo_total"]) if total["demo_total"] > 0 else np.nan
        bio_per_enr = (total["bio_total"] / total["enr_total"]) if total["enr_total"] > 0 else np.nan

        st.write(f"- Enrolment/Demo (proxy): **{enr_per_demo:.3f}**" if not np.isnan(enr_per_demo) else "- Enrolment/Demo (proxy): NA")
        st.write(f"- Biometric/Enrolment (proxy): **{bio_per_enr:.3f}**" if not np.isnan(bio_per_enr) else "- Biometric/Enrolment (proxy): NA")

        st.markdown("#### Suggested Actions")
        st.write("✅ Focus outreach in states/districts showing consistently low enrolment vs demographic proxy.")
        st.write("✅ Use anomaly spikes to trigger operational audits (device issues, reporting gaps, unusual update surges).")
        st.write("✅ Track biometric activity vs enrolment proxy to identify quality/training needs or policy-driven update waves.")

st.markdown("---")
st.caption("Built for UIDAI Hackathon 2026: reproducible analytics, anomaly monitoring, and decision-ready recommendations.")
