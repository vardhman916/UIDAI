# data_loader.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple
# ---------- Date parsing (robust to dd-mm-yyyy and yyyy-mm-dd) ----------
def parse_mixed_date(s: pd.Series) -> pd.Series:
    """
    Handles mixed date formats across your files:
    - biometric/enrolment: '01-03-2025' (dd-mm-yyyy)
    - demographic:        '2025-03-01' (yyyy-mm-dd)
    """
    s = s.astype(str).str.strip()

    # If string starts with 4 digits -> likely yyyy-mm-dd
    mask_yyyy = s.str.match(r"^\d{4}[-/]\d{2}[-/]\d{2}$", na=False)

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # yyyy-mm-dd
    if mask_yyyy.any():
        out.loc[mask_yyyy] = pd.to_datetime(s.loc[mask_yyyy], errors="coerce", yearfirst=True)

    # dd-mm-yyyy (or similar)
    if (~mask_yyyy).any():
        out.loc[~mask_yyyy] = pd.to_datetime(s.loc[~mask_yyyy], errors="coerce", dayfirst=True)

    return out


def _optimize_categories(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _safe_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(0, index=df.index)
    return df[present].fillna(0).sum(axis=1)


def load_and_aggregate(
    biometric_path: str,
    demographic_path: str,
    enrolment_path: str,
) -> Dict[str, pd.DataFrame]:
    """
    Returns:
    - state_daily: aggregated by [date, state]
    - district_daily: aggregated by [date, state, district_final]
    Also returns metadata in a dict if needed later.
    """

    # ----------------- Biometric -----------------
    bio_usecols = ["date", "state", "district_final", "bio_age_5_17", "bio_age_17_"]
    bio = pd.read_csv(r'D:\Aadhar_dashboard\data\final_aadhar_biometric_analysis.csv', usecols=bio_usecols)

    bio["date"] = parse_mixed_date(bio["date"])
    bio = bio.dropna(subset=["date", "state", "district_final"])
    bio = _optimize_categories(bio, ["state", "district_final"])

    bio["bio_total"] = _safe_sum(bio, ["bio_age_5_17", "bio_age_17_"])

    bio_district = (
        bio.groupby(["date", "state", "district_final"], observed=True)[["bio_total", "bio_age_5_17", "bio_age_17_"]]
        .sum()
        .reset_index()
    )
    bio_state = (
        bio.groupby(["date", "state"], observed=True)[["bio_total", "bio_age_5_17", "bio_age_17_"]]
        .sum()
        .reset_index()
    )

    # ----------------- Demographic -----------------
    demo_usecols = ["date", "state", "district_final", "demo_age_5_17", "demo_age_17_"]
    demo = pd.read_csv(r'D:\Aadhar_dashboard\data\final_aadhar_demographic_analysis.csv', usecols=demo_usecols)

    demo["date"] = parse_mixed_date(demo["date"])
    demo = demo.dropna(subset=["date", "state", "district_final"])
    demo = _optimize_categories(demo, ["state", "district_final"])

    demo["demo_total"] = _safe_sum(demo, ["demo_age_5_17", "demo_age_17_"])

    demo_district = (
        demo.groupby(["date", "state", "district_final"], observed=True)[["demo_total", "demo_age_5_17", "demo_age_17_"]]
        .sum()
        .reset_index()
    )
    demo_state = (
        demo.groupby(["date", "state"], observed=True)[["demo_total", "demo_age_5_17", "demo_age_17_"]]
        .sum()
        .reset_index()
    )

    # ----------------- Enrolment -----------------
    enr_usecols = ["date", "state", "district_final", "age_0_5", "age_5_17", "age_18_greater"]
    enr = pd.read_csv(r'D:\Aadhar_dashboard\data\final_aadhar_enrolment_analysis.csv', usecols=enr_usecols)

    enr["date"] = parse_mixed_date(enr["date"])
    enr = enr.dropna(subset=["date", "state", "district_final"])
    enr = _optimize_categories(enr, ["state", "district_final"])

    enr["enr_total"] = _safe_sum(enr, ["age_0_5", "age_5_17", "age_18_greater"])

    enr_district = (
        enr.groupby(["date", "state", "district_final"], observed=True)[["enr_total", "age_0_5", "age_5_17", "age_18_greater"]]
        .sum()
        .reset_index()
    )
    enr_state = (
        enr.groupby(["date", "state"], observed=True)[["enr_total", "age_0_5", "age_5_17", "age_18_greater"]]
        .sum()
        .reset_index()
    )

    # ----------------- Merge aggregates -----------------
    district_daily = (
        enr_district.merge(demo_district, on=["date", "state", "district_final"], how="outer")
        .merge(bio_district, on=["date", "state", "district_final"], how="outer")
    )

    state_daily = (
        enr_state.merge(demo_state, on=["date", "state"], how="outer")
        .merge(bio_state, on=["date", "state"], how="outer")
    )

    # Fill numeric NaNs with 0 (safe for sums)
    for df in (district_daily, state_daily):
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)

    return {
        "district_daily": district_daily,
        "state_daily": state_daily,
    }
