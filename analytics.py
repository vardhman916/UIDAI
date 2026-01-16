# analytics.py
from __future__ import annotations
import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Safe divide
    def sdiv(a, b):
        return np.where(b > 0, a / b, np.nan)

    # Key rates (interpretation: not "true population", but proxy comparisons)
    df["enr_per_demo"] = sdiv(df.get("enr_total", 0), df.get("demo_total", 0))
    df["bio_per_demo"] = sdiv(df.get("bio_total", 0), df.get("demo_total", 0))
    df["bio_per_enr"]  = sdiv(df.get("bio_total", 0), df.get("enr_total", 0))

    # Age composition (enrolment)
    total = df.get("enr_total", 0)
    df["share_0_5"] = sdiv(df.get("age_0_5", 0), total)
    df["share_5_17"] = sdiv(df.get("age_5_17", 0), total)
    df["share_18p"] = sdiv(df.get("age_18_greater", 0), total)

    return df


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def district_anomalies(df_district: pd.DataFrame, metric: str = "enr_total", group: str = "state") -> pd.DataFrame:
    """
    Finds outliers using z-score within each state (default) or globally.
    """
    df = df_district.copy()
    if group == "state":
        df["z"] = df.groupby("state", observed=True)[metric].transform(zscore)
    else:
        df["z"] = zscore(df[metric])

    df["anomaly_flag"] = (df["z"].abs() >= 3).astype(int)
    return df.sort_values("z", ascending=False)


def recommendation_blocks(state_row: pd.Series) -> list[str]:
    """
    Create short, judge-friendly actions per state summary row.
    """
    recs = []

    enr = float(state_row.get("enr_total", 0))
    demo = float(state_row.get("demo_total", 0))
    bio = float(state_row.get("bio_total", 0))

    enr_per_demo = state_row.get("enr_per_demo", np.nan)
    bio_per_enr = state_row.get("bio_per_enr", np.nan)

    if demo > 0 and enr_per_demo is not None and not np.isnan(enr_per_demo):
        if enr_per_demo < 0.05:
            recs.append("Low enrolment vs demographic proxy → consider targeted enrolment drives & outreach camps.")
        elif enr_per_demo > 0.25:
            recs.append("High enrolment activity vs demographic proxy → ensure capacity planning (operators, kits, slots).")

    if enr > 0 and bio_per_enr is not None and not np.isnan(bio_per_enr):
        if bio_per_enr < 0.6:
            recs.append("Lower biometric captures per enrolment proxy → review biometric quality, device calibration, operator training.")
        elif bio_per_enr > 1.2:
            recs.append("Higher biometric activity vs enrolment proxy → validate update patterns (biometric updates, re-capture campaigns).")

    if enr == 0 and (bio > 0 or demo > 0):
        recs.append("No enrolment recorded but other activity exists → check reporting gaps or operational issues.")

    if not recs:
        recs.append("No major red flags detected in selected period → continue monitoring and focus on district-level anomalies.")

    return recs
