import re
import pandas as pd
import numpy as np
from io import BytesIO
import streamlit as st

# ---------------- Helper Functions ----------------

def ensure_unique_columns(df):
    return df.loc[:, ~df.columns.duplicated()].copy()


def flatten_multilevel_columns(df):
    df.columns = [
        " ".join(str(l).strip() for l in col).strip()
        for col in df.columns
    ]
    return df


def find_column(df, patterns, how="partial", required=True):
    norm_map = {
        col: re.sub(r"\s+", " ", col.strip().lower().replace("\n", " "))
        for col in df.columns
    }
    pats = [
        re.sub(r"\s+", " ", p.strip().lower().replace("\n", " "))
        for p in patterns
    ]

    # exact
    for pat in pats:
        for col, norm in norm_map.items():
            if norm == pat:
                return col

    # partial
    if how == "partial":
        for col, norm in norm_map.items():
            for pat in pats:
                if pat in norm:
                    return col

    # regex
    if how == "regex":
        for pattern in patterns:
            for col in df.columns:
                if re.search(pattern, col, flags=re.IGNORECASE):
                    return col

    if required:
        raise ValueError(f"Could not find a required column among {patterns}\nAvailable: {list(df.columns)}")
    return None


def rename_columns(df, rename_map):
    for new, pats in rename_map.items():
        old = find_column(df, pats, how="partial", required=False)
        if old and old != new:
            df.rename(columns={old: new}, inplace=True)
    return df


def remove_equity_from_bb_ticker(df):
    df = df.copy()
    if "BB Ticker" in df.columns:
        df["BB Ticker"] = (
            df["BB Ticker"]
            .astype(str)
            .str.replace(r"\u00A0", " ", regex=True)
            .str.replace(r"(?i)\s*Equity\s*", "", regex=True)
            .str.strip()
        )
    return df


# ---------------- INVESTOR FILTER ----------------

def split_investors(df):
    seg_col = find_column(
        df,
        ["rystad energy upstream industry segment"],
        how="partial",
        required=False
    )

    if not seg_col:
        return pd.DataFrame(columns=df.columns), df

    mask = df[seg_col].astype(str).str.strip().str.lower().eq("investor")

    investors = df[mask].copy()
    investors["Exclusion Reason"] = "Investor Segment"

    non_investors = df[~mask].copy()
    return investors, non_investors


# ---------------- LEVEL 1 ----------------

def filter_companies_by_revenue(uploaded_file, sector_exclusions, total_thresholds):
    xls = pd.ExcelFile(uploaded_file)
    df = xls.parse("All Companies", header=[3,4])
    df.columns = [" ".join(map(str, c)).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]

    df = remove_equity_from_bb_ticker(df)

    # remove investors early
    investors, df = split_investors(df)

    rename_map = {
        "Company": ["company name", "company"],
        "BB Ticker": ["bb ticker"],
        "ISIN equity": ["isin equity"],
        "LEI": ["lei"],
        "Hydrocarbons Production (%)": ["hydrocarbons production"],
        "Fracking Revenue": ["fracking"],
        "Tar Sand Revenue": ["tar sands"],
        "Coalbed Methane Revenue": ["coalbed methane"],
        "Extra Heavy Oil Revenue": ["extra heavy oil"],
        "Ultra Deepwater Revenue": ["ultra deepwater"],
        "Arctic Revenue": ["arctic"],
        "Unconventional Production Revenue": ["unconventional production"]
    }

    df = rename_columns(df, rename_map)

    needed = list(rename_map.keys())
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    revenue_cols = needed[4:]

    no_data = df[df[revenue_cols].isnull().all(axis=1)].copy()
    df = df.dropna(subset=revenue_cols, how="all")

    for c in revenue_cols:
        df[c] = (
            df[c].astype(str)
            .str.replace("%", "", regex=True)
            .str.replace(",", "", regex=True)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for key, info in total_thresholds.items():
        secs = [s for s in info["sectors"] if s in df.columns]
        df[key] = df[secs].sum(axis=1) if secs else 0.0

    # build reasons
    reasons = []
    for _, r in df.iterrows():
        parts = []

        for sector, (flag, thr) in sector_exclusions.items():
            if flag and thr.strip():
                try:
                    if r[sector] > float(thr) / 100:
                        parts.append(f"{sector} > {thr}%")
                except:
                    pass

        for key, info in total_thresholds.items():
            t = info.get("threshold", "").strip()
            if t:
                try:
                    if r[key] > float(t) / 100:
                        parts.append(f"{key} > {t}%")
                except:
                    pass

        reasons.append("; ".join(parts))

    df["Exclusion Reason"] = reasons

    excluded = df[df["Exclusion Reason"] != ""].copy()
    retained = df[df["Exclusion Reason"] == ""].copy()

    return excluded, retained, no_data, investors


# ---------------- LEVEL 2 (UPSTREAM) ----------------

def filter_upstream_companies(df):
    df = flatten_multilevel_columns(df)
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]

    comp_col = find_column(df, ["company"], required=True)
    res_col = find_column(df, ["resources under development and field evaluation"], required=True)
    capex_avg_col = find_column(df, ["exploration capex 3-year average"], required=True)
    short_col = find_column(df, ["short-term expansion ≥20 mmboe"], required=True)
    capex10_col = find_column(df, ["exploration capex ≥10 musd"], required=True)

    df = df.rename(columns={
        comp_col: "Company",
        res_col: "Resources under Development and Field Evaluation",
        capex_avg_col: "Exploration CAPEX 3-year average",
        short_col: "Short-Term Expansion ≥20 mmboe",
        capex10_col: "Exploration CAPEX ≥10 MUSD",
    })

    for c in ["Resources under Development and Field Evaluation", "Exploration CAPEX 3-year average"]:
        df[c] = pd.to_numeric(
            df[c].astype(str)
            .str.replace(",", "", regex=True)
            .str.replace(r"[^\d.\-]", "", regex=True),
            errors="coerce"
        ).fillna(0)

    df["F2_Res"] = df["Resources under Development and Field Evaluation"] > 0
    df["F2_Avg"] = df["Exploration CAPEX 3-year average"] > 0
    df["F2_ST"] = df["Short-Term Expansion ≥20 mmboe"].astype(str).str.lower().eq("yes")
    df["F2_10M"] = df["Exploration CAPEX ≥10 MUSD"].astype(str).str.lower().eq("yes")

    df["Excluded"] = df[["F2_Res", "F2_Avg", "F2_ST", "F2_10M"]].any(axi
