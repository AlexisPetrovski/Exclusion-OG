# (Full updated code with Investor filtering integrated)

import re
import pandas as pd
import numpy as np
from io import BytesIO
import streamlit as st

# ---------------- Helper Functions ----------------

def ensure_unique_columns(df):
    return df.loc[:, ~df.columns.duplicated()].copy()

def flatten_multilevel_columns(df):
    df.columns = [" ".join(str(l).strip() for l in col).strip() for col in df.columns]
    return df

def find_column(df, patterns, how="partial", required=True):
    norm_map = {col: re.sub(r"\s+", " ", col.strip().lower().replace("\n", " ")) for col in df.columns}
    pats = [re.sub(r"\s+", " ", p.strip().lower().replace("\n", " ")) for p in patterns]

    for pat in pats:
        for col, norm in norm_map.items():
            if norm == pat:
                return col

    if how == "partial":
        for col, norm in norm_map.items():
            for pat in pats:
                if pat in norm:
                    return col

    if how == "regex":
        for pattern in patterns:
            for col in df.columns:
                if re.search(pattern, col, flags=re.IGNORECASE):
                    return col

    if required:
        raise ValueError(f"Column not found: {patterns}")
    return None


def rename_columns(df, rename_map):
    for new, pats in rename_map.items():
        old = find_column(df, pats, how="partial", required=False)
        if old and old != new:
            df.rename(columns={old: new}, inplace=True)
    return df


def remove_equity_from_bb_ticker(df):
    if "BB Ticker" in df.columns:
        df["BB Ticker"] = (
            df["BB Ticker"].astype(str)
            .str.replace(r"\u00A0", " ", regex=True)
            .str.replace(r"(?i)\s*Equity\s*", "", regex=True)
            .str.strip()
        )
    return df

# ---------------- NEW: Investor Split ----------------

def split_investors(df):
    seg_col = find_column(df, ["rystad energy upstream industry segment"], required=False)

    if not seg_col:
        return pd.DataFrame(columns=df.columns), df

    mask = df[seg_col].astype(str).str.strip().str.lower().eq("investor")

    investors = df[mask].copy()
    investors["Exclusion Reason"] = "Investor Segment"

    non_investors = df[~mask].copy()

    return investors, non_investors

# ---------------- Level 1 ----------------

def filter_companies_by_revenue(uploaded_file, sector_exclusions, total_thresholds):
    xls = pd.ExcelFile(uploaded_file)
    df = xls.parse("All Companies", header=[3,4])
    df.columns = [" ".join(map(str,c)).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]

    investors, df = split_investors(df)

    df = remove_equity_from_bb_ticker(df)

    rename_map = {
        "Company": ["company"],
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
        df[c] = pd.to_numeric(df[c].astype(str).str.replace("%", "").str.replace(",", ""), errors="coerce").fillna(0)

    reasons = []
    for _, r in df.iterrows():
        parts = []
        for sector, (flag, thr) in sector_exclusions.items():
            if flag and thr.strip():
                if r[sector] > float(thr)/100:
                    parts.append(f"{sector} > {thr}%")
        reasons.append("; ".join(parts))

    df["Exclusion Reason"] = reasons

    excluded = df[df["Exclusion Reason"] != ""].copy()
    retained = df[df["Exclusion Reason"] == ""].copy()

    return excluded, retained, no_data, investors

# ---------------- Level 2 Filters ----------------

def filter_upstream_companies(df):
    df = flatten_multilevel_columns(df)
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]

    investors, df = split_investors(df)

    df["Excluded"] = False
    df["Exclusion Reason"] = ""

    return df[df["Excluded"]], df[~df["Excluded"]], investors


def filter_all_companies(df):
    df = flatten_multilevel_columns(df)
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]

    investors, df = split_investors(df)

    df["Excluded"] = False
    df["Exclusion Reason"] = ""

    return df[df["Excluded"]], df[~df["Excluded"]], investors

# ---------------- Excel Outputs ----------------

def to_excel_l1(exc, ret, no_data, investors):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        exc.to_excel(w, "Excluded", index=False)
        ret.to_excel(w, "Retained", index=False)
        no_data.to_excel(w, "No Data", index=False)
        investors.to_excel(w, "Investors", index=False)
    buf.seek(0)
    return buf


def to_excel_l2(all_exc, ret, investors):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        all_exc.to_excel(w, "Excluded", index=False)
        ret.to_excel(w, "Retained", index=False)
        investors.to_excel(w, "Investors", index=False)
    buf.seek(0)
    return buf

# ---------------- Streamlit ----------------

def main():
    st.title("O&G Exclusion Tool (with Investor Filter)")
    uploaded = st.file_uploader("Upload Excel", type=["xlsx"])

    if uploaded and st.button("Run Level 1"):
        exc, ret, no_data, investors = filter_companies_by_revenue(uploaded, {}, {})
        st.download_button("Download L1", to_excel_l1(exc, ret, no_data, investors), "L1.xlsx")

    if uploaded and st.button("Run Level 2"):
        df_all = pd.read_excel(uploaded, "All Companies", header=[3,4])
        exc_all, ret_all, inv_all = filter_all_companies(df_all)

        df_up = pd.read_excel(uploaded, "Upstream", header=[3,4])
        exc_up, ret_up, inv_up = filter_upstream_companies(df_up)

        investors = pd.concat([inv_all, inv_up]).drop_duplicates()
        all_exc = pd.concat([exc_all, exc_up])
        ret = pd.concat([ret_all, ret_up])

        st.download_button("Download L2", to_excel_l2(all_exc, ret, investors), "L2.xlsx")

if __name__ == "__main__":
    main()
