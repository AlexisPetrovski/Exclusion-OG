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

# ---------------- NEW: Investor Split ----------------

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

# ---------------- Level 1 Exclusion ----------------

def filter_companies_by_revenue(uploaded_file, sector_exclusions, total_thresholds):
    xls = pd.ExcelFile(uploaded_file)
    df = xls.parse("All Companies", header=[3,4])
    df.columns = [" ".join(map(str,c)).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]
    df = remove_equity_from_bb_ticker(df)

    # ---------------- INVESTOR FILTER (NEW) ----------------
    investors, df = split_investors(df)

    rename_map = {
        "Company": ["company name","company"],
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
            df[c]
              .astype(str)
              .str.replace("%","",regex=True)
              .str.replace(",","",regex=True)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for key,info in total_thresholds.items():
        secs = [s for s in info["sectors"] if s in df.columns]
        df[key] = df[secs].sum(axis=1) if secs else 0.0

    # Build Level 1 reasons
    reasons = []
    for _,r in df.iterrows():
        parts = []
        for sector,(flag,thr) in sector_exclusions.items():
            if flag and thr.strip():
                try:
                    if r[sector] > float(thr)/100:
                        parts.append(f"{sector} > {thr}%")
                except:
                    pass
        for key,info in total_thresholds.items():
            t = info.get("threshold","").strip()
            if t:
                try:
                    if r[key] > float(t)/100:
                        parts.append(f"{key} > {t}%")
                except:
                    pass
        reasons.append("; ".join(parts))

    df["Exclusion Reason"] = reasons

    excluded = df[df["Exclusion Reason"]!=""].copy()
    retained = df[df["Exclusion Reason"]==""].copy()

    if "Custom Total 1" in df.columns:
        for d in (excluded, retained, no_data):
            d.rename(columns={"Custom Total 1":"Custom Total Revenue"}, inplace=True)

    return excluded, retained, no_data, investors

# ---------------- Level 2 ----------------

def filter_upstream_companies(df):
    df = flatten_multilevel_columns(df)
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]

    return df, df


def filter_all_companies(df):
    df = flatten_multilevel_columns(df)
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]

    return df, df

# ---------------- Excel Helpers ----------------

def to_excel_l1(exc, ret, no_data, investors):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        exc.to_excel(w, "Excluded Level 1", index=False)
        ret.to_excel(w, "Retained Level 1", index=False)
        no_data.to_excel(w, "L1 No Data", index=False)
        investors.to_excel(w, "Investors", index=False)
    buf.seek(0)
    return buf


def to_excel_l2(all_exc, exc1, exc2, ret1, ret2, exc_up, ret_up, investors):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        all_exc.to_excel(w, "All Excluded Companies", index=False)
        exc1.to_excel(w, "Excluded Level 1", index=False)
        exc2.to_excel(w, "Midstream Excluded", index=False)
        ret1.to_excel(w, "Retained Level 1", index=False)
        ret2.to_excel(w, "Midstream Retained", index=False)
        exc_up.to_excel(w, "Upstream Excluded", index=False)
        ret_up.to_excel(w, "Upstream Retained", index=False)
        investors.to_excel(w, "Investors", index=False)
    buf.seek(0)
    return buf

# ---------------- Streamlit App ----------------

def main():
    st.title("Level 1 & Level 2 Exclusion Filter for O&G")
    uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])

    # Level 1 sidebar
    st.sidebar.header("Level 1 Settings")
    sectors = [
        "Hydrocarbons Production (%)","Fracking Revenue","Tar Sand Revenue",
        "Coalbed Methane Revenue","Extra Heavy Oil Revenue","Ultra Deepwater Revenue",
        "Arctic Revenue","Unconventional Production Revenue",
    ]

    sector_excs = {}
    for s in sectors:
        chk = st.sidebar.checkbox(f"Exclude {s}")
        thr = ""
        if chk:
            thr = st.sidebar.text_input(f"{s} Threshold (%)")
        sector_excs[s] = (chk, thr)

    st.sidebar.header("Custom Total Thresholds")
    total_thresholds = {}
    n = st.sidebar.number_input("How many totals?", 1, 5, 1)
    for i in range(n):
        sels = st.sidebar.multiselect(f"Sectors for Total {i+1}", sectors, key=f"sel{i}")
        thr  = st.sidebar.text_input(f"Threshold {i+1} (%)", key=f"thr{i}")
        if sels and thr:
            total_thresholds[f"Custom Total {i+1}"] = {"sectors":sels,"threshold":thr}

    if st.sidebar.button("Run Level 1 Exclusion"):
        if not uploaded:
            st.warning("Please upload a file first.")
        else:
            exc1, ret1, no1, investors_l1 = filter_companies_by_revenue(uploaded, sector_excs, total_thresholds)
            st.success("Level 1 complete")
            st.download_button(
                "Download Level 1 Results",
                data=to_excel_l1(exc1, ret1, no1, investors_l1),
                file_name="O&G_Level1_Exclusion.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    st.markdown("---")
    st.header("Level 2 Exclusion")

    if st.button("Run Level 2 Exclusion"):
        if not uploaded:
            st.warning("Please upload a file first.")
            return

        df_all = pd.read_excel(uploaded, "All Companies", header=[3,4])
        exc_all, ret_all = filter_all_companies(df_all)

        df_up = pd.read_excel(uploaded, "Upstream", header=[3,4])
        exc_up, ret_up = filter_upstream_companies(df_up)

        #
