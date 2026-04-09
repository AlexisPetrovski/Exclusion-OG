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

    df["Excluded"] = df[["F2_Res", "F2_Avg", "F2_ST", "F2_10M"]].any(axis=1)

    df["Exclusion Reason"] = df.apply(lambda r: "; ".join(
        p for p in (
            "Resources > 0" if r["F2_Res"] else None,
            "CAPEX avg > 0" if r["F2_Avg"] else None,
            "Short-Term = Yes" if r["F2_ST"] else None,
            "CAPEX ≥10M = Yes" if r["F2_10M"] else None
        ) if p
    ), axis=1)

    exc = df[df["Excluded"]].copy()
    ret = df[~df["Excluded"]].copy()

    return exc, ret


# ---------------- LEVEL 2 (ALL COMPANIES) ----------------

def filter_all_companies(df):
    df = flatten_multilevel_columns(df)
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]

    df = df.iloc[1:].reset_index(drop=True)

    rename_map = {
        "Company": ["company"],
        "GOGEL Tab": ["gogel tab"],
        "BB Ticker": ["bb ticker"],
        "ISIN equity": ["isin equity"],
        "LEI": ["lei"],
        "Length of Pipelines under Development": ["length"],
        "Liquefaction Capacity (Export)": ["liquefaction"],
        "Regasification Capacity (Import)": ["regasification"],
        "Total Capacity under Development": ["total capacity"],
    }

    df = rename_columns(df, rename_map)

    for c in [
        "Length of Pipelines under Development",
        "Liquefaction Capacity (Export)",
        "Regasification Capacity (Import)",
        "Total Capacity under Development"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce").fillna(0)

    df["Midstream_Flag"] = (
        (df["Length of Pipelines under Development"] > 0) |
        (df["Liquefaction Capacity (Export)"] > 0) |
        (df["Regasification Capacity (Import)"] > 0) |
        (df["Total Capacity under Development"] > 0)
    )

    df["Excluded"] = df["Midstream_Flag"]
    df["Exclusion Reason"] = np.where(df["Midstream_Flag"], "Midstream Expansion > 0", "")

    exc = df[df["Excluded"]].copy()
    ret = df[~df["Excluded"]].copy()

    return exc, ret


# ---------------- EXCEL OUTPUTS ----------------

def to_excel_l1(exc, ret, no_data, investors):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        exc.to_excel(w, "Excluded Level 1", index=False)
        ret.to_excel(w, "Retained Level 1", index=False)
        no_data.to_excel(w, "No Data", index=False)
        investors.to_excel(w, "Investors", index=False)
    buf.seek(0)
    return buf


def to_excel_l2(all_exc, exc1, ret1, exc2, ret2, exc_up, ret_up, investors):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        all_exc.to_excel(w, "All Excluded", index=False)
        exc1.to_excel(w, "Level 1 Excluded", index=False)
        ret1.to_excel(w, "Level 1 Retained", index=False)
        exc2.to_excel(w, "Midstream Excluded", index=False)
        ret2.to_excel(w, "Midstream Retained", index=False)
        exc_up.to_excel(w, "Upstream Excluded", index=False)
        ret_up.to_excel(w, "Upstream Retained", index=False)
        investors.to_excel(w, "Investors", index=False)
    buf.seek(0)
    return buf


# ---------------- STREAMLIT ----------------

def main():
    st.title("Level 1 & Level 2 O&G Filter")
    uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])

    st.sidebar.header("Level 1 Settings")

    sectors = [
        "Hydrocarbons Production (%)","Fracking Revenue","Tar Sand Revenue",
        "Coalbed Methane Revenue","Extra Heavy Oil Revenue","Ultra Deepwater Revenue",
        "Arctic Revenue","Unconventional Production Revenue",
    ]

    sector_excs = {}
    for s in sectors:
        chk = st.sidebar.checkbox(s)
        thr = st.sidebar.text_input(f"{s} threshold", key=s)
        sector_excs[s] = (chk, thr)

    total_thresholds = {}

    if st.sidebar.button("Run Level 1"):
        if not uploaded:
            st.warning("Upload file")
        else:
            exc1, ret1, no1, investors = filter_companies_by_revenue(uploaded, sector_excs, total_thresholds)

            st.session_state["exc1"] = exc1
            st.session_state["ret1"] = ret1
            st.session_state["no1"] = no1
            st.session_state["investors"] = investors

            st.download_button("Download L1", data=to_excel_l1(exc1, ret1, no1, investors), file_name="L1.xlsx")

    st.markdown("---")
    st.header("Level 2")

    if st.button("Run Level 2"):
        if "exc1" not in st.session_state:
            st.error("Run Level 1 first")
            return

        exc1 = st.session_state["exc1"]
        ret1 = st.session_state["ret1"]
        no1 = st.session_state["no1"]
        investors = st.session_state["investors"]

        df_all = pd.read_excel(uploaded, "All Companies", header=[3,4])
        exc_all, ret_all = filter_all_companies(df_all)

        df_up = pd.read_excel(uploaded, "Upstream", header=[3,4])
        exc_up, ret_up = filter_upstream_companies(df_up)

        union = pd.concat([exc_all, exc_up]).drop_duplicates()
        ret2 = pd.concat([ret_all, ret_up]).drop_duplicates()

        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            union.to_excel(w, "All Excluded", index=False)
            exc1.to_excel(w, "Level 1 Excluded", index=False)
            ret1.to_excel(w, "Level 1 Retained", index=False)
            exc_all.to_excel(w, "Midstream Excluded", index=False)
            ret_all.to_excel(w, "Midstream Retained", index=False)
            exc_up.to_excel(w, "Upstream Excluded", index=False)
            ret_up.to_excel(w, "Upstream Retained", index=False)
            investors.to_excel(w, "Investors", index=False)

        st.download_button("Download L1+L2", data=buf.getvalue(), file_name="output.xlsx")


if __name__ == "__main__":
    main()
