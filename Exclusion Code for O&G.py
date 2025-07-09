import re
import pandas as pd
import numpy as np
from io import BytesIO
import streamlit as st

# ---------------- Helper Functions ----------------
def ensure_unique_columns(df):
    """
    If a column label appears more than once keep only the **first** copy.
    (All values are identical anyway because they came from the same sheet.)
    """
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

# ---------------- Level 1 Exclusion ----------------

def filter_companies_by_revenue(uploaded_file, sector_exclusions, total_thresholds):
    xls = pd.ExcelFile(uploaded_file)
    df = xls.parse("All Companies", header=[3,4])
    df.columns = [" ".join(map(str,c)).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]
    df = remove_equity_from_bb_ticker(df)

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

    # Fix any '.' company names
    raw = xls.parse("All Companies", header=[3,4]).iloc[:,[6]]
    raw = flatten_multilevel_columns(raw)
    raw = raw.loc[:, ~raw.columns.str.lower().str.startswith("parent company")]
    raw.columns = ["Company"]
    raw["Company"] = raw["Company"].fillna("").astype(str)
    for d in (excluded, retained, no_data):
        d.reset_index(drop=True, inplace=True)
        d["Company"] = d["Company"].replace(".", np.nan)
        d["Company"].fillna(raw["Company"], inplace=True)

    return excluded, retained, no_data

# ---------------- Level-2 — Upstream filter ----------------
def filter_upstream_companies(df):
    df = flatten_multilevel_columns(df)
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]

    # ---- locate the columns (works with any spelling) ----
    comp_col      = find_column(df, ["company"], how="partial", required=True)
    res_col       = find_column(df, ["resources under development and field evaluation"],
                                how="partial", required=True)
    capex_avg_col = find_column(df, ["exploration capex 3-year average"],
                                how="partial", required=True)
    short_col     = find_column(df, ["short-term expansion ≥20 mmboe"],
                                how="partial", required=True)
    capex10_col   = find_column(df, ["exploration capex ≥10 musd"],
                                how="partial", required=True)

    # ---- rename to canonical spellings so every sheet matches ----
    df = df.rename(columns={
        comp_col     : "Company",
        res_col      : "Resources under Development and Field Evaluation",
        capex_avg_col: "Exploration CAPEX 3-year average",
        short_col    : "Short-Term Expansion ≥20 mmboe",
        capex10_col  : "Exploration CAPEX ≥10 MUSD",
    })

    # ---- numeric conversion -------------------------------------------------
    num_cols = [
        "Resources under Development and Field Evaluation",
        "Exploration CAPEX 3-year average",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(
            df[c].astype(str)               # keep strings safe
                 .str.replace(",", "", regex=True)
                 .str.replace(r"[^\d.\-]", "", regex=True),   # strip any units / text
            errors="coerce"
        ).fillna(0)


    # ---- flagging rules -----------------------------------------------------
    df["F2_Res"] = df["Resources under Development and Field Evaluation"] > 0
    df["F2_Avg"] = df["Exploration CAPEX 3-year average"] > 0
    df["F2_ST"]  = df["Short-Term Expansion ≥20 mmboe"].astype(str).str.lower().eq("yes")
    df["F2_10M"] = df["Exploration CAPEX ≥10 MUSD"].astype(str).str.lower().eq("yes")
    df["Excluded"] = df[["F2_Res","F2_Avg","F2_ST","F2_10M"]].any(axis=1)

    df["Exclusion Reason"] = df.apply(
        lambda r: "; ".join(p for p in (
            "Resources under development and field evaluation > 0" if r["F2_Res"] else None,
            "3-yr CAPEX avg > 0" if r["F2_Avg"] else None,
            "Short-Term Expansion = Yes"   if r["F2_ST"]  else None,
            "CAPEX ≥10 MUSD = Yes"         if r["F2_10M"] else None,
        ) if p),
        axis=1
    )

    exc = df[df["Excluded"]].copy()
    ret = df[~df["Excluded"]].copy()
    return exc[[
        "Company",
        "Resources under Development and Field Evaluation",
        "Exploration CAPEX 3-year average",
        "Short-Term Expansion ≥20 mmboe",
        "Exploration CAPEX ≥10 MUSD",
        "Exclusion Reason"
    ]], ret[[
        "Company",
        "Resources under Development and Field Evaluation",
        "Exploration CAPEX 3-year average",
        "Short-Term Expansion ≥20 mmboe",
        "Exploration CAPEX ≥10 MUSD",
        "Exclusion Reason"
    ]]

# ---------------- Excel Helpers ----------------

def to_excel_l1(exc, ret, no_data):
    cols = [
        "Company","BB Ticker","ISIN equity","LEI",
        "Hydrocarbons Production (%)","Fracking Revenue","Tar Sand Revenue",
        "Coalbed Methane Revenue","Extra Heavy Oil Revenue","Ultra Deepwater Revenue",
        "Arctic Revenue","Unconventional Production Revenue","Exclusion Reason","Custom Total Revenue"
    ]
    exc     = remove_equity_from_bb_ticker(exc)
    ret     = remove_equity_from_bb_ticker(ret)
    no_data = remove_equity_from_bb_ticker(no_data)
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        exc.reindex(columns=cols).to_excel(w, "Excluded Level 1", index=False)
        ret.reindex(columns=cols).to_excel(w, "Retained Level 1", index=False)
        no_data.reindex(columns=cols).to_excel(w, "L1 No Data", index=False)
    buf.seek(0)
    return buf

def to_excel_l2(all_exc, exc1, exc2, ret1, ret2, exc_up, ret_up):
    cols = [
        # identity / Level-1 data
        "Company","BB Ticker","ISIN equity","LEI",
        "Hydrocarbons Production (%)","Fracking Revenue","Tar Sand Revenue",
        "Coalbed Methane Revenue","Extra Heavy Oil Revenue","Ultra Deepwater Revenue",
        "Arctic Revenue","Unconventional Production Revenue","Custom Total Revenue",
        # midstream
        "GOGEL Tab","Length of Pipelines under Development","Liquefaction Capacity (Export)",
        "Regasification Capacity (Import)","Total Capacity under Development",
        # upstream
        "Resources under Development and Field Evaluation","Exploration CAPEX 3-year average",
        "Short-Term Expansion ≥20 mmboe","Exploration CAPEX ≥10 MUSD",
        # …and finally the reason string
        "Exclusion Reason"
    ]
    cols = list(dict.fromkeys(cols))   # keep order, drop accidental dups

    # remove duplicates while preserving order
    cols = list(dict.fromkeys(cols))
    for df in (all_exc, exc1, exc2, ret1, ret2, exc_up, ret_up):
        df.update(remove_equity_from_bb_ticker(df))

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        all_exc.reindex(columns=cols).to_excel(w, "All Excluded Companies", index=False)
        exc1   .reindex(columns=cols).to_excel(w, "Excluded Level 1",      index=False)
        exc2   .reindex(columns=cols).to_excel(w, "Midstream Excluded",     index=False)
        ret1   .reindex(columns=cols).to_excel(w, "Retained Level 1",       index=False)
        ret2   .reindex(columns=cols).to_excel(w, "Midstream Retained",     index=False)
        exc_up .reindex(columns=cols).to_excel(w, "Upstream Excluded",      index=False)
        ret_up .reindex(columns=cols).to_excel(w, "Upstream Retained",      index=False)
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
            exc1, ret1, no1 = filter_companies_by_revenue(uploaded, sector_excs, total_thresholds)
            st.success("Level 1 complete")
            st.download_button(
                "Download Level 1 Results",
                data=to_excel_l1(exc1, ret1, no1),
                file_name="O&G_Level1_Exclusion.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    st.markdown("---")
    st.header("Level 2 Exclusion")
    st.write("Applies All-Companies + Upstream filters, merges duplicates, and fills in all data.")

    if st.button("Run Level 2 Exclusion"):
        df_all = pd.read_excel(uploaded, "All Companies", header=[3, 4])
        df_all = ensure_unique_columns(df_all)      #  <-- after reading
        exc_all, ret_all = filter_all_companies(df_all)
       # Upstream L2
        df_up = pd.read_excel(uploaded, "Upstream", header=[3, 4])
        df_up = ensure_unique_columns(df_up)        #  <-- after reading
        exc_up, ret_up = filter_upstream_companies(df_up)
        if not uploaded:
            st.warning("Please upload a file first.")
            return

        # Rerun L1 to get full df_l1_all
        exc1, ret1, no1 = filter_companies_by_revenue(uploaded, sector_excs, total_thresholds)
        df_l1_all = pd.concat([exc1, ret1, no1], ignore_index=True)
        df_l1_all = ensure_unique_columns(df_l1_all)

        # after you read the two Level-2 source sheets
        df_all = ensure_unique_columns(df_all)
        df_up  = ensure_unique_columns(df_up)

        # after you build exc_all / ret_all, exc_up / ret_up
        exc_all = ensure_unique_columns(exc_all)
        exc_up  = ensure_unique_columns(exc_up)

        # All-Companies L2
        df_all = pd.read_excel(uploaded, "All Companies", header=[3,4])
        exc_all, ret_all = filter_all_companies(df_all)

        # Upstream L2
        df_up = pd.read_excel(uploaded, "Upstream", header=[3,4])
        exc_up, ret_up = filter_upstream_companies(df_up)

        # Build All Excluded Companies union
        union = pd.concat([
            exc1[["Company"]],
            exc_all[["Company"]],
            exc_up[["Company"]]
        ]).drop_duplicates()

        # Merge in Level 1 Reason
        df_l1_meta = df_l1_all[["Company","Exclusion Reason"]].rename(columns={"Exclusion Reason":"L1_Reason"})
        union = union.merge(df_l1_meta, on="Company", how="left")

        # Merge in Level 2 All-Companies Reason
        union = union.merge(
            exc_all[["Company","Exclusion Reason"]].rename(columns={"Exclusion Reason":"L2_Reason_AC"}),
            on="Company", how="left"
        )
        # Merge in Level 2 Upstream Reason
        union = union.merge(
            exc_up[["Company","Exclusion Reason"]].rename(columns={"Exclusion Reason":"L2_Reason_UP"}),
            on="Company", how="left"
        )

        # Combine all three reasons into final Exclusion Reason
        union["Exclusion Reason"] = (
            union[["L1_Reason","L2_Reason_AC","L2_Reason_UP"]]
              .fillna("")
              .agg("; ".join, axis=1)
              .str.replace(r"(; )+", "; ")
              .str.strip("; ")
        )
        union.drop(columns=["L1_Reason","L2_Reason_AC","L2_Reason_UP"], inplace=True)
        meta_cols = df_l1_all.columns.difference(
            ["Company", "Exclusion Reason"]
        ).tolist()


        union = (
            union
                .merge(df_l1_all[["Company", *meta_cols]], on="Company", how="left")
                .merge(exc_all  .drop(columns=["Exclusion Reason"]), on="Company", how="left")
                .merge(exc_up   .drop(columns=["Exclusion Reason"]), on="Company", how="left")
        )
        union = union.merge(
            df_l1_all[["Company", "BB Ticker", "ISIN equity", "LEI"]],
            on="Company", how="left", suffixes=("", "_y")
        )
        union = union.drop(columns=[c for c in union.columns if c.endswith("_y")])

        # Retained Level 2
        all_names = set(df_l1_all["Company"])
        exc2_names = set(union["Company"])
        ret2 = pd.DataFrame({"Company":[c for c in all_names if c not in exc2_names]})
        ret2 = ret2.merge(df_l1_all, on="Company", how="left")

        # Upstream full merge
        exc_up_full = (
            exc_up
            .merge(df_l1_all.drop(columns=["Exclusion Reason"]),
            on="Company", how="left")
        )
        ret_up_full = (
            ret_up
                .merge(df_l1_all.drop(columns=["Exclusion Reason"]),
                    on="Company", how="left")
        )

        
        buf = to_excel_l2(
            all_exc=union,
            exc1=exc1,
            exc2=(
                # Build Excluded Level 2 sheet properly with its own L2 reasons
                df_l1_all
                  .merge(exc_all[["Company","Exclusion Reason"]]
                           .rename(columns={"Exclusion Reason":"L2_Reason"}), 
                         on="Company", how="inner")
                  .assign(**{"Exclusion Reason": lambda d: d["L2_Reason"]})
                  .drop(columns=["L2_Reason"])
            ),
            ret1=ret1,
            ret2=ret2,
            exc_up=exc_up_full,
            ret_up=ret_up_full
        )
        st.success("Level 2 complete")
        st.download_button(
            "Download Combined Level 1 & 2 Results",
            data=buf,
            file_name="O&G_Level1_Level2_Exclusion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ---------------- Level-2 — Midstream / “All-Companies” filter ----------------
def filter_all_companies(df: pd.DataFrame):
    """
    Implements Level-2 ‘mid-stream’ screen on the **All Companies** sheet.
    Returns (excluded_df, retained_df) with the canonical columns
    and an ‘Exclusion Reason’ column.
    """
    # 1. tidy columns ---------------------------------------------------------
    df = flatten_multilevel_columns(df)
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]
    df = df.iloc[1:].reset_index(drop=True)          # drop header rows if present
    df = ensure_unique_columns(df)

    # 2. rename the few columns we care about --------------------------------
    rename_map = {
        "Company": ["company"],
        "GOGEL Tab": ["gogel tab"],
        "BB Ticker": ["bb ticker"],
        "ISIN equity": ["isin equity"],
        "LEI": ["lei"],
        "Length of Pipelines under Development": ["length of pipelines"],
        "Liquefaction Capacity (Export)":        ["liquefaction capacity"],
        "Regasification Capacity (Import)":      ["regasification capacity"],
        "Total Capacity under Development":      ["total capacity under development"],
    }
    df = rename_columns(df, rename_map)

    # 3. make sure every canonical column exists -----------------------------
    needed = list(rename_map.keys())
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # 4. numeric conversion for the four capacity columns --------------------
    for c in needed[5:]:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=True),
            errors="coerce"
        ).fillna(0)

    # 5. flag & reason --------------------------------------------------------
    df["Midstream_Flag"] = (
        (df["Length of Pipelines under Development"] > 0) |
        (df["Liquefaction Capacity (Export)"]        > 0) |
        (df["Regasification Capacity (Import)"]      > 0) |
        (df["Total Capacity under Development"]      > 0)
    )
    df["Excluded"] = df["Midstream_Flag"]
    df["Exclusion Reason"] = np.where(
        df["Midstream_Flag"],
        "Midstream Expansion > 0",
        ""
    )

    excluded = df[df["Excluded"]].copy()
    retained = df[~df["Excluded"]].copy()
    return excluded, retained



if __name__ == "__main__":
    main()
