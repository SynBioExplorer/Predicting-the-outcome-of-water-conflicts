#!/usr/bin/env python3
"""
Feature engineering pipeline for transboundary water conflict prediction.

Inputs:
    data/raw/tfdd/          - TFDD events, treaties, BCU and basin shapefiles
    data/raw/cru/           - CRU TS4.09 gridded climate (precip, PET, temp)
    data/raw/spei/          - SPEI-3 drought index
    data/raw/wdi/           - World Bank WDI + WGI indicators
    data/raw/polity/        - Polity V democracy scores
    data/raw/aquastat/      - AQUASTAT water withdrawal indicators

Output:
    data/processed/events_enriched.parquet

Run with:
    conda run -n water-conflict python3 scripts/01_build_features.py
"""

import warnings
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import regionmask
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJ = Path("/Users/felix/Documents/Predicting-the-outcome-of-water-conflicts")
RAW = PROJ / "data" / "raw"
PROCESSED = PROJ / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# TFDD files live in project root (symlinks in data/raw/tfdd point there)
EVENTS_XLS = PROJ / "EventMaster111710.xls"
TREATIES_XLSX = PROJ / "MasterTreatiesDB_20230213.xlsx"
BCU_SHP = PROJ / "TFDDSpatialDatabase_20240807" / "BCUMaster313_20240807.shp"
BASIN_SHP = PROJ / "TFDDSpatialDatabase_20240807" / "BasinMaster313_20240807.shp"
CRU_PRE = RAW / "cru" / "cru_ts4.09.1901.2024.pre.dat.nc"
CRU_PET = RAW / "cru" / "cru_ts4.09.1901.2024.pet.dat.nc"
CRU_TMP = RAW / "cru" / "cru_ts4.09.1901.2024.tmp.dat.nc"
SPEI_NC = RAW / "spei" / "spei03.nc"
WDI_PQ = RAW / "wdi" / "wdi_indicators.parquet"
WGI_PQ = RAW / "wdi" / "wgi_indicators.parquet"
POLITY_XLS = RAW / "polity" / "p5v2018.xls"
AQUASTAT_CSV = RAW / "aquastat" / "aquastat_data.csv"

# BCU numeric columns that may contain 'null'/'N/A' strings in the shapefile
BCU_NUMERIC_COLS = [
    "Area_km2", "PopDen2022", "Dams_Exist", "Dam_Plnd", "EstDam24",
    "runoff", "withdrawal", "consumpt", "HydroPolTe", "InstitVuln",
    "NumberRipa", "Wetlands_k", "Count_of_t", "Count_of_R",
]

# ---------------------------------------------------------------------------
# Country code crosswalks
# ---------------------------------------------------------------------------
# TFDD CCODE -> WDI/WGI ISO-3166 alpha-3 (economy code)
# None means non-country actor; WDI lookup will produce NaN.
TFDD_TO_WDI: dict[str, str | None] = {
    "CZS": "CZE",   # Czechoslovakia -> Czech Republic
    "GFR": "DEU",   # West Germany -> Germany
    "GDR": "DEU",   # East Germany -> Germany
    "USR": "RUS",   # Soviet Union -> Russia
    "YGF": "SRB",   # Yugoslavia -> Serbia
    "DRV": "VNM",   # North Vietnam -> Vietnam
    "RVN": "VNM",   # South Vietnam -> Vietnam
    "ZAR": "COD",   # Zaire -> DR Congo
    "ROM": "ROU",   # Romania (legacy TFDD code)
    "DEN": "DNK",   # Denmark
    "ANT": "ANT",   # Netherlands Antilles (not in WDI; kept for completeness)
    "PAL": "PSE",   # Palestine
    "PLO": "PSE",   # PLO
    # Non-country actors
    "INT": None, "UNO": None, "EEC": None, "IMF": None,
    "ICJ": None, "OTH": None, "ARL": None, "NTO": None,
    "IBK": None, "UKN": None, "EUX": None, "GUF": None,
    "MON": None, "TWN": None,
}

# TFDD CCODE -> Polity V scode
TFDD_TO_POLITY: dict[str, str | None] = {
    "GBR": "UKG",
    "FRA": "FRN",
    "AUS": "AUL",
    "GER": "GFR",
    "GFR": "GFR",
    "GDR": "GMY",
    "CZS": "CZE",
    "ROM": "ROM",
    "USR": "USR",
    "YGF": "YUG",
    "DRV": "DRV",
    "RVN": "RVN",
    "ZAR": "ZAI",
    "DEN": "DEN",
    # Non-country actors
    "INT": None, "UNO": None, "EEC": None, "OTH": None,
    "ARL": None, "NTO": None, "IBK": None, "UKN": None,
    "EUX": None, "GUF": None, "MON": None, "TWN": None,
    "IMF": None, "ICJ": None,
}


def _map_tfdd_to_wdi(code: str | float) -> str | None:
    if pd.isna(code):
        return None
    return TFDD_TO_WDI.get(str(code), str(code))


def _map_tfdd_to_polity(code: str | float) -> str | None:
    if pd.isna(code):
        return None
    return TFDD_TO_POLITY.get(str(code), str(code))


# ===========================================================================
# Step 1: Load TFDD data
# ===========================================================================
def step1_load_tfdd() -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    print("\n[Step 1] Loading TFDD data...")

    # --- Events ---
    print("  Loading events...")
    events = pd.read_excel(EVENTS_XLS, sheet_name="EventMaster2")
    print(f"  Events raw: {events.shape}")

    events["DATE"] = pd.to_datetime(events["DATE"], errors="coerce")
    events["year"] = events["DATE"].dt.year.astype("Int64")
    events["month"] = events["DATE"].dt.month.astype("Int64")

    n_before = len(events)
    events = events.dropna(subset=["BAR_Scale"]).reset_index(drop=True)
    print(f"  Dropped {n_before - len(events)} rows missing BAR_Scale. Remaining: {len(events)}")

    events["BCode"] = events["BCode"].astype(str).str.strip()

    # --- BCU shapefile ---
    print("  Loading BCU shapefile...")
    bcu = gpd.read_file(BCU_SHP, encoding="utf-8")
    for col in BCU_NUMERIC_COLS:
        if col in bcu.columns:
            bcu[col] = pd.to_numeric(
                bcu[col].astype(str).str.strip().replace(
                    {"null": np.nan, "N/A": np.nan, "nan": np.nan, "": np.nan}
                ),
                errors="coerce",
            )
    print(f"  BCU records: {len(bcu)}")

    # --- Basin shapefile (projected -> WGS84 for regionmask) ---
    print("  Loading basin shapefile...")
    basins = gpd.read_file(BASIN_SHP).to_crs(epsg=4326)
    print(f"  Basins: {len(basins)}")

    # --- Treaties ---
    print("  Loading treaties...")
    treaties = pd.read_excel(TREATIES_XLSX)
    treaties["DateSigned"] = pd.to_datetime(
        treaties["DateSigned"], format="%m/%d/%Y", errors="coerce"
    )
    print(f"  Treaties: {len(treaties)}, valid dates: {treaties['DateSigned'].notna().sum()}")

    return events, bcu, basins, treaties


# ===========================================================================
# Step 2: Dyadic BCU merge (both sides)
# ===========================================================================
def step2_dyadic_bcu_merge(events: pd.DataFrame, bcu: gpd.GeoDataFrame) -> pd.DataFrame:
    print("\n[Step 2] Dyadic BCU merge...")

    bcu_attrs = ["BCCODE", "Basin_Name", "Continent_"] + BCU_NUMERIC_COLS
    bcu_df = bcu[[c for c in bcu_attrs if c in bcu.columns]].copy()

    # Side 1
    df = events.merge(
        bcu_df.rename(columns={c: f"{c}_1" for c in bcu_df.columns if c != "BCCODE"}),
        how="left",
        left_on="BCCODE1",
        right_on="BCCODE",
        suffixes=("", "_bcu1"),
    ).drop(columns=["BCCODE"], errors="ignore")

    # Side 2
    df = df.merge(
        bcu_df.rename(columns={c: f"{c}_2" for c in bcu_df.columns if c != "BCCODE"}),
        how="left",
        left_on="BCCODE2",
        right_on="BCCODE",
        suffixes=("", "_bcu2"),
    ).drop(columns=["BCCODE"], errors="ignore")

    n1 = df["Area_km2_1"].notna().sum()
    n2 = df["Area_km2_2"].notna().sum()
    print(f"  BCU_1 matched: {n1}/{len(df)} | BCU_2 matched: {n2}/{len(df)}")
    return df


# ===========================================================================
# Step 3: Temporal treaty counts (no future leakage)
# ===========================================================================
def step3_treaty_counts(events: pd.DataFrame, treaties: pd.DataFrame) -> pd.Series:
    """Count treaties signed in the same basin BEFORE each event date."""
    print("\n[Step 3] Temporal treaty counts (no future leakage)...")

    tx = treaties[["BCODE", "DateSigned"]].dropna(subset=["DateSigned"])

    # Build lookup: BCODE -> sorted numpy array of dates
    treaty_dates: dict[str, np.ndarray] = {}
    for bcode, grp in tx.groupby("BCODE"):
        treaty_dates[str(bcode)] = np.sort(grp["DateSigned"].values)

    counts = []
    for _, row in events.iterrows():
        bcode = str(row["BCode"])
        event_date = row["DATE"]
        if pd.isna(event_date) or bcode not in treaty_dates:
            counts.append(0)
        else:
            counts.append(int(np.sum(treaty_dates[bcode] < np.datetime64(event_date))))

    result = pd.Series(counts, index=events.index, name="treaties_before_event")
    print(f"  Treaty counts: mean={result.mean():.1f}, max={result.max()}, "
          f"zero={( result == 0).sum()}")
    return result


# ===========================================================================
# Step 4: Zonal aggregation of climate data
# ===========================================================================

def _build_basin_pixel_map(
    basins: gpd.GeoDataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build a mapping from BCODE -> flat pixel indices for each basin in the grid.
    Uses regionmask to rasterise all 313 basins at once.
    """
    masks = regionmask.mask_geopandas(basins, lon, lat)  # DataArray (lat, lon)
    masks_flat = masks.values.flatten()

    pixel_map: dict[str, np.ndarray] = {}
    for i in range(len(basins)):
        bcode = str(basins.iloc[i]["BCODE"])
        idx = np.where(masks_flat == i)[0]
        if len(idx) > 0:
            pixel_map[bcode] = idx

    return pixel_map


def _gridded_basin_means(
    nc_path: Path,
    variable: str,
    basins: gpd.GeoDataFrame,
    target_ym: set[tuple[int, int]],
    fill_threshold: float = -999.0,
) -> pd.DataFrame:
    """
    Compute basin-mean values for each (BCODE, year, month) in target_ym.
    Pre-computes basin masks once and vectorises across all time steps.

    Returns DataFrame with columns: [BCODE, year, month, <variable>]
    """
    if not nc_path.exists():
        print(f"    WARNING: {nc_path.name} not found, skipping {variable}.")
        return pd.DataFrame(columns=["BCODE", "year", "month", variable])

    ds = xr.open_dataset(nc_path)
    da = ds[variable]
    lon = ds["lon"].values
    lat = ds["lat"].values

    # Rasterise all basins
    pixel_map = _build_basin_pixel_map(basins, lon, lat)

    # Select only the time slices we need
    all_times = pd.DatetimeIndex(ds["time"].values)
    sel_mask = np.array([(t.year, t.month) in target_ym for t in all_times])
    sel_times = ds["time"].values[sel_mask]

    if len(sel_times) == 0:
        ds.close()
        return pd.DataFrame(columns=["BCODE", "year", "month", variable])

    arr = da.sel(time=sel_times).values  # (n_time, lat, lon)
    arr_flat = arr.reshape(arr.shape[0], -1)  # (n_time, n_pixels)

    times_dt = pd.DatetimeIndex(sel_times)
    years_arr = times_dt.year.values
    months_arr = times_dt.month.values

    records = []
    for bcode, pixels in pixel_map.items():
        vals = arr_flat[:, pixels]
        vals = np.where(vals < fill_threshold, np.nan, vals)
        basin_mean = np.nanmean(vals, axis=1)
        for i in range(len(times_dt)):
            records.append((bcode, int(years_arr[i]), int(months_arr[i]), float(basin_mean[i])))

    ds.close()
    return pd.DataFrame(records, columns=["BCODE", "year", "month", variable])


def step4_climate_features(
    events: pd.DataFrame,
    basins: gpd.GeoDataFrame,
) -> pd.DataFrame:
    print("\n[Step 4] Zonal aggregation of climate data...")

    # Target (year, month) combinations present in events
    valid_dates = events[events["DATE"].notna()]
    target_ym: set[tuple[int, int]] = set(
        zip(valid_dates["year"].astype(int), valid_dates["month"].astype(int))
    )
    print(f"  Target year-month pairs: {len(target_ym)}")

    # --- Precipitation ---
    print("  CRU precipitation...")
    t0 = time.time()
    pre_df = _gridded_basin_means(CRU_PRE, "pre", basins, target_ym)
    print(f"    Done {time.time()-t0:.1f}s | {len(pre_df):,} basin-month records "
          f"| {pre_df['BCODE'].nunique() if len(pre_df) else 0} basins")

    # --- Precipitation long-term mean and anomaly ---
    if len(pre_df) > 0 and CRU_PRE.exists():
        print("  Computing precipitation anomaly (long-term climatology)...")
        t0 = time.time()
        try:
            ds_pre = xr.open_dataset(CRU_PRE)
            lon = ds_pre["lon"].values
            lat = ds_pre["lat"].values
            pixel_map = _build_basin_pixel_map(basins, lon, lat)

            arr_all = ds_pre["pre"].values  # (1488, lat, lon)
            arr_flat = arr_all.reshape(arr_all.shape[0], -1)

            ltm: dict[str, float] = {}
            for bcode, pixels in pixel_map.items():
                vals = np.where(arr_flat[:, pixels] < -999, np.nan, arr_flat[:, pixels])
                ltm[bcode] = float(np.nanmean(vals))

            pre_df["pre_ltm"] = pre_df["BCODE"].map(ltm)
            pre_df["pre_anomaly"] = pre_df["pre"] - pre_df["pre_ltm"]
            ds_pre.close()
            print(f"    Done {time.time()-t0:.1f}s")
        except Exception as exc:
            print(f"    WARNING: Could not compute anomaly: {exc}")
            pre_df["pre_anomaly"] = np.nan
            pre_df["pre_ltm"] = np.nan

    # --- PET ---
    print("  CRU PET...")
    t0 = time.time()
    pet_df = _gridded_basin_means(CRU_PET, "pet", basins, target_ym)
    print(f"    Done {time.time()-t0:.1f}s")

    # --- SPEI-3 ---
    print("  SPEI-3...")
    t0 = time.time()
    spei_df = _gridded_basin_means(SPEI_NC, "spei", basins, target_ym)
    print(f"    Done {time.time()-t0:.1f}s")

    # --- Temperature (optional) ---
    print("  CRU temperature (optional)...")
    t0 = time.time()
    tmp_df = _gridded_basin_means(CRU_TMP, "tmp", basins, target_ym)
    if len(tmp_df) > 0:
        print(f"    Done {time.time()-t0:.1f}s")
    else:
        print("    Not available, skipped.")

    # --- Combine climate DataFrames ---
    # Outer merge to keep all (BCODE, year, month) combinations
    climate = pre_df.copy()
    for extra_df, extra_cols in [
        (pet_df, ["pet"]),
        (spei_df, ["spei"]),
        (tmp_df, ["tmp"] if "tmp" in tmp_df.columns else []),
    ]:
        if len(extra_df) > 0 and extra_cols:
            climate = climate.merge(
                extra_df[["BCODE", "year", "month"] + extra_cols],
                on=["BCODE", "year", "month"],
                how="outer",
            )

    # --- Attach to events ---
    # Convert event year/month to plain int for join key alignment
    df = events.copy()
    df["_ey"] = df["year"].astype(float).astype("Int64")
    df["_em"] = df["month"].astype(float).astype("Int64")

    climate_int = climate.copy()
    climate_int["year"] = climate_int["year"].astype("Int64")
    climate_int["month"] = climate_int["month"].astype("Int64")

    df = df.merge(
        climate_int.rename(columns={"year": "_ey", "month": "_em"}),
        how="left",
        left_on=["BCode", "_ey", "_em"],
        right_on=["BCODE", "_ey", "_em"],
    ).drop(columns=["BCODE", "_ey", "_em"], errors="ignore")

    n_pre = df["pre"].notna().sum() if "pre" in df.columns else 0
    print(f"  Climate join: {n_pre}/{len(df)} events have precipitation data")
    return df


# ===========================================================================
# Step 5: Country-level features (WDI, WGI, Polity V, AQUASTAT)
# ===========================================================================

def _merge_indicator(
    events: pd.DataFrame,
    indicator: pd.DataFrame,
    indicator_key: str,      # column name in indicator (e.g. 'economy', 'scode')
    year_col: str,           # year column in indicator
    value_cols: list[str],
    side: int,               # 1 or 2
    ccode_mapper,            # callable: TFDD_CCODE -> indicator key
    col_suffix: str,         # appended to value_cols in output
) -> pd.DataFrame:
    """
    Left-join country-level indicators for one side of the dyad.

    Strategy:
      1. Map TFDD CCODE to indicator code.
      2. Exact year join.
      3. For rows still missing after exact join, fall back to the nearest
         preceding year in the indicator (forward fill per country).
      4. All three row categories (matched, unmatched code, NaN code) are
         handled so that the output has exactly the same number of rows as input.
    """
    ccode_col = f"CCODE{side}"
    df = events.copy()
    df["_key"] = df[ccode_col].map(ccode_mapper)
    df["_yr"] = df["year"].astype(float)

    ind = indicator[[indicator_key, year_col] + value_cols].copy()
    ind = ind.dropna(subset=[indicator_key, year_col])
    ind[year_col] = ind[year_col].astype(float)
    ind = ind.sort_values([indicator_key, year_col]).reset_index(drop=True)

    # Exact join on (key, year)
    joined = df.merge(
        ind.rename(columns={indicator_key: "_key", year_col: "_yr"}),
        on=["_key", "_yr"],
        how="left",
    )

    # Forward-fill missing: for each country, find nearest past year
    missing_mask = joined[value_cols[0]].isna() & joined["_key"].notna()
    if missing_mask.any():
        # Build a {key -> DataFrame indexed by year} lookup
        ind_by_key: dict[str, pd.DataFrame] = {
            k: grp.set_index(year_col).sort_index()
            for k, grp in ind.groupby(indicator_key)
        }
        for i in joined[missing_mask].index:
            key = joined.at[i, "_key"]
            yr = joined.at[i, "_yr"]
            if key not in ind_by_key or pd.isna(yr):
                continue
            past = ind_by_key[key]
            past = past[past.index <= yr]
            if len(past) > 0:
                row = past.iloc[-1]
                for col in value_cols:
                    if col in row.index:
                        joined.at[i, col] = row[col]

    # Rename value columns with suffix
    rename = {c: f"{c}_{col_suffix}" for c in value_cols}
    joined = joined.rename(columns=rename)

    # Drop helper columns; retain only original events cols + new value cols
    joined = joined.drop(columns=["_key", "_yr"], errors="ignore")

    # Guard against column duplicates from previous merges
    joined = joined.loc[:, ~joined.columns.duplicated(keep="first")]
    return joined


def step5_country_features(events: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 5] Country-level features (WDI, WGI, Polity, AQUASTAT)...")

    df = events.copy()

    # --- World Development Indicators ---
    print("  WDI...")
    wdi = pd.read_parquet(WDI_PQ)
    wdi["year"] = wdi["time"].astype(int)
    wdi_cols = [
        "NY.GDP.PCAP.CD", "SP.POP.TOTL",
        "MS.MIL.XPND.GD.ZS", "ER.H2O.FWTL.ZS", "ER.H2O.INTR.PC",
    ]
    wdi_sub = wdi[["economy", "year"] + wdi_cols].copy()
    for side in [1, 2]:
        df = _merge_indicator(
            df, wdi_sub, "economy", "year", wdi_cols, side,
            ccode_mapper=_map_tfdd_to_wdi, col_suffix=f"wdi{side}",
        )
    cov = df["NY.GDP.PCAP.CD_wdi1"].notna().sum()
    print(f"  GDP_pc_1 coverage: {cov}/{len(df)}")

    # --- Worldwide Governance Indicators ---
    print("  WGI...")
    wgi = pd.read_parquet(WGI_PQ)
    wgi["year"] = wgi["time"].astype(int)
    wgi_cols = ["CC.EST", "GE.EST", "PV.EST", "RL.EST"]
    wgi_sub = wgi[["economy", "year"] + wgi_cols].copy()
    for side in [1, 2]:
        df = _merge_indicator(
            df, wgi_sub, "economy", "year", wgi_cols, side,
            ccode_mapper=_map_tfdd_to_wdi, col_suffix=f"wgi{side}",
        )
    cov = df["RL.EST_wgi1"].notna().sum()
    print(f"  Rule-of-law_1 coverage: {cov}/{len(df)}")

    # --- Polity V ---
    print("  Polity V...")
    pol = pd.read_excel(POLITY_XLS)
    pol["polity2"] = pd.to_numeric(pol["polity2"], errors="coerce")
    pol.loc[pol["polity2"].isin([-66, -77, -88]), "polity2"] = np.nan
    pol_sub = pol[["scode", "year", "polity2"]].copy()
    for side in [1, 2]:
        df = _merge_indicator(
            df, pol_sub, "scode", "year", ["polity2"], side,
            ccode_mapper=_map_tfdd_to_polity, col_suffix=f"pol{side}",
        )
    cov = df["polity2_pol1"].notna().sum()
    print(f"  Polity2_1 coverage: {cov}/{len(df)}")

    # --- AQUASTAT ---
    print("  AQUASTAT...")
    aq = pd.read_csv(AQUASTAT_CSV, comment="#")
    aq["year"] = pd.to_numeric(aq["time_x"], errors="coerce")
    aq_rename = {
        "ER.H2O.INTR.PC": "aq_intr_pc",
        "ER.H2O.FWTL.ZS": "aq_fwtl_zs",
        "ER.H2O.FWAG.ZS": "aq_fwag_zs",
        "ER.H2O.FWDM.ZS": "aq_fwdm_zs",
        "ER.H2O.FWIN.ZS": "aq_fwin_zs",
    }
    aq = aq.rename(columns=aq_rename)
    aq_cols = list(aq_rename.values())
    aq_sub = aq[["economy", "year"] + aq_cols].copy()
    for side in [1, 2]:
        df = _merge_indicator(
            df, aq_sub, "economy", "year", aq_cols, side,
            ccode_mapper=_map_tfdd_to_wdi, col_suffix=f"aq{side}",
        )
    cov = df["aq_intr_pc_aq1"].notna().sum()
    print(f"  AQUASTAT intr_pc_1 coverage: {cov}/{len(df)}")

    print(f"  Shape after step 5: {df.shape}")
    return df


# ===========================================================================
# Step 6: Dyadic asymmetry features
# ===========================================================================
def step6_asymmetry_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 6] Dyadic asymmetry features...")

    eps = 1e-6

    # Population density log-ratio
    df["pop_ratio"] = np.log(
        df["PopDen2022_1"].clip(lower=eps) / df["PopDen2022_2"].clip(lower=eps)
    ).replace([np.inf, -np.inf], np.nan)

    # Withdrawal log-ratio
    df["withdrawal_ratio"] = np.log(
        df["withdrawal_1"].clip(lower=eps) / df["withdrawal_2"].clip(lower=eps)
    ).replace([np.inf, -np.inf], np.nan)

    # Dam log-ratio (+1 offset to handle zeros)
    dam1 = df["Dams_Exist_1"].fillna(0).clip(lower=0) + 1
    dam2 = df["Dams_Exist_2"].fillna(0).clip(lower=0) + 1
    df["dam_ratio"] = np.log(dam1 / dam2).replace([np.inf, -np.inf], np.nan)

    # Institutional vulnerability difference
    df["instit_vuln_diff"] = df["InstitVuln_1"] - df["InstitVuln_2"]

    # Max hydropolitical tension across the dyad
    df["hydropol_max"] = df[["HydroPolTe_1", "HydroPolTe_2"]].max(axis=1)

    # GDP per capita log-ratio
    gdp1 = df["NY.GDP.PCAP.CD_wdi1"].clip(lower=eps)
    gdp2 = df["NY.GDP.PCAP.CD_wdi2"].clip(lower=eps)
    df["gdp_ratio"] = np.log(gdp1 / gdp2).replace([np.inf, -np.inf], np.nan)

    # Polity democracy score difference
    df["polity_diff"] = df["polity2_pol1"] - df["polity2_pol2"]

    # Water stress (withdrawal % of availability) difference
    df["water_stress_diff"] = df["ER.H2O.FWTL.ZS_wdi1"] - df["ER.H2O.FWTL.ZS_wdi2"]

    print("  Created: pop_ratio, withdrawal_ratio, dam_ratio, instit_vuln_diff, "
          "hydropol_max, gdp_ratio, polity_diff, water_stress_diff")
    return df


# ===========================================================================
# Step 7: Temporal features
# ===========================================================================
def step7_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 7] Temporal features...")

    df = df.sort_values("DATE").reset_index(drop=True)

    # Decade (as integer)
    df["decade"] = (df["year"].astype(float) // 10 * 10).astype("Int64")

    # Cold War binary
    df["cold_war"] = (df["year"].astype(float) < 1991).fillna(False).astype(int)

    # Prior-5-year basin event count and cooperation momentum
    print("  Computing events_prior_5yr and cooperation_momentum (may take ~30s)...")
    events_prior_5yr: list[int | float] = []
    coop_momentum: list[float] = []

    for i, row in df.iterrows():
        bcode = row["BCode"]
        event_date = row["DATE"]
        if pd.isna(event_date):
            events_prior_5yr.append(np.nan)
            coop_momentum.append(np.nan)
            continue

        cutoff = event_date - pd.Timedelta(days=5 * 365.25)
        mask = (
            (df["BCode"] == bcode)
            & (df["DATE"] < event_date)
            & (df["DATE"] >= cutoff)
        )
        prior = df.loc[mask, "BAR_Scale"]
        events_prior_5yr.append(int(mask.sum()))
        coop_momentum.append(float(prior.mean()) if len(prior) > 0 else np.nan)

    df["events_prior_5yr"] = events_prior_5yr
    df["cooperation_momentum"] = coop_momentum

    n_coop = np.sum(~np.isnan(coop_momentum))
    print(f"  events_prior_5yr: mean={np.nanmean(events_prior_5yr):.1f} | "
          f"cooperation_momentum coverage: {n_coop}/{len(df)}")
    return df


# ===========================================================================
# Step 8: Rate-of-change features (Wolf et al. 2003)
# ===========================================================================
def step8_rate_of_change(df: pd.DataFrame, treaties: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 8] Rate-of-change features...")

    # --- Treaty rate: new treaties in basin over prior 5 years ---
    tx = treaties[["BCODE", "DateSigned"]].dropna(subset=["DateSigned"])
    treaty_dates_map: dict[str, np.ndarray] = {}
    for bcode, grp in tx.groupby("BCODE"):
        treaty_dates_map[str(bcode)] = np.sort(grp["DateSigned"].values)

    treaty_rate_5yr: list[int | float] = []
    for _, row in df.iterrows():
        bcode = str(row["BCode"])
        event_date = row["DATE"]
        if pd.isna(event_date) or bcode not in treaty_dates_map:
            treaty_rate_5yr.append(np.nan)
            continue
        dates = treaty_dates_map[bcode]
        cutoff = event_date - pd.Timedelta(days=5 * 365.25)
        n = int(np.sum((dates < np.datetime64(event_date)) & (dates >= np.datetime64(cutoff))))
        treaty_rate_5yr.append(n)

    df["treaty_rate_5yr"] = treaty_rate_5yr

    # --- Event escalation: OLS slope of BAR_Scale over prior 5 events in basin ---
    print("  Computing event escalation slopes...")
    df = df.sort_values("DATE").reset_index(drop=True)
    escalation: list[float] = []
    for i, row in df.iterrows():
        bcode = row["BCode"]
        event_date = row["DATE"]
        if pd.isna(event_date):
            escalation.append(np.nan)
            continue
        mask = (
            (df["BCode"] == bcode)
            & (df["DATE"] < event_date)
            & df["BAR_Scale"].notna()
        )
        prior = df.loc[mask].tail(5)
        if len(prior) < 2:
            escalation.append(np.nan)
        else:
            x = np.arange(len(prior), dtype=float)
            y = prior["BAR_Scale"].values.astype(float)
            slope, *_ = stats.linregress(x, y)
            escalation.append(float(slope))

    df["event_escalation"] = escalation
    esc_cov = int(np.sum(~np.isnan(escalation)))
    print(f"  treaty_rate_5yr mean: {np.nanmean(treaty_rate_5yr):.2f} | "
          f"event_escalation coverage: {esc_cov}/{len(df)}")
    return df


# ===========================================================================
# Step 9: Event features
# ===========================================================================
def step9_event_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 9] Event features...")

    # Issue type: group rare types (<20 events) into 'other'
    issue_counts = df["Issue_Type1"].value_counts()
    rare = set(issue_counts[issue_counts < 20].index)
    df["issue_type_grouped"] = df["Issue_Type1"].apply(
        lambda x: "other" if x in rare else str(x)
    )

    # Bilateral interaction
    df["bilateral"] = (df["NUMBER_OF_Countries"] == 2).astype(int)

    # Fill missing basin count
    df["NUMBER_OF_BASINS"] = df["NUMBER_OF_BASINS"].fillna(1)

    # Continent (preferring BCU side 1)
    if "Continent__1" in df.columns:
        df["Continent"] = df["Continent__1"]
    else:
        df["Continent"] = np.nan

    n_cat = df["issue_type_grouped"].nunique()
    print(f"  issue_type_grouped: {n_cat} categories | bilateral: {df['bilateral'].sum()}/{len(df)}")
    return df


# ===========================================================================
# Step 10: Target variable
# ===========================================================================
def step10_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 10] Target variable...")

    def _encode(bar: float) -> int | float:
        if pd.isna(bar):
            return np.nan
        if bar < 0:
            return 0   # Conflict
        if bar == 0:
            return 1   # Neutral
        if bar <= 3:
            return 2   # Mild cooperation
        return 3       # Strong cooperation

    df["target"] = df["BAR_Scale"].apply(_encode).astype("Int64")

    labels = {
        0: "Conflict (BAR<0)",
        1: "Neutral (BAR=0)",
        2: "Mild cooperation (0<BAR<=3)",
        3: "Strong cooperation (BAR>3)",
    }
    print("  Target distribution:")
    for cls, cnt in df["target"].value_counts().sort_index().items():
        print(f"    {cls} [{labels.get(cls,'?')}]: {cnt} ({100*cnt/len(df):.1f}%)")

    return df


# ===========================================================================
# Summary report
# ===========================================================================
def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"Output shape: {df.shape}")

    def count_group(keywords: list[str]) -> int:
        return sum(1 for c in df.columns if any(k in c for k in keywords))

    bcu_n = count_group(["Area_km2", "PopDen2022", "Dams_Exist", "Dam_Plnd", "EstDam24",
                          "runoff", "withdrawal", "consumpt", "HydroPolTe", "InstitVuln",
                          "NumberRipa", "Wetlands_k", "Count_of"])
    climate_n = count_group(["pre", "pet", "spei", "tmp", "anomaly", "ltm"])
    country_n = count_group(["wdi1", "wdi2", "wgi1", "wgi2", "pol1", "pol2", "aq1", "aq2"])
    asym_n = sum(1 for c in ["pop_ratio", "withdrawal_ratio", "dam_ratio", "instit_vuln_diff",
                              "hydropol_max", "gdp_ratio", "polity_diff", "water_stress_diff"]
                 if c in df.columns)
    temporal_n = sum(1 for c in ["treaties_before_event", "events_prior_5yr", "cooperation_momentum",
                                  "decade", "cold_war", "treaty_rate_5yr", "event_escalation"]
                     if c in df.columns)

    print(f"\nFeature groups:")
    print(f"  BCU dyadic attributes:           {bcu_n}")
    print(f"  Climate (gridded zonal stats):   {climate_n}")
    print(f"  Country-level (WDI/WGI/Pol/AQ): {country_n}")
    print(f"  Dyadic asymmetry:                {asym_n}")
    print(f"  Temporal/historical:              {temporal_n}")

    if "target" in df.columns:
        print("\nTarget distribution (verification):")
        for v, cnt in df["target"].value_counts().sort_index().items():
            print(f"  class {v}: {cnt} ({100*cnt/len(df):.1f}%)")

    print("\nTop-10 columns by missingness (%):")
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    for col, pct in miss.head(10).items():
        print(f"  {col}: {pct:.1f}%")

    high_miss = miss[miss > 20]
    if len(high_miss) > 0:
        print(f"\nColumns with >20% missing: {len(high_miss)}")
    else:
        print("\nNo features with >20% missing data.")

    print(f"\nOutput file: {PROCESSED / 'events_enriched.parquet'}")
    print("=" * 70)


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    t_start = time.time()
    print("=" * 70)
    print("WATER CONFLICT FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # Step 1
    events, bcu, basins, treaties = step1_load_tfdd()

    # Step 2
    df = step2_dyadic_bcu_merge(events, bcu)

    # Step 3
    df["treaties_before_event"] = step3_treaty_counts(df, treaties)

    # Step 4
    df = step4_climate_features(df, basins)

    # Step 5
    df = step5_country_features(df)

    # Step 6
    df = step6_asymmetry_features(df)

    # Step 7
    df = step7_temporal_features(df)

    # Step 8
    df = step8_rate_of_change(df, treaties)

    # Step 9
    df = step9_event_features(df)

    # Step 10
    df = step10_target_variable(df)

    # Drop free-text / identifier columns not useful for modelling
    drop_cols = [
        "EVENT_SUMMARY", "Comments", "SOURCE", "Source_Source",
        "LOCATION", "EVENT_TYPE", "KLL_ISSUE_NUMBER", "EVENT_ISSUE",
        "FBIS_Region", "DOC_DATE", "COUNTRY_LIST", "Macro_Event",
        "GROUPED_ID", "BAR_ID", "SOURCE_ID", "DYAD_CODE",
        "ID1", "EVENT_MASTER", "Interaction_ID", "UNIQUE_ID",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Final shape sanity check
    print(f"\n[Final] shape={df.shape}")

    # Save
    out_path = PROCESSED / "events_enriched.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    print_summary(df)
    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
