#!/usr/bin/env python3
"""
Download FAO AQUASTAT water statistics data.

Source: FAO (Food and Agriculture Organization of the United Nations)
URL: https://data.fao.org/aquastat/

The FAO AQUASTAT bulk downloads are frequently unavailable (403/521 errors).
As a robust fallback, the World Bank distributes the same underlying AQUASTAT
data through the WDI API. These indicators originate from FAO/AQUASTAT and
cover the same variables needed for the water conflict analysis.

Key variables:
  - Water dependency ratio (%)
  - Total water withdrawal (10^9 m3/year)
  - Total renewable water resources (10^9 m3/year)
  - Area equipped for irrigation (1000 ha)
  - Agricultural water withdrawal (% and volume)
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "aquastat"


def download_via_faostat_bulk() -> pd.DataFrame:
    """Try FAO bulk download endpoints."""
    print("\n[AQUASTAT] Attempting FAOSTAT bulk download...")

    bulk_urls = [
        "https://bulks-faostat.fao.org/production/Environment_WaterResources_E_All_Data.zip",
        "https://bulks-faostat.fao.org/production/Environment_WaterResources_E_All_Data_NOFLAG.csv.zip",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    for url in bulk_urls:
        try:
            print(f"  Trying: {url}")
            resp = requests.get(url, timeout=120, stream=True, headers=headers)
            resp.raise_for_status()

            import io
            import zipfile

            z = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            print(f"  ZIP contains: {csv_files}")

            if csv_files:
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f, encoding="latin-1")
                print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                return df
        except Exception as e:
            print(f"  Failed: {e}")

    return None


def download_via_worldbank_batched() -> pd.DataFrame:
    """
    Download AQUASTAT-equivalent indicators from World Bank in small batches.

    The World Bank WDI distributes the same underlying FAO/AQUASTAT data.
    We fetch indicators in small batches (2-3 at a time) to avoid 504 timeouts,
    and use a shorter time range since AQUASTAT data is sparse before 1960.
    """
    print("\n[AQUASTAT] Downloading via World Bank (AQUASTAT-equivalent indicators)...")

    import wbgapi as wb

    # Group indicators into small batches to avoid API timeouts
    indicator_batches = [
        {
            "ER.H2O.INTR.PC": "Renewable internal freshwater resources per capita (m3)",
            "ER.H2O.INTR.K3": "Renewable internal freshwater resources (10^9 m3)",
        },
        {
            "ER.H2O.FWTL.ZS": "Annual freshwater withdrawals, total (% of internal resources)",
            "ER.H2O.FWTL.K3": "Annual freshwater withdrawals, total (10^9 m3)",
        },
        {
            "ER.H2O.FWAG.ZS": "Annual freshwater withdrawals, agriculture (% of total)",
            "ER.H2O.FWIN.ZS": "Annual freshwater withdrawals, industry (% of total)",
            "ER.H2O.FWDM.ZS": "Annual freshwater withdrawals, domestic (% of total)",
        },
        {
            "AG.LND.IRIG.AG.ZS": "Agricultural irrigated land (% of total agricultural land)",
            "SH.H2O.BASW.ZS": "People using at least basic drinking water services (% of pop)",
        },
    ]

    all_dfs = []
    # Use a shorter time range to avoid massive API requests
    time_range = range(1960, 2025)

    for i, batch in enumerate(indicator_batches, 1):
        indicators = list(batch.keys())
        print(f"  Batch {i}/{len(indicator_batches)}: {indicators}")

        for retry in range(3):
            try:
                df = wb.data.DataFrame(
                    indicators,
                    time=time_range,
                    labels=True,
                    columns="series",
                    numericTimeKeys=True,
                )
                df = df.reset_index()
                all_dfs.append(df)
                print(f"    OK: {df.shape[0]} rows, {df.shape[1]} columns")
                break
            except Exception as e:
                print(f"    Attempt {retry+1} failed: {e}")
                if retry < 2:
                    time.sleep(5 * (retry + 1))

    if not all_dfs:
        return None

    # Merge all batches on the common index columns
    result = all_dfs[0]
    for df in all_dfs[1:]:
        # Find common columns (economy, time, Country)
        common_cols = [c for c in result.columns if c in df.columns and c not in [
            col for col in df.columns if col not in result.columns
        ]]
        merge_on = [c for c in ["economy", "Country", "Time"] if c in result.columns and c in df.columns]
        if not merge_on:
            # Fallback: just concatenate along columns
            merge_on = list(set(result.columns) & set(df.columns) - set(
                [c for c in result.columns if c.startswith("ER.") or c.startswith("AG.") or c.startswith("SH.")]
            ))
        try:
            result = result.merge(df, on=merge_on, how="outer")
        except Exception:
            # If merge fails, just concatenate
            result = pd.concat([result, df], axis=0, ignore_index=True)

    print(f"  Combined result: {result.shape[0]} rows, {result.shape[1]} columns")

    # Add source metadata as a comment in the CSV header
    result.attrs["source"] = "World Bank WDI (underlying data from FAO AQUASTAT)"
    result.attrs["note"] = (
        "These indicators originate from FAO AQUASTAT but are distributed "
        "via the World Bank WDI API. They cover the same underlying data."
    )

    return result


def download_aquastat():
    """Try multiple methods to download AQUASTAT data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT_DIR / "aquastat_data.csv"

    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  Already exists: {dest} ({dest.stat().st_size/(1024**2):.1f} MB)")
        return True

    # Method 1: FAOSTAT bulk download (often blocked/unavailable)
    df = download_via_faostat_bulk()

    # Method 2: World Bank batched download (reliable fallback, same data)
    if df is None:
        df = download_via_worldbank_batched()

    if df is not None:
        # Write source metadata as comment, then data
        with open(dest, "w") as f:
            f.write("# Source: FAO AQUASTAT (via World Bank WDI API)\n")
            f.write("# Variables: water resources, withdrawals, irrigation, water services\n")
            f.write(f"# Downloaded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        df.to_csv(dest, index=False, mode="a")
        size_mb = dest.stat().st_size / (1024**2)
        print(f"\n  Saved: {dest} ({size_mb:.2f} MB)")
        return True

    print("\n  ALL download methods failed for AQUASTAT")
    return False


def verify_aquastat() -> dict:
    """Verify downloaded AQUASTAT data."""
    dest = OUTPUT_DIR / "aquastat_data.csv"
    try:
        df = pd.read_csv(dest, comment="#")
        assert len(df) > 100, f"Too few rows: {len(df)}"
        return {
            "status": "OK",
            "path": str(dest),
            "size_mb": round(dest.stat().st_size / (1024**2), 2),
            "shape": df.shape,
            "columns": list(df.columns[:10]),
        }
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


if __name__ == "__main__":
    print("=" * 60)
    print("FAO AQUASTAT Water Statistics Download")
    print("=" * 60)

    result = download_aquastat()

    print("\n" + "-" * 40)
    print("Verification:")
    info = verify_aquastat()
    if info["status"] == "OK":
        print(f"  aquastat: OK ({info['size_mb']} MB, shape={info['shape']})")
        print(f"  Columns: {info['columns']}")
    else:
        print(f"  aquastat: FAILED - {info['error']}")

    sys.exit(0 if info["status"] == "OK" else 1)
