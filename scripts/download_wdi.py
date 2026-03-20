#!/usr/bin/env python3
"""
Download World Bank WDI/WGI indicators and Polity V data.

Sources:
  - World Development Indicators (WDI): via wbgapi Python package (database 2)
  - Worldwide Governance Indicators (WGI): via wbgapi (database 3)
  - Polity V: Center for Systemic Peace

WDI indicators:
  - NY.GDP.PCAP.CD: GDP per capita (current US$)
  - SP.POP.TOTL: Total population
  - MS.MIL.XPND.GD.ZS: Military expenditure (% of GDP)
  - ER.H2O.FWTL.ZS: Annual freshwater withdrawals (% of internal resources)
  - ER.H2O.INTR.PC: Renewable internal freshwater resources per capita (m3)

WGI indicators (database 3, GOV_WGI_ prefix):
  - GOV_WGI_RL.EST: Rule of law estimate
  - GOV_WGI_PV.EST: Political stability and absence of violence estimate
  - GOV_WGI_GE.EST: Government effectiveness estimate
  - GOV_WGI_CC.EST: Control of corruption estimate
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WDI_DIR = PROJECT_ROOT / "data" / "raw" / "wdi"
POLITY_DIR = PROJECT_ROOT / "data" / "raw" / "polity"

WDI_INDICATORS = [
    "NY.GDP.PCAP.CD",
    "SP.POP.TOTL",
    "MS.MIL.XPND.GD.ZS",
    "ER.H2O.FWTL.ZS",
    "ER.H2O.INTR.PC",
]

# WGI database 3 uses GOV_WGI_ prefix for indicator codes
WGI_INDICATORS = [
    "GOV_WGI_RL.EST",
    "GOV_WGI_PV.EST",
    "GOV_WGI_GE.EST",
    "GOV_WGI_CC.EST",
]

POLITY_URL = "https://www.systemicpeace.org/inscr/p5v2018.xls"

# Browser-like headers required by some servers (e.g. systemicpeace.org)
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)",
    "Accept": "*/*",
}


def download_wdi() -> pd.DataFrame:
    """Download WDI indicators using wbgapi."""
    import wbgapi as wb

    print("\n[WDI] Downloading World Development Indicators...")
    print(f"  Indicators: {WDI_INDICATORS}")

    # Ensure we're on the WDI database (database 2)
    wb.db = 2

    df = wb.data.DataFrame(
        WDI_INDICATORS,
        time=range(1960, 2025),
        labels=True,
        columns="series",
        numericTimeKeys=True,
    )
    df = df.reset_index()

    print(f"  Downloaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def download_wgi() -> pd.DataFrame:
    """Download WGI indicators using wbgapi (database 3)."""
    import wbgapi as wb

    print("\n[WGI] Downloading Worldwide Governance Indicators...")
    print(f"  Indicators: {WGI_INDICATORS}")

    # Switch to WGI database
    original_db = wb.db
    wb.db = 3  # WGI database

    try:
        df = wb.data.DataFrame(
            WGI_INDICATORS,
            labels=True,
            columns="series",
            numericTimeKeys=True,
        )
        df = df.reset_index()

        # Rename columns to remove GOV_WGI_ prefix for cleaner downstream use
        rename_map = {col: col.replace("GOV_WGI_", "") for col in df.columns if col.startswith("GOV_WGI_")}
        df = df.rename(columns=rename_map)

        print(f"  Downloaded: {df.shape[0]} rows, {df.shape[1]} columns")
    finally:
        wb.db = original_db  # Restore original database

    return df


def download_polity(max_retries: int = 3) -> bool:
    """Download Polity V Excel file."""
    POLITY_DIR.mkdir(parents=True, exist_ok=True)
    dest = POLITY_DIR / "p5v2018.xls"

    print(f"\n[Polity V] Downloading from {POLITY_URL}")

    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"  Already exists: {dest} ({dest.stat().st_size/(1024**2):.1f} MB)")
        return True

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}/{max_retries}")
            resp = requests.get(POLITY_URL, timeout=120, headers=BROWSER_HEADERS)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            size_mb = dest.stat().st_size / (1024**2)
            print(f"  Done: {dest.name} ({size_mb:.1f} MB)")
            return True
        except (requests.RequestException, OSError) as e:
            print(f"  Error on attempt {attempt}: {e}")
            if dest.exists():
                dest.unlink()
            if attempt < max_retries:
                time.sleep(5 * attempt)

    return False


def run_downloads():
    """Execute all WDI/WGI/Polity downloads."""
    WDI_DIR.mkdir(parents=True, exist_ok=True)
    POLITY_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # WDI
    try:
        wdi_path = WDI_DIR / "wdi_indicators.parquet"
        if wdi_path.exists() and wdi_path.stat().st_size > 10_000:
            print(f"  WDI already exists: {wdi_path}")
            results["wdi"] = True
        else:
            df_wdi = download_wdi()
            df_wdi.to_parquet(wdi_path, index=False)
            print(f"  Saved: {wdi_path} ({wdi_path.stat().st_size/(1024**2):.1f} MB)")
            results["wdi"] = True
    except Exception as e:
        print(f"  WDI FAILED: {e}")
        results["wdi"] = False

    # WGI
    try:
        wgi_path = WDI_DIR / "wgi_indicators.parquet"
        if wgi_path.exists() and wgi_path.stat().st_size > 10_000:
            print(f"  WGI already exists: {wgi_path}")
            results["wgi"] = True
        else:
            df_wgi = download_wgi()
            df_wgi.to_parquet(wgi_path, index=False)
            print(f"  Saved: {wgi_path} ({wgi_path.stat().st_size/(1024**2):.1f} MB)")
            results["wgi"] = True
    except Exception as e:
        print(f"  WGI FAILED: {e}")
        results["wgi"] = False

    # Polity V
    results["polity"] = download_polity()

    return results


def verify_wdi() -> dict:
    """Verify downloaded WDI/WGI/Polity files."""
    verification = {}

    # WDI
    wdi_path = WDI_DIR / "wdi_indicators.parquet"
    try:
        df = pd.read_parquet(wdi_path)
        assert len(df) > 1000, f"Too few rows: {len(df)}"
        verification["wdi"] = {
            "status": "OK",
            "path": str(wdi_path),
            "size_mb": round(wdi_path.stat().st_size / (1024**2), 2),
            "shape": df.shape,
            "columns": list(df.columns[:10]),
        }
    except Exception as e:
        verification["wdi"] = {"status": "FAILED", "error": str(e)}

    # WGI
    wgi_path = WDI_DIR / "wgi_indicators.parquet"
    try:
        df = pd.read_parquet(wgi_path)
        assert len(df) > 100, f"Too few rows: {len(df)}"
        verification["wgi"] = {
            "status": "OK",
            "path": str(wgi_path),
            "size_mb": round(wgi_path.stat().st_size / (1024**2), 2),
            "shape": df.shape,
            "columns": list(df.columns[:10]),
        }
    except Exception as e:
        verification["wgi"] = {"status": "FAILED", "error": str(e)}

    # Polity
    polity_path = POLITY_DIR / "p5v2018.xls"
    try:
        df = pd.read_excel(polity_path)
        assert len(df) > 100, f"Too few rows: {len(df)}"
        verification["polity"] = {
            "status": "OK",
            "path": str(polity_path),
            "size_mb": round(polity_path.stat().st_size / (1024**2), 2),
            "shape": df.shape,
            "columns": list(df.columns[:10]),
        }
    except Exception as e:
        verification["polity"] = {"status": "FAILED", "error": str(e)}

    return verification


if __name__ == "__main__":
    print("=" * 60)
    print("World Bank WDI/WGI + Polity V Data Download")
    print("=" * 60)

    results = run_downloads()

    print("\n" + "-" * 40)
    print("Verification:")
    verification = verify_wdi()
    for key, info in verification.items():
        if info["status"] == "OK":
            print(f"  {key}: OK ({info['size_mb']} MB, shape={info['shape']})")
            print(f"    Columns: {info['columns']}")
        else:
            print(f"  {key}: FAILED - {info['error']}")

    all_ok = all(v["status"] == "OK" for v in verification.values())
    sys.exit(0 if all_ok else 1)
