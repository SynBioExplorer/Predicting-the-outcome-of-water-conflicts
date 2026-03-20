#!/usr/bin/env python3
"""
Master Download Script - Water Conflict Prediction Project
===========================================================

Orchestrates download of all external datasets required for enriching the
TFDD (Transboundary Freshwater Dispute Database) with climate, socioeconomic,
governance, and water resource indicators.

Datasets:
  1. CRU TS 4.09 - Climate data (precipitation, temperature, PET)
  2. SPEI - Standardised Precipitation-Evapotranspiration Index
  3. World Bank WDI/WGI - Socioeconomic and governance indicators
  4. Polity V - Political regime characteristics
  5. FAO AQUASTAT - Water resource statistics (via World Bank)
  6. ERA5-Land - PLACEHOLDER (requires CDS API key)
  7. EM-DAT - PLACEHOLDER (requires manual registration)
  8. GRanD - PLACEHOLDER (requires Earthdata login)

Usage:
  python scripts/00_download_data.py              # Download all data
  python scripts/00_download_data.py --verify-only # Verify existing downloads
  python scripts/00_download_data.py --status      # Print status summary
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def check_file(path: Path, min_size_bytes: int = 1000) -> dict:
    """Check if a file exists and meets minimum size requirements."""
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    size = path.stat().st_size
    if size < min_size_bytes:
        return {
            "status": "TOO_SMALL",
            "path": str(path),
            "size_bytes": size,
            "min_bytes": min_size_bytes,
        }
    return {
        "status": "OK",
        "path": str(path),
        "size_mb": round(size / (1024**2), 2),
    }


def verify_cru() -> dict:
    """Verify CRU TS 4.09 downloads."""
    results = {}
    cru_dir = DATA_DIR / "cru"
    for var in ["pre", "tmp", "pet"]:
        nc = cru_dir / f"cru_ts4.09.1901.2024.{var}.dat.nc"
        results[f"CRU {var}"] = check_file(nc, min_size_bytes=50_000_000)
    return results


def verify_spei() -> dict:
    """Verify SPEI downloads."""
    results = {}
    spei_dir = DATA_DIR / "spei"
    for name in ["spei03", "spei12"]:
        nc = spei_dir / f"{name}.nc"
        results[f"SPEI {name}"] = check_file(nc, min_size_bytes=50_000_000)
    return results


def verify_wdi() -> dict:
    """Verify WDI/WGI/Polity downloads."""
    results = {}
    wdi_dir = DATA_DIR / "wdi"
    polity_dir = DATA_DIR / "polity"

    results["WDI indicators"] = check_file(
        wdi_dir / "wdi_indicators.parquet", min_size_bytes=100_000
    )
    results["WGI indicators"] = check_file(
        wdi_dir / "wgi_indicators.parquet", min_size_bytes=50_000
    )
    results["Polity V"] = check_file(
        polity_dir / "p5v2018.xls", min_size_bytes=1_000_000
    )
    return results


def verify_aquastat() -> dict:
    """Verify AQUASTAT downloads."""
    results = {}
    aquastat_dir = DATA_DIR / "aquastat"
    results["AQUASTAT"] = check_file(
        aquastat_dir / "aquastat_data.csv", min_size_bytes=10_000
    )
    return results


def verify_placeholders() -> dict:
    """Check status of datasets requiring credentials (placeholders)."""
    results = {}
    results["ERA5-Land"] = {
        "status": "PLACEHOLDER",
        "note": "Requires CDS API key. Set up ~/.cdsapirc and run download_era5.py",
    }
    results["EM-DAT"] = {
        "status": "PLACEHOLDER",
        "note": "Requires manual registration at https://www.emdat.be/",
    }
    results["GRanD"] = {
        "status": "PLACEHOLDER",
        "note": "Requires Earthdata login. Register at https://urs.earthdata.nasa.gov/",
    }
    return results


def print_summary(all_results: dict):
    """Print a formatted summary table of all datasets."""
    print("\n" + "=" * 80)
    print("DATA DOWNLOAD SUMMARY")
    print("=" * 80)

    status_counts = {"OK": 0, "MISSING": 0, "TOO_SMALL": 0, "PLACEHOLDER": 0}

    print(f"\n{'Dataset':<25} {'Status':<12} {'Size (MB)':<12} {'Path/Note'}")
    print("-" * 80)

    for category, results in all_results.items():
        print(f"\n  [{category}]")
        for name, info in results.items():
            status = info["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

            if status == "OK":
                size = info.get("size_mb", "?")
                path = info.get("path", "")
                # Show just filename
                fname = Path(path).name if path else ""
                print(f"  {name:<23} {'OK':<12} {size:<12} {fname}")
            elif status == "PLACEHOLDER":
                note = info.get("note", "")
                print(f"  {name:<23} {'SKIP':<12} {'N/A':<12} {note}")
            else:
                detail = info.get("error", info.get("path", ""))
                print(f"  {name:<23} {'FAIL':<12} {'---':<12} {detail}")

    print("\n" + "-" * 80)
    ok = status_counts.get("OK", 0)
    missing = status_counts.get("MISSING", 0) + status_counts.get("TOO_SMALL", 0)
    placeholder = status_counts.get("PLACEHOLDER", 0)
    total = ok + missing + placeholder

    print(f"  Total: {total} datasets | OK: {ok} | Missing: {missing} | Placeholder: {placeholder}")

    if missing > 0:
        print(f"\n  WARNING: {missing} dataset(s) missing or incomplete.")
        print("  Run individual download scripts to retry:")
        print("    conda run -n water-conflict python3 scripts/download_cru.py")
        print("    conda run -n water-conflict python3 scripts/download_spei.py")
        print("    conda run -n water-conflict python3 scripts/download_wdi.py")
        print("    conda run -n water-conflict python3 scripts/download_aquastat.py")
    else:
        print("\n  All required datasets are present.")

    print("=" * 80)

    return missing == 0


def run_downloads():
    """Run all download scripts sequentially."""
    scripts_dir = PROJECT_ROOT / "scripts"
    download_scripts = [
        ("CRU TS 4.09", "download_cru.py"),
        ("SPEI", "download_spei.py"),
        ("WDI/WGI/Polity", "download_wdi.py"),
        ("AQUASTAT", "download_aquastat.py"),
    ]

    import subprocess

    for name, script in download_scripts:
        script_path = scripts_dir / script
        if not script_path.exists():
            print(f"\n[SKIP] {name}: script not found ({script_path})")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {name} ({script})")
        print(f"{'='*60}")

        result = subprocess.run(
            ["conda", "run", "-n", "water-conflict", "python3", str(script_path)],
            capture_output=False,
        )

        if result.returncode != 0:
            print(f"\n[WARNING] {name} download had issues (exit code {result.returncode})")


def main():
    parser = argparse.ArgumentParser(
        description="Download external datasets for water conflict prediction"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads, don't download anything",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print status summary and exit",
    )
    args = parser.parse_args()

    if args.verify_only or args.status:
        all_results = {
            "Climate": verify_cru(),
            "Drought": verify_spei(),
            "Socioeconomic": verify_wdi(),
            "Water Resources": verify_aquastat(),
            "Credential-Required": verify_placeholders(),
        }
        success = print_summary(all_results)
        sys.exit(0 if success else 1)

    # Run all downloads
    print("Starting data downloads for water conflict prediction project")
    print(f"Output directory: {DATA_DIR}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    run_downloads()

    # Final verification
    print("\n\nFinal Verification")
    print("=" * 60)
    all_results = {
        "Climate": verify_cru(),
        "Drought": verify_spei(),
        "Socioeconomic": verify_wdi(),
        "Water Resources": verify_aquastat(),
        "Credential-Required": verify_placeholders(),
    }
    success = print_summary(all_results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
