#!/usr/bin/env python3
"""
Download SPEI (Standardised Precipitation-Evapotranspiration Index) data.

Source: CSIC (Consejo Superior de Investigaciones Cientificas)
URL: https://spei.csic.es/
Reference: Vicente-Serrano et al. (2010) J. Climate
DOI: 10.20350/digitalCSIC/16497

Version: SPEIbase v2.10 (based on CRU TS 4.08, Jan 1901 - Dec 2023)
Repository: https://digital.csic.es/handle/10261/364137

Files:
  - spei03.nc: 3-month SPEI (short-term drought) ~362 MB
  - spei12.nc: 12-month SPEI (long-term drought) ~360 MB
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# CSIC Digital Repository bitstream URLs (SPEIbase v2.10)
# The bitstream ID matches the SPEI timescale number
CSIC_HANDLE = "10261/364137"
CSIC_BASE = f"https://digital.csic.es/bitstream/{CSIC_HANDLE}"

FILES = {
    "spei03": {
        "filename": "spei03.nc",
        "url": f"{CSIC_BASE}/3/spei03.nc",
        "expected_size_mb": 362,
    },
    "spei12": {
        "filename": "spei12.nc",
        "url": f"{CSIC_BASE}/12/spei12.nc",
        "expected_size_mb": 360,
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "spei"


def download_with_curl(url: str, dest: Path, max_retries: int = 10) -> bool:
    """
    Download a file using curl with automatic resume on failure.
    Much more resilient than Python requests for large files on slow servers.
    """
    for attempt in range(1, max_retries + 1):
        print(f"  Attempt {attempt}/{max_retries}: {url}")
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-C", "-",          # resume from where we left off
                    "-L",               # follow redirects
                    "-o", str(dest),
                    "--retry", "5",
                    "--retry-delay", "10",
                    "--connect-timeout", "30",
                    "--max-time", "0",   # no overall timeout
                    "--speed-limit", "1000",   # abort if <1KB/s
                    "--speed-time", "120",     # for 120 seconds
                    "--progress-bar",
                    url,
                ],
                timeout=3600,  # 1 hour max per attempt
                capture_output=False,
            )

            if result.returncode == 0:
                return True
            elif result.returncode == 33:
                print("  Server doesn't support resume, restarting...")
                if dest.exists():
                    dest.unlink()
            else:
                print(f"  curl exited with code {result.returncode}")

        except subprocess.TimeoutExpired:
            print(f"  Attempt {attempt} timed out")
        except Exception as e:
            print(f"  Error: {e}")

        if attempt < max_retries:
            wait = min(30, 10 * attempt)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    return False


def download_spei():
    """Download all SPEI files from CSIC Digital Repository."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for key, info in FILES.items():
        filename = info["filename"]
        url = info["url"]
        dest = OUTPUT_DIR / filename

        print(f"\n[SPEI] Downloading {key}: {filename}")

        # Skip if already downloaded and large enough
        if dest.exists() and dest.stat().st_size > 100_000_000:
            print(f"  Already exists: {dest} ({dest.stat().st_size/(1024**2):.0f} MB)")
            results[key] = True
            continue

        if not download_with_curl(url, dest):
            print(f"  FAILED to download {key}")
            results[key] = False
            continue

        size_mb = dest.stat().st_size / (1024**2)
        print(f"  Done: {filename} ({size_mb:.0f} MB)")
        results[key] = True

    return results


def verify_spei() -> dict:
    """Verify downloaded SPEI files."""
    import xarray as xr

    verification = {}
    for key, info in FILES.items():
        filename = info["filename"]
        path = OUTPUT_DIR / filename
        try:
            ds = xr.open_dataset(path)
            assert "spei" in ds.data_vars, f"Variable 'spei' not in dataset: {list(ds.data_vars)}"
            assert ds.sizes["time"] > 100, f"Too few time steps: {ds.sizes['time']}"
            size_mb = path.stat().st_size / (1024**2)
            verification[key] = {
                "status": "OK",
                "path": str(path),
                "size_mb": round(size_mb, 1),
                "dims": dict(ds.sizes),
            }
            ds.close()
        except Exception as e:
            verification[key] = {"status": "FAILED", "error": str(e)}

    return verification


if __name__ == "__main__":
    print("=" * 60)
    print("SPEI Drought Index Data Download (SPEIbase v2.10)")
    print("=" * 60)

    results = download_spei()

    print("\n" + "-" * 40)
    print("Verification:")
    verification = verify_spei()
    for key, info in verification.items():
        if info["status"] == "OK":
            print(f"  {key}: OK ({info['size_mb']} MB, dims={info['dims']})")
        else:
            print(f"  {key}: FAILED - {info['error']}")

    all_ok = all(v["status"] == "OK" for v in verification.values())
    sys.exit(0 if all_ok else 1)
