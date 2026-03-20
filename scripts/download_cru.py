#!/usr/bin/env python3
"""
Download CRU TS 4.09 climate data (precipitation, temperature, PET).

Source: University of East Anglia Climatic Research Unit
URL: https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/
License: Open Government Licence v3.0

Variables:
  - pre: precipitation (mm/month) ~661 MB compressed
  - tmp: near-surface temperature (deg C) ~422 MB compressed
  - pet: potential evapotranspiration (mm/day) ~69 MB compressed

Note: The CRU server has very low bandwidth (0.2-0.4 MB/s) and frequent
connection drops. This script uses chunked downloads with resume support
via curl to handle the unreliable connection.
"""

import gzip
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "cru"

BASE_URL = "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/cruts.2503051245.v4.09"

VARIABLES = {
    "pre": "cru_ts4.09.1901.2024.pre.dat.nc.gz",
    "tmp": "cru_ts4.09.1901.2024.tmp.dat.nc.gz",
    "pet": "cru_ts4.09.1901.2024.pet.dat.nc.gz",
}


def download_with_curl_resume(url: str, dest: Path, max_retries: int = 10) -> bool:
    """
    Download a file using curl with automatic resume on failure.

    curl -C - automatically resumes from where it left off,
    which is critical for the slow/flaky CRU server.
    """
    for attempt in range(1, max_retries + 1):
        print(f"  Attempt {attempt}/{max_retries}: {url}")
        try:
            # -C - : resume from where we left off
            # -L   : follow redirects
            # --retry 5 --retry-delay 10 : curl-level retries
            # --connect-timeout 30 : connection timeout
            # --max-time 0 : no overall timeout (let it run)
            # --speed-limit 1000 --speed-time 120 : abort if <1KB/s for 120s
            result = subprocess.run(
                [
                    "curl",
                    "-C", "-",        # resume
                    "-L",             # follow redirects
                    "-o", str(dest),
                    "--retry", "3",
                    "--retry-delay", "10",
                    "--connect-timeout", "30",
                    "--max-time", "0",
                    "--speed-limit", "1000",
                    "--speed-time", "120",
                    "--progress-bar",
                    url,
                ],
                timeout=3600,  # 1 hour max per attempt
                capture_output=False,
            )

            if result.returncode == 0:
                return True
            elif result.returncode == 33:
                # HTTP range not supported, start fresh
                print("  Server doesn't support resume, restarting...")
                if dest.exists():
                    dest.unlink()
            else:
                print(f"  curl exited with code {result.returncode}")

        except subprocess.TimeoutExpired:
            print(f"  Attempt {attempt} timed out after 1 hour")
        except Exception as e:
            print(f"  Error: {e}")

        if attempt < max_retries:
            wait = min(30, 10 * attempt)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    return False


def decompress_gz(gz_path: Path, nc_path: Path) -> bool:
    """Decompress a .gz file to .nc, removing the .gz after success."""
    try:
        print(f"  Decompressing {gz_path.name} -> {nc_path.name}")
        with gzip.open(gz_path, "rb") as f_in:
            with open(nc_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        return True
    except Exception as e:
        print(f"  Decompression error: {e}")
        return False


def download_cru():
    """Download all CRU TS 4.09 variables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for var, filename in VARIABLES.items():
        nc_name = filename.replace(".gz", "")
        nc_path = OUTPUT_DIR / nc_name
        gz_path = OUTPUT_DIR / filename
        url = f"{BASE_URL}/{var}/{filename}"

        print(f"\n[CRU] Downloading {var}: {filename}")

        # Skip if already downloaded and decompressed
        if nc_path.exists() and nc_path.stat().st_size > 50_000_000:
            print(f"  Already exists: {nc_path} ({nc_path.stat().st_size/(1024**2):.0f} MB)")
            results[var] = True
            continue

        # Download with curl resume support
        if not download_with_curl_resume(url, gz_path):
            print(f"  FAILED to download {var}")
            results[var] = False
            continue

        # Decompress
        if not decompress_gz(gz_path, nc_path):
            print(f"  FAILED to decompress {var}")
            results[var] = False
            continue

        size_mb = nc_path.stat().st_size / (1024**2)
        print(f"  Done: {nc_path.name} ({size_mb:.0f} MB)")
        results[var] = True

    return results


def verify_cru() -> dict:
    """Verify downloaded CRU files."""
    import xarray as xr

    verification = {}
    for var in VARIABLES:
        nc_name = VARIABLES[var].replace(".gz", "")
        nc_path = OUTPUT_DIR / nc_name
        try:
            ds = xr.open_dataset(nc_path)
            assert var in ds.data_vars, f"Variable '{var}' not in dataset"
            assert ds.sizes["time"] > 100, f"Too few time steps: {ds.sizes['time']}"
            size_mb = nc_path.stat().st_size / (1024**2)
            verification[var] = {
                "status": "OK",
                "path": str(nc_path),
                "size_mb": round(size_mb, 1),
                "dims": dict(ds.sizes),
            }
            ds.close()
        except Exception as e:
            verification[var] = {"status": "FAILED", "error": str(e)}

    return verification


if __name__ == "__main__":
    print("=" * 60)
    print("CRU TS 4.09 Climate Data Download")
    print("=" * 60)

    results = download_cru()

    # Verify
    print("\n" + "-" * 40)
    print("Verification:")
    verification = verify_cru()
    for var, info in verification.items():
        if info["status"] == "OK":
            print(f"  {var}: OK ({info['size_mb']} MB, dims={info['dims']})")
        else:
            print(f"  {var}: FAILED - {info['error']}")

    # Exit code
    all_ok = all(v["status"] == "OK" for v in verification.values())
    sys.exit(0 if all_ok else 1)
