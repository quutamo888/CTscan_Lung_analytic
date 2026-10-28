"""
Downloader and Extractor for LUNA16 subset0.zip from Zenodo using requests.
Supports resume and live progress reporting.
"""
import sys
import time
import zipfile
from pathlib import Path
import requests

SUBSET0_URL = "https://zenodo.org/records/2595813/files/subset0.zip"
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "luna16_raw"


def download_subset0():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = RAW_DIR / "subset0.zip"
    part_path = RAW_DIR / "subset0.zip.part"

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    initial_bytes = part_path.stat().st_size if part_path.exists() else 0

    if initial_bytes > 0:
        headers["Range"] = f"bytes={initial_bytes}-"
        print(f"Resuming download from {initial_bytes / (1024**2):.1f} MB...")
    else:
        print(f"Starting download to {dest_path.name}...")

    resp = requests.get(SUBSET0_URL, headers=headers, stream=True, timeout=30)
    if resp.status_code not in (200, 206):
        print(f"[ERROR] HTTP status {resp.status_code}")
        sys.exit(1)

    content_len = resp.headers.get("Content-Length")
    total_size = int(content_len) + initial_bytes if content_len else None

    mode = "ab" if initial_bytes > 0 else "wb"
    downloaded = initial_bytes
    start_time = time.time()
    last_report = start_time

    with open(part_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024 * 4):  # 4MB chunks
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_report >= 5:
                pct = (downloaded / total_size * 100) if total_size else 0
                speed = (downloaded - initial_bytes) / (now - start_time) / (1024**2)
                print(f"[{pct:5.1f}%] {downloaded / (1024**3):.2f}/{total_size/(1024**3):.2f} GB | Speed: {speed:.1f} MB/s", flush=True)
                last_report = now

    part_path.rename(dest_path)
    print(f"\n[SUCCESS] Download completed: {dest_path}")


def extract_subset0():
    dest_path = RAW_DIR / "subset0.zip"
    print(f"Extracting {dest_path.name} to {RAW_DIR}...")
    with zipfile.ZipFile(dest_path, "r") as z:
        items = z.infolist()
        total = len(items)
        for i, item in enumerate(items, start=1):
            # Extract only .mhd and .raw directly into RAW_DIR (strip any subfolder)
            filename = Path(item.filename).name
            if filename.endswith(".mhd") or filename.endswith(".raw"):
                target = RAW_DIR / filename
                with z.open(item) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            if i % 20 == 0 or i == total:
                print(f"Extracted {i}/{total} files...", flush=True)
    print(f"[SUCCESS] All CT scans extracted to {RAW_DIR}")


if __name__ == "__main__":
    if not (RAW_DIR / "subset0.zip").exists():
        download_subset0()
    extract_subset0()
