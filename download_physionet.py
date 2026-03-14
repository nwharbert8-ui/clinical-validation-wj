#!/usr/bin/env python3
"""Download PhysioNet Sepsis Challenge 2019 Training Set A.
20,336 individual .psv files. Open access, no credentials needed."""

import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/"
DEST_DIR = r"C:\Users\nwhar\repos\clinical-wj-data\training_setA"
MAX_WORKERS = 8  # Parallel downloads
EXPECTED_FILES = 20336


def get_file_list():
    """Get list of .psv files from PhysioNet directory listing."""
    print(f"Fetching file list from {BASE_URL}...")
    req = urllib.request.Request(BASE_URL, headers={'Accept': 'text/html'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode('utf-8')

    # Parse .psv filenames from HTML directory listing
    files = []
    for part in html.split('href="'):
        if '.psv' in part:
            fname = part.split('"')[0].split('/')[-1]
            if fname.endswith('.psv'):
                files.append(fname)

    files = sorted(set(files))
    print(f"  Found {len(files):,} .psv files")
    return files


def download_file(fname):
    """Download a single .psv file."""
    url = BASE_URL + fname
    dest = os.path.join(DEST_DIR, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 100:
        return fname, True, "exists"
    try:
        urllib.request.urlretrieve(url, dest)
        return fname, True, "downloaded"
    except Exception as e:
        return fname, False, str(e)


def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    # Check if already downloaded
    existing = [f for f in os.listdir(DEST_DIR) if f.endswith('.psv')]
    if len(existing) >= EXPECTED_FILES - 100:
        print(f"Already have {len(existing):,} .psv files. Skipping download.")
        return

    # Get file list
    files = get_file_list()
    if not files:
        print("ERROR: Could not get file list. Check URL manually.")
        sys.exit(1)

    # Download in parallel
    print(f"\nDownloading {len(files):,} files to {DEST_DIR}")
    print(f"  Workers: {MAX_WORKERS}")

    t0 = time.time()
    downloaded = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, f): f for f in files}
        for i, future in enumerate(as_completed(futures)):
            fname, success, msg = future.result()
            if success:
                if msg == "downloaded":
                    downloaded += 1
                else:
                    skipped += 1
            else:
                failed += 1

            total_done = downloaded + skipped + failed
            if total_done % 500 == 0 or total_done == len(files):
                elapsed = time.time() - t0
                rate = total_done / elapsed if elapsed > 0 else 0
                eta = (len(files) - total_done) / rate if rate > 0 else 0
                print(f"  {total_done:,}/{len(files):,} "
                      f"(new: {downloaded}, skip: {skipped}, fail: {failed}) "
                      f"  {rate:.0f}/s  ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Downloaded: {downloaded:,}")
    print(f"  Skipped: {skipped:,}")
    print(f"  Failed: {failed:,}")
    print(f"  Total files: {len(os.listdir(DEST_DIR)):,}")


if __name__ == '__main__':
    main()
