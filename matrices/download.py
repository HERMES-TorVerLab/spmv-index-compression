#!/usr/bin/env python3
"""download.py — fetch the matrices listed in manifest.csv.

Layout:  matrices/<category>/<name>/<name>.mtx

- The archive is streamed and decompressed on the fly; only <name>.mtx is
  written to disk (right-hand sides and auxiliary files are skipped).
- Existing matrices are skipped, so the script can be re-run after a failure.
- --local-cache DIR reuses <name>.tar.gz or <name>.mtx already present there.
- --sizes only issues HEAD requests and reports the compressed download size.
"""
import argparse
import csv
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))


def target_path(root, entry):
    return os.path.join(root, entry["category"], entry["name"], entry["name"] + ".mtx")


def is_matrix_market(path):
    with open(path, "rb") as matrix_file:
        return matrix_file.read(14) == b"%%MatrixMarket"


def extract_matrix(archive_stream, entry, destination):
    """Streams a .tar.gz, writing only the member <name>/<name>.mtx."""
    wanted = entry["name"] + ".mtx"
    partial = destination + ".partial"
    with tarfile.open(fileobj=archive_stream, mode="r|gz") as archive:
        for member in archive:
            if member.isfile() and os.path.basename(member.name) == wanted:
                with archive.extractfile(member) as source, open(partial, "wb") as sink:
                    shutil.copyfileobj(source, sink, length=16 << 20)
                os.replace(partial, destination)
                return True
    return False


def fetch(entry, root, local_cache):
    destination = target_path(root, entry)
    started = time.time()
    if os.path.exists(destination) and is_matrix_market(destination):
        return entry["name"], "present", 0.0
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    if local_cache:
        cached_matrix = os.path.join(local_cache, entry["name"] + ".mtx")
        cached_archive = os.path.join(local_cache, entry["name"] + ".tar.gz")
        if os.path.exists(cached_matrix):
            shutil.copyfile(cached_matrix, destination)
            return entry["name"], "local mtx", time.time() - started
        if os.path.exists(cached_archive):
            with open(cached_archive, "rb") as archive_stream:
                found = extract_matrix(archive_stream, entry, destination)
            return entry["name"], "local tar" if found else "local tar: member missing", time.time() - started

    with urllib.request.urlopen(entry["url"], timeout=120) as response:
        found = extract_matrix(response, entry, destination)
    if not found:
        return entry["name"], "member missing", time.time() - started
    if not is_matrix_market(destination):
        return entry["name"], "not Matrix Market", time.time() - started
    return entry["name"], "downloaded", time.time() - started


def head_size(entry):
    request = urllib.request.Request(entry["url"], method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers.get("Content-Length", 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    parser.add_argument("--root", default=HERE)
    parser.add_argument("--tiers", default="S,M,L,XL")
    parser.add_argument("--names", default=None, help="comma-separated matrices, default all of the tiers")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--local-cache", default=None)
    parser.add_argument("--sizes", action="store_true")
    args = parser.parse_args()

    wanted_tiers = set(args.tiers.split(","))
    with open(args.manifest) as manifest_file:
        entries = [entry for entry in csv.DictReader(manifest_file) if entry["tier"] in wanted_tiers]
    if args.names:
        wanted_names = set(args.names.split(","))
        entries = [entry for entry in entries if entry["name"] in wanted_names]

    if args.sizes:
        total = 0
        per_tier = {}
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(head_size, entry): entry for entry in entries}
            for future in as_completed(futures):
                size = future.result()
                total += size
                tier = futures[future]["tier"]
                per_tier[tier] = per_tier.get(tier, 0) + size
        for tier in ("S", "M", "L", "XL"):
            if tier in per_tier:
                print(f"tier {tier:<3} {per_tier[tier] / 2**30:8.2f} GiB compressed")
        print(f"total    {total / 2**30:8.2f} GiB compressed, {len(entries)} matrices")
        return

    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(fetch, entry, args.root, args.local_cache) for entry in entries]
        for future in as_completed(futures):
            try:
                name, status, seconds = future.result()
            except Exception as error:        # keep going; report at the end
                failures += 1
                print(f"[FAIL] {error}", flush=True)
                continue
            print(f"[{status:>11}] {name} ({seconds:.0f} s)", flush=True)
    print(f"done, {failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
