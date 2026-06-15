"""Robust, non-interactive .sens downloader for ScanNet v2 scenes.

Replaces the official download-scannet.py for our use case. Improvements:
- No interactive input() prompts (TOS / skip-sens).
- No release-list fetch (the official script's first, un-timed network call,
  which hangs forever when the TUM server is slow/down).
- Per-read socket timeout so a stalled connection RAISES instead of hanging.
- Retries with backoff; resumable (skips a complete .sens, uses a .part file).
- Constructs the URL directly: <base>/v2/scans/<scene>/<scene>.sens

Honors http(s)_proxy env vars (set by `module load eth_proxy` on Euler).
"""
from __future__ import annotations

import argparse
import os
import socket
import ssl
import time
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
# NOTE: ScanNet v2 reuses the v1 .sens files, so the .sens live under v1/scans
# (the official download-scannet.py does the same swap). v2/scans/*.sens -> 404.
BASE = "http://kaldir.vc.cit.tum.de/scannet/v1/scans"


def fetch_one(scene: str, scans_dir: str, timeout: int, retries: int) -> str:
    final = os.path.join(scans_dir, scene, f"{scene}.sens")
    if os.path.isfile(final) and os.path.getsize(final) > 0:
        print(f"[{scene}] present ({os.path.getsize(final)/1e9:.2f} GB), skip", flush=True)
        return "skip"
    os.makedirs(os.path.dirname(final), exist_ok=True)
    tmp = final + ".part"
    url = f"{BASE}/{scene}/{scene}.sens"
    for attempt in range(1, retries + 1):
        try:
            socket.setdefaulttimeout(timeout)
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=timeout) as r:
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
            if total and done < total:
                raise IOError(f"short read {done}/{total}")
            os.rename(tmp, final)
            print(f"[{scene}] OK {done/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)
            return "ok"
        except Exception as e:  # noqa: BLE001
            print(f"[{scene}] attempt {attempt}/{retries} failed: {repr(e)[:110]}", flush=True)
            try:
                os.remove(tmp)
            except OSError:
                pass
            if attempt < retries:
                time.sleep(min(120, 10 * attempt))
    return "fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True, help="build root; .sens go to <out>/scans/<scene>/")
    ap.add_argument("--start", type=int, default=97)
    ap.add_argument("--end", type=int, default=199)
    ap.add_argument("--timeout", type=int, default=120, help="per-read socket timeout (s)")
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    scans_dir = os.path.join(args.out, "scans")
    os.makedirs(scans_dir, exist_ok=True)
    ok = skip = fail = 0
    failed = []
    for i in range(args.start, args.end + 1):
        scene = f"scene{i:04d}_00"
        res = fetch_one(scene, scans_dir, args.timeout, args.retries)
        if res == "ok":
            ok += 1
        elif res == "skip":
            skip += 1
        else:
            fail += 1
            failed.append(scene)
    print(f"Download done: ok={ok} skip={skip} fail={fail} (range {args.start}..{args.end})", flush=True)
    if failed:
        print("FAILED scenes (server may be down; re-run to resume): " + ", ".join(failed), flush=True)


if __name__ == "__main__":
    main()
