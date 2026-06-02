# -*- coding: utf-8 -*-
"""
One-command download of the DeepLabV3+ (ResNet-101, Cityscapes) checkpoint into
./models. The weight file (~450 MB) is too large to commit to git, so it is hosted
as a release asset.

Usage:
    python download_weights.py
    WEIGHTS_URL=https://.../weight.pth.tar python download_weights.py   # override

Set WEIGHTS_URL below (or via env) to your GitHub Release asset URL after you
upload the file once (Releases -> draft a new release -> attach the .pth.tar).
"""
import os
import sys

import config

# Paste your GitHub Release asset URL here (or pass via the WEIGHTS_URL env var).
WEIGHTS_URL = os.getenv(
    "WEIGHTS_URL",
    "https://github.com/<owner>/<repo>/releases/download/v1.0/"
    "best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar",
)
DEST = config.WEIGHT_PATH


def main():
    if os.path.exists(DEST) and os.path.getsize(DEST) > 1_000_000:
        print(f"Already present: {DEST} ({os.path.getsize(DEST)/1e6:.0f} MB)")
        return
    if "<owner>" in WEIGHTS_URL:
        sys.exit("Set WEIGHTS_URL (in this file or env) to your release asset URL first.\n"
                 "Or download manually per models/README.md.")
    import requests
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"Downloading {WEIGHTS_URL}\n      -> {DEST}")
    with requests.get(WEIGHTS_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(DEST, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done/1e6:6.0f} / {total/1e6:.0f} MB ({pct:4.1f}%)", end="")
    print(f"\nDone: {DEST}")


if __name__ == "__main__":
    main()
