"""Download the OpenCV Zoo models used by HomePulse Vision and verify pinned checksums.

Usage: python scripts/download_models.py [--dest models] [--print-hashes]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/opencv/{repo}/resolve/{rev}/{file}"

# repo, pinned commit, file, sha256, licence (from the zoo model card / LICENSE file)
MODELS = [
    ("face_detection_yunet", "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310",
     "face_detection_yunet_2023mar.onnx", "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", "MIT"),
    ("text_detection_ppocr", "c299e81aedeaf5d4f741fccccea491fb32db005f",
     "text_detection_en_ppocrv3_2023may.onnx", "03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587", "Apache-2.0"),
    ("text_recognition_crnn", "6487ac545965a2c277aff293e2fdded8b0e6f9b9",
     "text_recognition_CRNN_EN_2021sep.onnx", "a84b1f6e11a65c2d733cb0cc1f014aae3f99051e3f11447dc282faa678eee544", "Apache-2.0"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parent.parent / "models"))
    ap.add_argument("--print-hashes", action="store_true")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    ok = True
    for repo, rev, name, want, licence in MODELS:
        out = dest / name
        if not out.exists() or (want and sha256(out) != want):
            url = HF.format(repo=repo, rev=rev, file=name)
            print(f"downloading {name} ({licence}) from {url}")
            tmp = out.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(out)
        got = sha256(out)
        if args.print_hashes:
            print(f"{name} {got}")
        elif got != want:
            print(f"CHECKSUM MISMATCH {name}: got {got}, want {want}", file=sys.stderr)
            out.unlink()
            ok = False
        else:
            print(f"ok {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
