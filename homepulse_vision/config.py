from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("HPV_MODELS_DIR", ROOT / "models"))

BEDROCK_REGION = os.environ.get("HPV_BEDROCK_REGION", "us-east-1")
MODEL_ID = os.environ.get("HPV_MODEL_ID", "us.amazon.nova-2-lite-v1:0")

MAX_UPLOAD_BYTES = int(os.environ.get("HPV_MAX_UPLOAD_BYTES", 4 * 1024 * 1024))
MAX_IMAGE_SIDE = 4096
WORK_MAX_SIDE = 1280


def model_path(name: str) -> str:
    p = MODELS_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run python scripts/download_models.py")
    return str(p)
