import os
import json
from datetime import datetime


def ensure_dirs():
    os.makedirs("outputs/images", exist_ok=True)
    os.makedirs("outputs/metadata", exist_ok=True)


def save_metadata(meta: dict):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/metadata/{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path
