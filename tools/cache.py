"""Content-addressed cache helpers.

Cache lookup uses image bytes rather than file names, so sample identity cannot
change the selected online evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_VERSION = 2


def image_content_key(img_path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(img_path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(cache_dir: str | None) -> dict[str, Any]:
    if not cache_dir:
        return {}
    path = Path(cache_dir) / "tool_scores.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("version") != CACHE_VERSION:
        return {}
    entries = payload.get("entries", {})
    return entries if isinstance(entries, dict) else {}


def map_path(cache_dir: str, image_key: str, evidence_type: str) -> Path:
    return Path(cache_dir) / "maps" / f"{image_key}.{evidence_type}.png"


def atomic_write_cache(cache_dir: str, entries: dict[str, Any]) -> None:
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tool_scores.json"
    temp_path = directory / "tool_scores.json.tmp"
    payload = {"version": CACHE_VERSION, "entries": entries}
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=True)
    temp_path.replace(path)
