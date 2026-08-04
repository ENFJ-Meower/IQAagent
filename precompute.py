"""Precompute content-addressed scalar evidence and fixed-scale maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.cache import atomic_write_cache, image_content_key, load_cache, map_path
from tools.iqa_tools import build_evidence_indices
from tools.perceptual_tools import (
    compute_raw_evidence,
    detail_contact_sheet,
    gradient_magnitude_map,
    noise_residual_map,
)

PRESETS = {
    "koniq": {
        "images_dir": PROJECT_ROOT / "data" / "koniq" / "images",
        "labels_csv": PROJECT_ROOT / "data" / "koniq" / "koniq10k_val.csv",
        "img_col": "img_id",
    },
    "spaq": {
        "images_dir": PROJECT_ROOT / "data" / "spaq" / "images",
        "labels_csv": PROJECT_ROOT / "data" / "spaq" / "spaq_test.csv",
        "img_col": "image_id",
    },
}


def _image_names(
    images_dir: Path,
    labels_csv: Path | None,
    img_col: str,
) -> list[str]:
    if labels_csv is not None:
        try:
            frame = pd.read_csv(labels_csv, usecols=[img_col])
        except ValueError as exc:
            raise ValueError(f"Missing image column: {img_col}") from exc
        return [str(value) for value in frame[img_col].tolist()]

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=PRESETS, default="koniq")
    parser.add_argument("--images-dir")
    parser.add_argument("--labels-csv")
    parser.add_argument("--img-col")
    parser.add_argument("--cache-dir")
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preset = PRESETS[args.dataset]
    images_dir = Path(args.images_dir or preset["images_dir"])
    labels_csv = Path(args.labels_csv or preset["labels_csv"])
    img_col = args.img_col or preset["img_col"]
    cache_dir = Path(args.cache_dir or PROJECT_ROOT / "cache" / args.dataset)

    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    names = _image_names(images_dir, labels_csv, img_col)
    if args.limit is not None:
        names = names[: args.limit]

    entries = load_cache(str(cache_dir))
    maps_dir = cache_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    computed = 0
    skipped = 0
    failed = 0

    for index, image_name in enumerate(names, start=1):
        img_path = images_dir / image_name
        if not img_path.exists():
            failed += 1
            print(f"[{index}/{len(names)}] missing")
            continue

        try:
            image_key = image_content_key(str(img_path))
            map_paths = {
                kind: map_path(str(cache_dir), image_key, kind)
                for kind in ("detail", "gradient", "noise")
            }
            complete = image_key in entries and all(
                path.exists() for path in map_paths.values()
            )
            if complete:
                skipped += 1
                print(f"[{index}/{len(names)}] cached {image_key[:10]}")
                continue

            raw = compute_raw_evidence(str(img_path))
            indices = build_evidence_indices(raw)
            detail_contact_sheet(str(img_path), str(map_paths["detail"]))
            gradient_magnitude_map(str(img_path), str(map_paths["gradient"]))
            noise_residual_map(str(img_path), str(map_paths["noise"]))
            entries[image_key] = {"raw": raw, "indices": indices}
            computed += 1
            print(f"[{index}/{len(names)}] ok {image_key[:10]}")
            if computed % 20 == 0:
                atomic_write_cache(str(cache_dir), entries)
        except Exception as exc:  # noqa: BLE001 - continue after one corrupt image
            failed += 1
            print(f"[{index}/{len(names)}] error {type(exc).__name__}: {exc}")

    atomic_write_cache(str(cache_dir), entries)
    print("=" * 48)
    print(f"Computed: {computed}")
    print(f"Cached  : {skipped}")
    print(f"Failed  : {failed}")
    print(f"Cache   : {cache_dir}")


if __name__ == "__main__":
    main()
