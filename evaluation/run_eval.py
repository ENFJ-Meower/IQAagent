"""Batch evaluation for KonIQ-10k and SPAQ.

Dataset metadata and MOS are confined to this offline evaluation module. The
online pipeline receives only an image path; the adapter converts it to image
bytes before calling the VLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import evaluate, label_to_100, score_100_to_native
from pipeline import IQAPipeline
from skills.vlm_client import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_OPEN_MODEL,
    VLMClient,
)

DATASET_PRESETS = {
    "koniq": {
        "images_dir": PROJECT_ROOT / "data" / "koniq" / "images",
        "labels_csv": PROJECT_ROOT / "data" / "koniq" / "koniq10k_val.csv",
        "img_col": "img_id",
        "mos_col": "img_mos",
        "label_min": 1.0,
        "label_max": 5.0,
    },
    "spaq": {
        "images_dir": PROJECT_ROOT / "data" / "spaq" / "images",
        "labels_csv": PROJECT_ROOT / "data" / "spaq" / "spaq_test.csv",
        "img_col": "image_id",
        "mos_col": "MOS",
        "label_min": 0.0,
        "label_max": 100.0,
    },
}


def _atomic_write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(path)


def _valid_metrics(
    records: list[dict[str, Any]], label_min: float, label_max: float
) -> dict[str, float | int | None]:
    valid = [record for record in records if record.get("status") == "ok"]
    return evaluate(
        [float(record["predicted_score_0_100"]) for record in valid],
        [float(record["mos_native"]) for record in valid],
        label_min,
        label_max,
    )


async def run_async(
    rows: list[tuple[str, float]],
    images_dir: Path,
    pipeline: IQAPipeline,
    workers: int,
    output_path: Path,
    existing: list[dict[str, Any]],
    label_min: float,
    label_max: float,
    preflight: bool = True,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(workers)
    result_records = list(existing)
    write_lock = asyncio.Lock()
    started = time.time()
    newly_completed = 0

    async def process(image_name: str, mos_native: float, slot: int) -> dict[str, Any]:
        nonlocal newly_completed
        img_path = images_dir / image_name
        if not img_path.exists():
            record = {
                "image_id": image_name,
                "status": "missing",
                "error": "image file not found",
                "mos_native": mos_native,
            }
        else:
            try:
                result = await pipeline.assess_async(str(img_path), slot=slot)
                predicted = float(result["score"])
                vlm_valid = bool(result.get("vlm_valid"))
                diagnostics = result.get("vlm_diagnostics", {})
                assessment_diag = diagnostics.get("assessment", {})
                scoring_diag = diagnostics.get("scoring", {})
                api_errors = [
                    str(item.get("error"))
                    for item in (assessment_diag, scoring_diag)
                    if item.get("error")
                ]
                record = {
                    "image_id": image_name,
                    "status": "ok" if vlm_valid else "fallback",
                    "predicted_score_0_100": predicted,
                    "predicted_score_native": score_100_to_native(
                        predicted, label_min, label_max
                    ),
                    "mos_native": mos_native,
                    "mos_0_100": label_to_100(mos_native, label_min, label_max),
                    "distortions": json.dumps(
                        result.get("distortions", []), ensure_ascii=False
                    ),
                    "primary_distortion": result.get("primary_distortion", ""),
                    "reasoning": result.get("reasoning", ""),
                    "router_profile": result.get("router", {}).get("rule_profile", ""),
                    "evidence_anchor": result.get("router", {}).get("evidence_anchor"),
                    "selected_evidence": json.dumps(
                        result.get("router", {}).get("selected_evidence", []),
                        ensure_ascii=False,
                    ),
                    "fusion_weights": json.dumps(
                        result.get("router", {}).get("fusion_weights", {}),
                        ensure_ascii=False,
                    ),
                    "vlm_valid": vlm_valid,
                    "vlm_direct_score": result.get("vlm_direct_score"),
                    "vlm_dimension_score": result.get("vlm_dimension_score"),
                    "vlm_dimension_scores": json.dumps(
                        result.get("vlm_dimension_scores", {}),
                        ensure_ascii=False,
                    ),
                    "assessment_finish_reason": assessment_diag.get(
                        "finish_reason", ""
                    ),
                    "scoring_finish_reason": scoring_diag.get("finish_reason", ""),
                    "assessment_prompt_tokens": assessment_diag.get("prompt_tokens"),
                    "scoring_prompt_tokens": scoring_diag.get("prompt_tokens"),
                    "assessment_response_preview": assessment_diag.get(
                        "text_preview", ""
                    ),
                    "scoring_response_preview": scoring_diag.get("text_preview", ""),
                    "warnings": json.dumps(
                        result.get("warnings", []), ensure_ascii=False
                    ),
                    "error": " | ".join(api_errors),
                }
            except Exception as exc:  # noqa: BLE001 - isolate per-image failures
                record = {
                    "image_id": image_name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "mos_native": mos_native,
                }

        async with write_lock:
            result_records.append(record)
            newly_completed += 1
            done = newly_completed
            if done == 1 or done % 10 == 0:
                elapsed = max(time.time() - started, 1e-6)
                metrics = _valid_metrics(result_records, label_min, label_max)
                srcc = metrics["SRCC"]
                srcc_text = "n/a" if srcc is None else f"{srcc:.4f}"
                print(
                    f"[new {done}/{len(rows)}] valid={metrics['N']} "
                    f"SRCC={srcc_text} MAE_100={metrics['MAE_100']} "
                    f"rate={done / elapsed:.3f} img/s"
                )
                if record.get("status") == "fallback":
                    print(
                        "  VLM fallback detected: "
                        f"{record.get('error') or record.get('warnings')}"
                    )
            if done % 20 == 0:
                _atomic_write_csv(result_records, output_path)
        return record

    async def limited(image_name: str, mos_native: float, slot: int) -> dict[str, Any]:
        async with semaphore:
            return await process(image_name, mos_native, slot)

    pending_rows = list(rows)
    if preflight and pending_rows:
        image_name, mos_native = pending_rows.pop(0)
        first = await limited(image_name, mos_native, 0)
        if first.get("status") in {"fallback", "error"}:
            _atomic_write_csv(result_records, output_path)
            detail = first.get("error") or first.get("warnings")
            raise RuntimeError(
                "VLM preflight failed; stopped before the full dataset. "
                f"Diagnostic: {detail}"
            )

    tasks = [
        limited(image_name, mos_native, index % workers)
        for index, (image_name, mos_native) in enumerate(pending_rows)
    ]
    if tasks:
        await asyncio.gather(*tasks)
    return result_records


def _resolve_dataset(args: argparse.Namespace) -> dict[str, Any]:
    preset = DATASET_PRESETS[args.dataset]
    return {
        "images_dir": Path(args.images_dir or preset["images_dir"]),
        "labels_csv": Path(args.labels_csv or preset["labels_csv"]),
        "img_col": args.img_col or preset["img_col"],
        "mos_col": args.mos_col or preset["mos_col"],
        "label_min": (
            args.label_min if args.label_min is not None else preset["label_min"]
        ),
        "label_max": (
            args.label_max if args.label_max is not None else preset["label_max"]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_PRESETS, default="koniq")
    parser.add_argument("--images-dir")
    parser.add_argument("--labels-csv")
    parser.add_argument("--img-col")
    parser.add_argument("--mos-col")
    parser.add_argument("--label-min", type=float)
    parser.add_argument("--label-max", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--vlm", choices=["server", "local", "mock"], default="server")
    parser.add_argument("--model", default=DEFAULT_OPEN_MODEL)
    parser.add_argument(
        "--model-revision",
        help="Exact Hugging Face revision for local reproducibility",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", "EMPTY"))
    parser.add_argument("--cache-dir")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=DEFAULT_MAX_IMAGE_PIXELS,
        help=(
            "Maximum pixel area sent per image. Default 262144 keeps Qwen2.5-VL "
            "multi-image requests within a 4096-token server context."
        ),
    )
    parser.add_argument(
        "--preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after the first image when the VLM score is unavailable.",
    )
    parser.add_argument("--resume")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _resolve_dataset(args)

    if not config["labels_csv"].exists():
        raise FileNotFoundError(f"Labels CSV not found: {config['labels_csv']}")
    if not config["images_dir"].is_dir():
        raise NotADirectoryError(f"Images directory not found: {config['images_dir']}")
    if config["label_max"] <= config["label_min"]:
        raise ValueError("label-max must be greater than label-min")

    frame = pd.read_csv(config["labels_csv"])
    missing_columns = {
        config["img_col"],
        config["mos_col"],
    } - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")
    if args.limit is not None:
        frame = frame.head(args.limit)

    existing: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            previous = pd.read_csv(resume_path)
            if "image_id" not in previous.columns:
                raise ValueError("Resume CSV lacks image_id")
            completed = (
                previous[previous["status"] == "ok"]
                if "status" in previous.columns
                else previous
            )
            existing = completed.to_dict("records")
            done_ids = {str(value) for value in completed["image_id"].tolist()}

    rows = [
        (str(row[config["img_col"]]), float(row[config["mos_col"]]))
        for _, row in frame.iterrows()
        if str(row[config["img_col"]]) not in done_ids
    ]

    workers = max(1, args.workers)
    if args.vlm == "local" and workers != 1:
        print("Local GPU mode forces --workers 1 to avoid concurrent model calls.")
        workers = 1

    client = VLMClient(
        mode=args.vlm,
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        revision=args.model_revision,
        max_image_pixels=args.max_image_pixels,
    )
    pipeline = IQAPipeline(client, cache_dir=args.cache_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(
        args.output or PROJECT_ROOT / "results" / f"eval_{args.dataset}_{timestamp}.csv"
    )

    print(f"Offline dataset preset: {args.dataset}")
    print(f"Images: {config['images_dir']}")
    print(f"Rows: {len(frame)} ({len(done_ids)} resumed)")
    print(f"Backbone: {args.model} via {args.vlm}")
    print(f"Model revision: {args.model_revision or 'provider/default'}")
    print("Prediction scale: 0-100")
    print(f"Maximum pixels per VLM image: {args.max_image_pixels}")
    print(f"VLM preflight: {'enabled' if args.preflight else 'disabled'}")
    print(
        f"Native label scale: {config['label_min']}-{config['label_max']} "
        "(used only by offline evaluator)"
    )

    records = asyncio.run(
        run_async(
            rows,
            config["images_dir"],
            pipeline,
            workers,
            output_path,
            existing,
            config["label_min"],
            config["label_max"],
            args.preflight,
        )
    )
    _atomic_write_csv(records, output_path)
    metrics = _valid_metrics(records, config["label_min"], config["label_max"])

    print("=" * 48)
    print(f"Valid N    : {metrics['N']}")
    print(f"SRCC       : {metrics['SRCC']}")
    print(f"MAE (0-100): {metrics['MAE_100']}")
    print(f"MAE native : {metrics['MAE_native']}")
    print(f"Saved      : {output_path}")


if __name__ == "__main__":
    main()
