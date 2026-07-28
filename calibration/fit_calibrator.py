"""
Offline MAE calibration script.

Fits an isotonic regression (monotone) mapping from VLM predicted scores to MOS scale.
Isotonic regression guarantees rank-preservation (SRCC unchanged) and is robust on small samples.
Saves parameters to calibration/params.json for use in run_eval.py --calibrate.

Usage:
    python calibration/fit_calibrator.py --results results/eval_koniq_v06_50imgs.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate(pred, mos):
    srcc, _ = spearmanr(pred, mos)
    mae = float(np.mean(np.abs(np.array(pred) - np.array(mos))))
    return srcc, mae


def fit_isotonic(pred_scores: list, mos_scores: list) -> dict:
    """Fit isotonic regression and save as a lookup table over [1.0, 5.0]."""
    x = np.array(pred_scores, dtype=np.float64)
    y = np.array(mos_scores, dtype=np.float64)

    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(x, y)

    # Build a fine-grained lookup table so we can save/load without the sklearn object
    grid = np.linspace(1.0, 5.0, 200)
    mapped = ir.predict(grid)
    mapped = np.clip(mapped, 1.0, 5.0)

    return {
        'method': 'isotonic',
        'grid': grid.tolist(),
        'mapped': mapped.tolist(),
    }


def apply_isotonic(score: float, params: dict) -> float:
    """Apply isotonic calibration via linear interpolation on the lookup table."""
    grid = np.array(params['grid'])
    mapped = np.array(params['mapped'])
    return float(np.clip(np.interp(score, grid, mapped), 1.0, 5.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='Path to eval CSV')
    parser.add_argument('--out', default='calibration/params.json', help='Output params path')
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    if 'predicted_score' not in df.columns or 'mos_score' not in df.columns:
        print("ERROR: CSV must have 'predicted_score' and 'mos_score' columns")
        sys.exit(1)

    pred = df['predicted_score'].tolist()
    mos = df['mos_score'].tolist()

    srcc_before, mae_before = evaluate(pred, mos)
    print(f"Before calibration:  SRCC={srcc_before:.4f}  MAE={mae_before:.4f}")

    params = fit_isotonic(pred, mos)

    calibrated = [apply_isotonic(p, params) for p in pred]
    srcc_after, mae_after = evaluate(calibrated, mos)
    print(f"After calibration:   SRCC={srcc_after:.4f}  MAE={mae_after:.4f}")
    print(f"MAE reduction: {(mae_before - mae_after) / mae_before * 100:.1f}%")

    srcc_delta = abs(srcc_after - srcc_before)
    if srcc_delta > 0.002:
        print(f"WARNING: SRCC changed by {srcc_after - srcc_before:+.4f}")
    else:
        print(f"SRCC unchanged (delta={srcc_after - srcc_before:+.5f}) ✓")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(params, f)
    print(f"Parameters saved to: {out_path}")


if __name__ == '__main__':
    main()

