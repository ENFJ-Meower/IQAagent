"""
Precompute tool scores and evidence maps for a dataset.
Results cached to cache/<dataset>/ — reusable across all Skill/Prompt changes.
"""
import os
import sys
import json
import argparse
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tools.perceptual_tools import (
    gradient_sharpness_score,
    noise_level_score,
    noise_residual_map,
    gradient_magnitude_map,
    patch_quality_scores,
)
from tools.iqa_tools import brisque_score

DATASET_CONFIG = {
    'koniq': {
        'images_dir': 'E:/Agent/data/koniq10k/koniq-10k/512x384',
        'labels_csv': 'E:/Agent/data/koniq10k/koniq10k_val.csv',
        'img_col': 'img_id',
    },
    'koniq_train': {
        'images_dir': 'E:/Agent/data/koniq10k/koniq-10k/512x384',
        'labels_csv': 'E:/Agent/data/koniq10k/koniq10k_train.csv',
        'img_col': 'img_id',
    },
    'spaq': {
        'images_dir': 'E:/Agent/data/SPAQ/SPAQ-master/SPAQ-master/images',
        'labels_csv': 'E:/Agent/data/SPAQ/spaqTest.csv',
        'img_col': 'image_id',
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='koniq', choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--cache-dir', default='E:/Agent/cache')
    args = parser.parse_args()

    import pandas as pd
    cfg = DATASET_CONFIG[args.dataset]
    df = pd.read_csv(cfg['labels_csv'])
    if args.limit:
        df = df.head(args.limit)

    cache_dir = os.path.join(args.cache_dir, args.dataset)
    maps_dir = os.path.join(cache_dir, 'maps')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    scores_path = os.path.join(cache_dir, 'tool_scores.json')
    existing = {}
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            existing = json.load(f)

    total = len(df)
    done = 0
    skipped = 0

    for i, row in enumerate(df.itertuples(index=False)):
        img_id = getattr(row, cfg['img_col'])
        img_path = os.path.join(cfg['images_dir'], img_id)

        if not os.path.exists(img_path):
            print(f"[{i+1}/{total}] SKIP (not found): {img_id}")
            continue

        # Check if already cached
        noise_map_path = os.path.join(maps_dir, img_id + '.noise.png')
        grad_map_path  = os.path.join(maps_dir, img_id + '.grad.png')
        already_cached = (
            img_id in existing and
            os.path.exists(noise_map_path) and
            os.path.exists(grad_map_path)
        )
        if already_cached:
            skipped += 1
            print(f"[{i+1}/{total}] CACHED: {img_id}")
            continue

        # Compute tool scores
        sharpness = gradient_sharpness_score(img_path)
        noise = noise_level_score(img_path)
        brisque = brisque_score(img_path)
        patch_info = patch_quality_scores(img_path)

        existing[img_id] = {
            'sharpness': sharpness,
            'noise': noise,
            'brisque': brisque,
            'worst_sharpness': patch_info['worst_sharpness'],
            'worst_sharpness_mos': patch_info['worst_sharpness_mos'],
            'worst_noise': patch_info['worst_noise'],
        }

        # Generate evidence maps
        noise_residual_map(img_path, noise_map_path)
        gradient_magnitude_map(img_path, grad_map_path)

        done += 1
        print(f"[{i+1}/{total}] OK: {img_id}  sharp={sharpness:.1f}  brisque={brisque:.2f}  noise={noise:.3f}")

        # Save scores every 50 images
        if done % 50 == 0:
            with open(scores_path, 'w') as f:
                json.dump(existing, f)

    # Final save
    with open(scores_path, 'w') as f:
        json.dump(existing, f)

    print(f"\n{'='*40}")
    print(f"Dataset : {args.dataset} ({total} images)")
    print(f"Computed: {done}  Skipped(cached): {skipped}")
    print(f"Cache   : {cache_dir}")
    print(f"{'='*40}")


if __name__ == '__main__':
    main()
