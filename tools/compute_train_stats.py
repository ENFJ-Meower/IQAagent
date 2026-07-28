"""
Compute sharpness and BRISQUE statistics on KonIQ train set.
Used to derive piecewise normalization breakpoints for Mode B.
Compliant with task spec §4.1 (train set is not an evaluation set).
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.perceptual_tools import gradient_sharpness_score, noise_level_score
from tools.iqa_tools import brisque_score

IMAGES_DIR = 'E:/Agent/data/koniq10k/koniq-10k/512x384'
TRAIN_CSV  = 'E:/Agent/data/koniq10k/koniq10k_train.csv'
N_SAMPLE   = 300


def main():
    df = pd.read_csv(TRAIN_CSV)
    df = df.sample(n=min(N_SAMPLE, len(df)), random_state=42).reset_index(drop=True)
    print(f"Sampling {len(df)} images from KonIQ train set...\n")

    records = []
    for i, row in df.iterrows():
        img_path = os.path.join(IMAGES_DIR, row['img_id'])
        if not os.path.exists(img_path):
            continue

        sharp = gradient_sharpness_score(img_path)
        noise = noise_level_score(img_path)
        brisque = brisque_score(img_path)
        mos = float(row['img_mos'])

        records.append({
            'img_id': row['img_id'],
            'sharpness_raw': sharp,
            'noise_raw': noise,
            'brisque_raw': brisque,
            'mos': mos,
        })

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(df)}] {row['img_id']}  sharp={sharp:.1f}  brisque={brisque:.2f}  mos={mos:.3f}")

    stats_df = pd.DataFrame(records)
    out_path = 'E:/Agent/tools/train_stats.csv'
    stats_df.to_csv(out_path, index=False)

    print(f"\n=== Sharpness Raw ===")
    s = stats_df['sharpness_raw']
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p:2d} = {np.percentile(s, p):.2f}")
    print(f"  mean = {s.mean():.2f}   std = {s.std():.2f}")

    print(f"\n=== BRISQUE Raw ===")
    b = stats_df['brisque_raw']
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p:2d} = {np.percentile(b, p):.2f}")
    print(f"  mean = {b.mean():.2f}   std = {b.std():.2f}")

    print(f"\n=== MOS ===")
    m = stats_df['mos']
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p:2d} = {np.percentile(m, p):.3f}")
    print(f"  mean = {m.mean():.3f}   std = {m.std():.3f}")

    # Sharpness vs MOS 分段分析
    print(f"\n=== Sharpness 分段 vs MOS 均值 ===")
    bins = [0, 200, 500, 1000, 1500, 2000, 3000, 99999]
    labels = ['0-200', '200-500', '500-1k', '1k-1.5k', '1.5k-2k', '2k-3k', '3k+']
    stats_df['sharp_bin'] = pd.cut(stats_df['sharpness_raw'], bins=bins, labels=labels)
    g = stats_df.groupby('sharp_bin', observed=True).agg(
        count=('mos', 'count'),
        mean_mos=('mos', 'mean'),
        mean_sharp=('sharpness_raw', 'mean'),
    )
    print(g.to_string())

    print(f"\nStats saved to: {out_path}")


if __name__ == '__main__':
    main()
