import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import IQAPipeline
from skills.vlm_client import VLMClient

API_KEY = os.environ.get('VLM_API_KEY', '')
BASE_URL = os.environ.get('VLM_BASE_URL', 'https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1')

DATASET_CONFIG = {
    'koniq': {
        'images_dir': 'E:/Agent/data/koniq10k/koniq-10k/512x384',
        'labels_csv': 'E:/Agent/data/koniq10k/koniq10k_val.csv',
        'img_col': 'img_id',
        'mos_col': 'img_mos',
    },
    'spaq': {
        'images_dir': 'E:/Agent/data/SPAQ/SPAQ-master/SPAQ-master/images',
        'labels_csv': 'E:/Agent/data/SPAQ/spaqTest.csv',
        'img_col': 'image_id',
        'mos_col': 'MOS',
    },
    'koniq_train': {
        'images_dir': 'E:/Agent/data/koniq10k/koniq-10k/512x384',
        'labels_csv': 'E:/Agent/data/koniq10k/koniq10k_train.csv',
        'img_col': 'img_id',
        'mos_col': 'img_mos',
    },
}


def evaluate(pred_scores, mos_scores):
    srcc, _ = spearmanr(pred_scores, mos_scores)
    mae = np.mean(np.abs(np.array(pred_scores) - np.array(mos_scores)))
    return {'SRCC': float(srcc), 'MAE': float(mae)}


# Global offset (kept for --calibrate flag, but §4.5 non-compliant)
GLOBAL_OFFSET = 0.645


def calibrate_score_monotonic(raw_score: float, offset: float = GLOBAL_OFFSET) -> float:
    shifted = raw_score + offset
    if shifted > 5.0:
        return 5.0 + 0.01 * (shifted - 5.0)
    elif shifted < 1.0:
        return 1.0 + 0.01 * (shifted - 1.0)
    return shifted


async def run_async(args, cfg, df, pipeline, out_path, done_ids, existing_results):
    """Main async evaluation loop with --workers true concurrent slots."""
    img_col = cfg['img_col']
    mos_col = cfg['mos_col']
    images_dir = cfg['images_dir']
    total = len(df)

    rows = [(i, getattr(row, img_col), float(getattr(row, mos_col)))
            for i, row in enumerate(df.itertuples(index=False))
            if getattr(row, img_col) not in done_ids]

    if not rows:
        print("All images already processed.")
        return existing_results

    semaphore = asyncio.Semaphore(args.workers)
    results = list(existing_results)
    lock = asyncio.Lock()
    completed = [0]
    t_start = time.time()

    async def process(i, img_name, mos, slot):
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            return None

        async with semaphore:
            try:
                result = await pipeline.assess_async(img_path, slot=slot)
                pred = result['score']
            except Exception as e:
                print(f"  [slot-{slot}] ERROR {img_name}: {e}")
                pred = 3.0
                result = {'score': pred, 'distortions': [], 'reasoning': '', 'tool_scores': {}}

        rec = {
            'image_id': img_name,
            'predicted_score': pred,
            'mos_score': mos,
            'distortions': str(result.get('distortions', [])),
            'reasoning': result.get('reasoning', ''),
            'brisque_mos': result.get('tool_scores', {}).get('brisque_mos', -1),
        }

        async with lock:
            results.append(rec)
            completed[0] += 1
            n = completed[0]
            done_total = len(done_ids) + n
            if n % 5 == 0 or n == 1:
                elapsed = time.time() - t_start
                rate = n / elapsed if elapsed > 0 else 0
                preds = [r['predicted_score'] for r in results]
                mos_list = [r['mos_score'] for r in results]
                if len(preds) > 1:
                    m = evaluate(preds, mos_list)
                    eta = (total - done_total) / rate if rate > 0 else 0
                    print(f"[{done_total}/{total}] SRCC={m['SRCC']:.4f} MAE={m['MAE']:.4f} "
                          f"({rate:.2f}img/s, ETA {eta/60:.0f}min)")
            if completed[0] % 50 == 0:
                pd.DataFrame(results).to_csv(out_path, index=False)

        return rec

    tasks = [process(i, img_name, mos, i % args.workers) for i, img_name, mos in rows]
    await asyncio.gather(*tasks)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='koniq', choices=['koniq', 'spaq', 'koniq_train'])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--vlm', default='api', choices=['api', 'mock'])
    parser.add_argument('--model', default='qwen3.8-max-preview')
    parser.add_argument('--mode', default='zero-shot', choices=['zero-shot', 'train-augmented'])
    parser.add_argument('--cache-dir', default=None,
                        help='Path to precomputed cache dir (e.g. E:/Agent/cache/koniq)')
    parser.add_argument('--use-memory', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--output', default=None)
    parser.add_argument('--workers', type=int, default=3,
                        help='True async concurrent slots (default 3)')
    parser.add_argument('--resume', default=None, help='Path to previous partial CSV to resume from')
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    df = pd.read_csv(cfg['labels_csv'])

    if args.limit:
        df = df.head(args.limit)

    done_ids = set()
    existing_results = []
    if args.resume and os.path.exists(args.resume):
        prev = pd.read_csv(args.resume)
        done_ids = set(prev['image_id'].tolist())
        existing_results = prev.to_dict('records')
        print(f"Resuming: {len(done_ids)} done, {len(df) - len(done_ids)} remaining")

    if args.vlm == 'api':
        vlm = VLMClient(mode='api', model_name=args.model, api_key=API_KEY, base_url=BASE_URL)
    else:
        vlm = VLMClient(mode='mock')

    # Single shared pipeline — AsyncOpenAI client is coroutine-safe
    pipeline = IQAPipeline(vlm_client=vlm, use_memory=args.use_memory,
                           mode=args.mode, cache_dir=args.cache_dir)

    os.makedirs('E:/Agent/results', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = args.output or f'E:/Agent/results/eval_{args.dataset}_{ts}.csv'

    print(f"Dataset : {args.dataset} ({len(df)} images, {len(done_ids)} already done)")
    print(f"Model   : {args.model}  Mode: {args.mode}  Workers: {args.workers}")
    print(f"Output  : {out_path}")
    print("-" * 50)

    results = asyncio.run(
        run_async(args, cfg, df, pipeline, out_path, done_ids, existing_results)
    )

    if not results:
        print("No results collected.")
        return

    out_df = pd.DataFrame(results)

    if args.calibrate:
        out_df['predicted_score'] = out_df['predicted_score'].apply(calibrate_score_monotonic)
        print(f"[Calibration applied: monotonic shift +{GLOBAL_OFFSET}]")

    preds = out_df['predicted_score'].tolist()
    mos_list = out_df['mos_score'].tolist()
    metrics = evaluate(preds, mos_list)

    print(f"\n{'='*40}")
    print(f"Dataset : {args.dataset} ({len(out_df)} images)")
    print(f"Model   : {args.model}")
    print(f"Mode    : {args.mode}")
    print(f"SRCC    : {metrics['SRCC']:.4f}")
    print(f"MAE     : {metrics['MAE']:.4f}")
    print(f"{'='*40}")

    out_df.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")


if __name__ == '__main__':
    main()
