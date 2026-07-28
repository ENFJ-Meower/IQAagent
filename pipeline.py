import os
import tempfile
from pathlib import Path

from skills.prompts import (
    DISTORTION_DETECTION_PROMPT,
    DISTORTION_ANALYSIS_PROMPT,
    SCORING_PROMPT,
)
from skills.vlm_client import VLMClient, extract_json
from tools.perceptual_tools import (
    gradient_sharpness_score,
    noise_level_score,
    noise_residual_map,
    gradient_magnitude_map,
    patch_quality_scores,
)
from tools.iqa_tools import brisque_score, normalize_to_mos_scale
from router.tool_selector import compute_tool_composite


class IQAPipeline:
    def __init__(self, vlm_client: VLMClient, use_memory: bool = False,
                 mode: str = 'zero-shot', cache_dir: str = None):
        self.vlm = vlm_client
        self.use_memory = use_memory
        self.mode = mode
        self.cache_dir = cache_dir
        self._score_cache = {}
        self._maps_dir = None
        if cache_dir:
            scores_path = os.path.join(cache_dir, 'tool_scores.json')
            if os.path.exists(scores_path):
                import json
                with open(scores_path) as f:
                    self._score_cache = json.load(f)
                print(f"[Cache] Loaded {len(self._score_cache)} precomputed entries from {scores_path}")
            self._maps_dir = os.path.join(cache_dir, 'maps')
        self.reranker = None
        if use_memory:
            from memory.reranker import ThurstonReranker
            self.reranker = ThurstonReranker()

    def assess(self, img_path: str) -> dict:
        img_path = str(img_path)
        img_id = os.path.basename(img_path)
        cached = self._score_cache.get(img_id)

        # --- Objective tool scores (from cache or computed) ---
        if cached:
            sharpness = cached['sharpness']
            noise = cached['noise']
            brisque = cached['brisque']
            local_sharp_mos = cached['worst_sharpness_mos']
            local_noise = cached['worst_noise']
        else:
            sharpness = gradient_sharpness_score(img_path)
            noise = noise_level_score(img_path)
            brisque = brisque_score(img_path)
            patch_info = patch_quality_scores(img_path)
            local_sharp_mos = patch_info['worst_sharpness_mos']
            local_noise = patch_info['worst_noise']

        sharpness_mos = normalize_to_mos_scale(sharpness, 'sharpness', self.mode)
        brisque_mos = normalize_to_mos_scale(brisque, 'brisque', self.mode)

        tool_scores = {
            'sharpness': sharpness,
            'sharpness_mos': sharpness_mos,
            'noise': noise,
            'brisque': brisque,
            'brisque_mos': brisque_mos,
            'local_sharp_mos': local_sharp_mos,
            'local_noise': local_noise,
        }

        # --- Call 1: Distortion detection (original image only, short output) ---
        detect_result = extract_json(
            self.vlm.call(DISTORTION_DETECTION_PROMPT, images=[img_path], max_tokens=128)
        )
        distortions = detect_result.get('distortion_set', {}).get('Global', []) if detect_result else []

        # --- Router: distortion-aware composite ---
        tool_composite, sw, bw = compute_tool_composite(sharpness_mos, brisque_mos, distortions)

        # --- Evidence maps ---
        tmp_dir = tempfile.mkdtemp()
        noise_map_path = os.path.join(tmp_dir, 'noise_map.png')
        grad_map_path = os.path.join(tmp_dir, 'grad_map.png')
        noise_residual_map(img_path, noise_map_path)
        gradient_magnitude_map(img_path, grad_map_path)

        # --- Call 2: Distortion analysis (skip if no distortions detected) ---
        distortion_summary = 'No significant distortions detected.'
        distortion_analysis = []
        if distortions:
            analysis_prompt = DISTORTION_ANALYSIS_PROMPT.format(
                distortion_list=', '.join(distortions)
            )
            analysis_result = extract_json(
                self.vlm.call(analysis_prompt, images=[img_path], max_tokens=256)
            )
            if analysis_result:
                distortion_analysis = analysis_result.get('distortion_analysis', [])
                if distortion_analysis:
                    lines = [f"- {d.get('type','?')}: {d.get('severity','?')} — {d.get('explanation','')}"
                             for d in distortion_analysis]
                    distortion_summary = '\n'.join(lines)

        # --- Call 3: Scoring with tool anchor + evidence maps ---
        score_prompt = SCORING_PROMPT.format(
            sharpness_mos=sharpness_mos,
            noise=noise,
            brisque_mos=brisque_mos,
            tool_composite=tool_composite,
            local_sharp_mos=local_sharp_mos,
            local_noise=local_noise,
            distortion_summary=distortion_summary,
        )
        score_result = extract_json(
            self.vlm.call(score_prompt, images=[img_path, noise_map_path, grad_map_path], max_tokens=256)
        )

        quality_score = float(score_result.get('quality_score', tool_composite) if score_result else tool_composite)
        quality_score = max(1.0, min(5.0, quality_score))
        reasoning = score_result.get('reasoning', '') if score_result else ''
        primary_distortion = score_result.get('primary_distortion', '') if score_result else ''

        try:
            os.remove(noise_map_path)
            os.remove(grad_map_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass

        # --- Memory reranking (optional) ---
        if self.use_memory and self.reranker is not None:
            quality_score = self.reranker.rerank(
                query_img_path=img_path,
                initial_score=quality_score,
                reasoning_text=reasoning,
                vlm_client=self.vlm,
            )
            quality_score = max(1.0, min(5.0, quality_score))

        return {
            'score': quality_score,
            'distortions': distortions,
            'distortion_analysis': distortion_analysis,
            'primary_distortion': primary_distortion,
            'reasoning': reasoning,
            'tool_scores': tool_scores,
        }

    async def assess_async(self, img_path: str, slot: int = 0) -> dict:
        """Async version of assess() — uses call_async for true concurrent API calls.
        CPU-bound map generation is offloaded to executor to avoid blocking the event loop.
        """
        import asyncio
        loop = asyncio.get_event_loop()

        img_path = str(img_path)
        img_id = os.path.basename(img_path)
        cached = self._score_cache.get(img_id)

        if cached:
            sharpness = cached['sharpness']
            noise = cached['noise']
            brisque = cached['brisque']
            local_sharp_mos = cached['worst_sharpness_mos']
            local_noise = cached['worst_noise']
        else:
            sharpness = await loop.run_in_executor(None, gradient_sharpness_score, img_path)
            noise = await loop.run_in_executor(None, noise_level_score, img_path)
            brisque = await loop.run_in_executor(None, brisque_score, img_path)
            patch_info = await loop.run_in_executor(None, patch_quality_scores, img_path)
            local_sharp_mos = patch_info['worst_sharpness_mos']
            local_noise = patch_info['worst_noise']

        sharpness_mos = normalize_to_mos_scale(sharpness, 'sharpness', self.mode)
        brisque_mos = normalize_to_mos_scale(brisque, 'brisque', self.mode)

        tool_scores = {
            'sharpness': sharpness, 'sharpness_mos': sharpness_mos,
            'noise': noise, 'brisque': brisque, 'brisque_mos': brisque_mos,
            'local_sharp_mos': local_sharp_mos, 'local_noise': local_noise,
        }

        # Call 1: detection
        print(f"  [slot-{slot}] detect  {img_id}")
        detect_result = extract_json(
            await self.vlm.call_async(DISTORTION_DETECTION_PROMPT, images=[img_path], max_tokens=128)
        )
        distortions = detect_result.get('distortion_set', {}).get('Global', []) if detect_result else []

        tool_composite, sw, bw = compute_tool_composite(sharpness_mos, brisque_mos, distortions)

        tmp_dir = tempfile.mkdtemp()
        noise_map_path = os.path.join(tmp_dir, 'noise_map.png')
        grad_map_path = os.path.join(tmp_dir, 'grad_map.png')
        await loop.run_in_executor(None, noise_residual_map, img_path, noise_map_path)
        await loop.run_in_executor(None, gradient_magnitude_map, img_path, grad_map_path)

        # Call 2: analysis
        distortion_summary = 'No significant distortions detected.'
        distortion_analysis = []
        if distortions:
            print(f"  [slot-{slot}] analyze {img_id}  distortions={distortions}")
            analysis_prompt = DISTORTION_ANALYSIS_PROMPT.format(
                distortion_list=', '.join(distortions)
            )
            analysis_result = extract_json(
                await self.vlm.call_async(analysis_prompt, images=[img_path], max_tokens=256)
            )
            if analysis_result:
                distortion_analysis = analysis_result.get('distortion_analysis', [])
                if distortion_analysis:
                    lines = [f"- {d.get('type','?')}: {d.get('severity','?')} — {d.get('explanation','')}"
                             for d in distortion_analysis]
                    distortion_summary = '\n'.join(lines)

        # Call 3: scoring
        print(f"  [slot-{slot}] score   {img_id}  composite={tool_composite:.2f}")
        score_prompt = SCORING_PROMPT.format(
            sharpness_mos=sharpness_mos, noise=noise, brisque_mos=brisque_mos,
            tool_composite=tool_composite, local_sharp_mos=local_sharp_mos,
            local_noise=local_noise, distortion_summary=distortion_summary,
        )
        score_result = extract_json(
            await self.vlm.call_async(score_prompt,
                                      images=[img_path, noise_map_path, grad_map_path],
                                      max_tokens=256)
        )

        quality_score = float(score_result.get('quality_score', tool_composite) if score_result else tool_composite)
        quality_score = max(1.0, min(5.0, quality_score))
        reasoning = score_result.get('reasoning', '') if score_result else ''
        primary_distortion = score_result.get('primary_distortion', '') if score_result else ''

        try:
            os.remove(noise_map_path)
            os.remove(grad_map_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass

        print(f"  [slot-{slot}] DONE    {img_id}  score={quality_score:.2f}")
        return {
            'score': quality_score,
            'distortions': distortions,
            'distortion_analysis': distortion_analysis,
            'primary_distortion': primary_distortion,
            'reasoning': reasoning,
            'tool_scores': tool_scores,
        }
