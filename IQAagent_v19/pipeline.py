"""Dataset-blind IQA pipeline."""

from __future__ import annotations

import asyncio
import math
import os
import tempfile
from typing import Any

from router.tool_selector import (
    compute_evidence_anchor,
    dimension_score,
    fuse_scores,
    sanitize_distortions,
    select_evidence_types,
)
from skills.prompts import (
    SCORING_PROMPT,
    TECHNICAL_ASSESSMENT_PROMPT,
)
from skills.vlm_client import VLMCallResult, VLMClient, extract_json
from tools.cache import image_content_key, load_cache, map_path
from tools.iqa_tools import build_evidence_indices
from tools.perceptual_tools import (
    compute_raw_evidence,
    detail_contact_sheet,
    gradient_magnitude_map,
    noise_residual_map,
)

_EVIDENCE_DESCRIPTIONS = {
    "detail": "detail contact sheet: full view, four corners, and center crop",
    "gradient": "fixed-scale gradient map: brighter pixels mean stronger edges",
    "noise": "fixed-scale residual map: brighter pixels mean larger local residuals",
}


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        return None
    return number


class IQAPipeline:
    def __init__(self, vlm_client: VLMClient, cache_dir: str | None = None):
        self.vlm = vlm_client
        self.cache_dir = cache_dir
        self._cache = load_cache(cache_dir)

    @staticmethod
    def _guard_online_prompt(prompt: str, img_path: str) -> None:
        """Fail closed if a file identity accidentally enters a prompt."""

        forbidden = {
            str(img_path),
            os.path.basename(str(img_path)),
        }
        for token in forbidden:
            if token and token in prompt:
                raise ValueError("Online prompt contains forbidden file identity")

    def _load_or_compute_tools(
        self, img_path: str
    ) -> tuple[str, dict[str, Any], dict[str, float | None], list[str]]:
        image_key = image_content_key(img_path)
        cached = self._cache.get(image_key)
        warnings: list[str] = []
        if isinstance(cached, dict) and isinstance(cached.get("raw"), dict):
            raw = cached["raw"]
            indices = cached.get("indices") or build_evidence_indices(raw)
        else:
            raw = compute_raw_evidence(img_path)
            indices = build_evidence_indices(raw)

        if raw.get("brisque_raw") is None:
            warnings.append(
                "BRISQUE unavailable; router renormalized the remaining evidence."
            )
        return image_key, raw, indices, warnings

    def _cached_map(self, image_key: str, evidence_type: str) -> str | None:
        if not self.cache_dir:
            return None
        path = map_path(self.cache_dir, image_key, evidence_type)
        return str(path) if path.exists() else None

    def _prepare_visual_evidence(
        self,
        img_path: str,
        image_key: str,
        evidence_types: list[str],
        temp_dir: str,
    ) -> tuple[list[str], str]:
        generators = {
            "detail": detail_contact_sheet,
            "gradient": gradient_magnitude_map,
            "noise": noise_residual_map,
        }
        paths = [img_path]
        descriptions = ["Image 1: original image"]
        for evidence_type in evidence_types:
            cached = self._cached_map(image_key, evidence_type)
            if cached:
                evidence_path = cached
            else:
                evidence_path = os.path.join(temp_dir, f"{evidence_type}.png")
                generators[evidence_type](img_path, evidence_path)
            paths.append(evidence_path)
            descriptions.append(
                f"Image {len(paths)}: {_EVIDENCE_DESCRIPTIONS[evidence_type]}"
            )
        return paths, "\n".join(f"- {item}" for item in descriptions)

    @staticmethod
    def _distortion_summary(analysis: list[dict[str, Any]]) -> str:
        if not analysis:
            return "No confident dominant issue was established."
        lines = []
        for item in analysis:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('type', '?')}: {item.get('severity', '?')} - "
                f"{item.get('evidence', '')}"
            )
        return "\n".join(lines) or "No confident dominant issue was established."

    @staticmethod
    def _build_scoring_prompt(
        indices: dict[str, float | None],
        distortion_summary: str,
        evidence_description: str,
    ) -> str:
        brisque = indices.get("brisque_quality")
        brisque_text = "unavailable" if brisque is None else f"{float(brisque):.2f}/100"
        return SCORING_PROMPT.format(
            global_sharpness=float(indices["global_sharpness"]),
            local_sharpness=float(indices["local_sharpness"]),
            brisque_quality_text=brisque_text,
            noise_severity=float(indices["noise_severity"]),
            exposure_quality=float(indices["exposure_quality"]),
            blockiness_quality=float(indices["blockiness_quality"]),
            distortion_summary=distortion_summary,
            evidence_description=evidence_description,
        )

    @staticmethod
    def _parse_assessment(
        result: dict[str, Any],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        candidate = result.get("issues", [])
        if not isinstance(candidate, list):
            return [], []
        analysis = [item for item in candidate if isinstance(item, dict)]
        distortions = sanitize_distortions(
            [item.get("type") for item in analysis]
        )
        filtered = [
            item
            for item in analysis
            if isinstance(item.get("type"), str) and item["type"] in distortions
        ]
        return distortions, filtered

    @staticmethod
    def _finalize(
        raw: dict[str, Any],
        indices: dict[str, float | None],
        warnings: list[str],
        distortions: list[str],
        analysis: list[dict[str, Any]],
        score_result: dict[str, Any],
        anchor: float,
        evidence_weights: dict[str, float],
        profile: str,
        evidence_types: list[str],
        assessment_call: VLMCallResult,
        scoring_call: VLMCallResult,
    ) -> dict[str, Any]:
        direct_score = _safe_float(score_result.get("quality_score"))
        dimensions = score_result.get("dimension_scores")
        dimensions = dimensions if isinstance(dimensions, dict) else {}
        aggregate_dimensions, dimension_weights = dimension_score(dimensions)
        final_score, fusion_weights = fuse_scores(
            anchor,
            direct_score,
            aggregate_dimensions,
        )
        vlm_valid = direct_score is not None or aggregate_dimensions is not None
        if not vlm_valid:
            warnings.append("VLM score unavailable; used deterministic evidence only.")
        if assessment_call.error:
            warnings.append(f"Assessment API error: {assessment_call.error}")
        if scoring_call.error:
            warnings.append(f"Scoring API error: {scoring_call.error}")
        if assessment_call.finish_reason == "length":
            warnings.append("Assessment output was truncated at the token limit.")
        if scoring_call.finish_reason == "length":
            warnings.append("Scoring output was truncated at the token limit.")

        return {
            "score": final_score,
            "score_scale": "0-100",
            "vlm_valid": vlm_valid,
            "vlm_direct_score": direct_score,
            "vlm_dimension_score": aggregate_dimensions,
            "vlm_dimension_scores": dimensions,
            "vlm_dimension_weights": dimension_weights,
            "distortions": distortions,
            "distortion_analysis": analysis,
            "primary_distortion": score_result.get("primary_distortion", "none"),
            "reasoning": score_result.get("reasoning", ""),
            "raw_tools": raw,
            "evidence_indices": indices,
            "router": {
                "rule_profile": profile,
                "selected_evidence": evidence_types,
                "evidence_anchor": anchor,
                "evidence_weights": evidence_weights,
                "fusion_weights": fusion_weights,
            },
            "vlm_diagnostics": {
                "assessment": assessment_call.diagnostic(),
                "scoring": scoring_call.diagnostic(),
            },
            "warnings": warnings,
        }

    def assess(self, img_path: str) -> dict[str, Any]:
        img_path = str(img_path)
        image_key, raw, indices, warnings = self._load_or_compute_tools(img_path)

        self._guard_online_prompt(TECHNICAL_ASSESSMENT_PROMPT, img_path)
        assessment_call = self.vlm.call_result(
            TECHNICAL_ASSESSMENT_PROMPT,
            images=[img_path],
            max_tokens=220,
        )
        assessment_result = extract_json(assessment_call.text)
        distortions, analysis = self._parse_assessment(assessment_result)
        if not assessment_result or "issues" not in assessment_result:
            warnings.append("Technical assessment response was invalid.")

        anchor, evidence_weights, profile = compute_evidence_anchor(
            indices, distortions
        )
        evidence_types = select_evidence_types(distortions)

        with tempfile.TemporaryDirectory(prefix="iqa_evidence_") as temp_dir:
            images, descriptions = self._prepare_visual_evidence(
                img_path, image_key, evidence_types, temp_dir
            )
            score_prompt = self._build_scoring_prompt(
                indices,
                self._distortion_summary(analysis),
                descriptions,
            )
            self._guard_online_prompt(score_prompt, img_path)
            scoring_call = self.vlm.call_result(
                score_prompt,
                images=images,
                max_tokens=320,
            )
            score_result = extract_json(scoring_call.text)
        if not score_result or not (
            "quality_score" in score_result or "dimension_scores" in score_result
        ):
            warnings.append("Final scoring response was invalid.")

        return self._finalize(
            raw,
            indices,
            warnings,
            distortions,
            analysis,
            score_result,
            anchor,
            evidence_weights,
            profile,
            evidence_types,
            assessment_call,
            scoring_call,
        )

    async def assess_async(self, img_path: str, slot: int = 0) -> dict[str, Any]:
        img_path = str(img_path)
        image_label = image_content_key(img_path)[:10]
        print(f"  [slot-{slot}] start {image_label}")

        tool_task = asyncio.to_thread(self._load_or_compute_tools, img_path)
        self._guard_online_prompt(TECHNICAL_ASSESSMENT_PROMPT, img_path)
        assessment_task = self.vlm.call_async_result(
            TECHNICAL_ASSESSMENT_PROMPT,
            images=[img_path],
            max_tokens=220,
        )
        (image_key, raw, indices, warnings), assessment_call = await asyncio.gather(
            tool_task, assessment_task
        )

        assessment_result = extract_json(assessment_call.text)
        distortions, analysis = self._parse_assessment(assessment_result)
        if not assessment_result or "issues" not in assessment_result:
            warnings.append("Technical assessment response was invalid.")

        anchor, evidence_weights, profile = compute_evidence_anchor(
            indices, distortions
        )
        evidence_types = select_evidence_types(distortions)

        with tempfile.TemporaryDirectory(prefix="iqa_evidence_") as temp_dir:
            images, descriptions = await asyncio.to_thread(
                self._prepare_visual_evidence,
                img_path,
                image_key,
                evidence_types,
                temp_dir,
            )
            score_prompt = self._build_scoring_prompt(
                indices,
                self._distortion_summary(analysis),
                descriptions,
            )
            self._guard_online_prompt(score_prompt, img_path)
            scoring_call = await self.vlm.call_async_result(
                score_prompt,
                images=images,
                max_tokens=320,
            )
            score_result = extract_json(scoring_call.text)
        if not score_result or not (
            "quality_score" in score_result or "dimension_scores" in score_result
        ):
            warnings.append("Final scoring response was invalid.")

        result = self._finalize(
            raw,
            indices,
            warnings,
            distortions,
            analysis,
            score_result,
            anchor,
            evidence_weights,
            profile,
            evidence_types,
            assessment_call,
            scoring_call,
        )
        print(f"  [slot-{slot}] done  {image_label} score={result['score']:.2f}")
        return result
