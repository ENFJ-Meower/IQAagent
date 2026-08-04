"""VLM adapters for an open-weight Qwen2.5-VL backbone.

Image file paths are never placed in prompts. Server mode sends base64 data
URLs; local mode passes in-memory PIL images to the processor.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_OPEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MAX_IMAGE_PIXELS = 512 * 512


@dataclass(frozen=True)
class VLMCallResult:
    """Text plus provider diagnostics for one model request."""

    text: str = ""
    error: str = ""
    finish_reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def diagnostic(self, preview_chars: int = 500) -> dict[str, Any]:
        result = asdict(self)
        result["text_preview"] = result.pop("text")[:preview_chars]
        return result


def _bounded_image_bytes(
    img_path: str,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> tuple[bytes, str]:
    """Encode an image after bounding its pixel area.

    Qwen2.5-VL can otherwise allocate up to 16,384 visual tokens to one image.
    Bounding the actual payload keeps multi-image requests inside short server
    contexts such as 4,096 tokens without exposing a file identity.
    """

    path = Path(img_path)
    suffix = path.suffix.lower()
    original_mime = "image/png" if suffix == ".png" else "image/jpeg"
    with Image.open(path) as source:
        width, height = source.size
        if width * height <= max_pixels:
            with open(path, "rb") as stream:
                return stream.read(), original_mime

        scale = math.sqrt(max_pixels / float(width * height))
        resized_width = max(28, int(width * scale) // 28 * 28)
        resized_height = max(28, int(height * scale) // 28 * 28)
        resized = source.convert("RGB").resize(
            (resized_width, resized_height),
            Image.Resampling.LANCZOS,
        )
        output = io.BytesIO()
        if suffix == ".png":
            resized.save(output, format="PNG", optimize=True)
            mime = "image/png"
        else:
            resized.save(output, format="JPEG", quality=92, optimize=True)
            mime = "image/jpeg"
        return output.getvalue(), mime


def _image_content(
    img_path: str,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> dict[str, Any]:
    payload, mime = _bounded_image_bytes(img_path, max_pixels)
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"
        },
    }


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    reasoning = getattr(message, "reasoning_content", "")
    return reasoning if isinstance(reasoning, str) else ""


def _usage_value(response: Any, name: str) -> int | None:
    usage = getattr(response, "usage", None)
    value = getattr(usage, name, None) if usage is not None else None
    return int(value) if isinstance(value, int) else None


class VLMClient:
    def __init__(
        self,
        mode: str = "server",
        model_name: str = DEFAULT_OPEN_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        revision: str | None = None,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    ):
        if mode == "api":
            mode = "server"
        if mode not in {"server", "local", "mock"}:
            raise ValueError("mode must be 'server', 'local', or 'mock'")

        self.mode = mode
        self.model_name = model_name
        self.last_error = ""
        self.client = None
        self.async_client = None
        self.model = None
        self.processor = None
        if max_image_pixels < 28 * 28:
            raise ValueError("max_image_pixels must be at least 784")
        self.max_image_pixels = int(max_image_pixels)

        if mode == "server":
            from openai import AsyncOpenAI, OpenAI

            resolved_key = api_key or "EMPTY"
            resolved_url = base_url or "http://127.0.0.1:8000/v1"
            self.client = OpenAI(api_key=resolved_key, base_url=resolved_url)
            self.async_client = AsyncOpenAI(api_key=resolved_key, base_url=resolved_url)
        elif mode == "local":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                model_name,
                revision=revision,
                max_pixels=self.max_image_pixels,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                dtype="auto",
                device_map="auto",
                revision=revision,
            )

    def call(
        self,
        prompt: str,
        images: list[str] | None = None,
        max_tokens: int = 512,
        retries: int = 3,
    ) -> str:
        return self.call_result(prompt, images, max_tokens, retries).text

    def call_result(
        self,
        prompt: str,
        images: list[str] | None = None,
        max_tokens: int = 512,
        retries: int = 3,
    ) -> VLMCallResult:
        if self.mode == "mock":
            return VLMCallResult(text=self._mock_response(prompt), finish_reason="stop")
        if self.mode == "local":
            return self._call_local_result(prompt, images or [], max_tokens)

        content: list[dict[str, Any]] = []
        for img_path in images or []:
            content.append(_image_content(img_path, self.max_image_pixels))
        content.append({"type": "text", "text": prompt})

        last_error = ""
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                self.last_error = ""
                choice = response.choices[0]
                return VLMCallResult(
                    text=_message_text(choice.message),
                    finish_reason=str(choice.finish_reason or ""),
                    prompt_tokens=_usage_value(response, "prompt_tokens"),
                    completion_tokens=_usage_value(response, "completion_tokens"),
                )
            except Exception as exc:  # noqa: BLE001 - provider-specific errors vary
                last_error = str(exc)
                self.last_error = last_error
                if attempt < retries - 1:
                    time.sleep(2**attempt)
        return VLMCallResult(error=last_error)

    async def call_async(
        self,
        prompt: str,
        images: list[str] | None = None,
        max_tokens: int = 512,
        retries: int = 3,
    ) -> str:
        return (await self.call_async_result(prompt, images, max_tokens, retries)).text

    async def call_async_result(
        self,
        prompt: str,
        images: list[str] | None = None,
        max_tokens: int = 512,
        retries: int = 3,
    ) -> VLMCallResult:
        if self.mode == "mock":
            return VLMCallResult(text=self._mock_response(prompt), finish_reason="stop")
        if self.mode == "local":
            return await asyncio.to_thread(
                self._call_local_result, prompt, images or [], max_tokens
            )

        content: list[dict[str, Any]] = []
        for img_path in images or []:
            content.append(_image_content(img_path, self.max_image_pixels))
        content.append({"type": "text", "text": prompt})

        last_error = ""
        for attempt in range(retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                self.last_error = ""
                choice = response.choices[0]
                return VLMCallResult(
                    text=_message_text(choice.message),
                    finish_reason=str(choice.finish_reason or ""),
                    prompt_tokens=_usage_value(response, "prompt_tokens"),
                    completion_tokens=_usage_value(response, "completion_tokens"),
                )
            except Exception as exc:  # noqa: BLE001 - provider-specific errors vary
                last_error = str(exc)
                self.last_error = last_error
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
        return VLMCallResult(error=last_error)

    def _call_local(self, prompt: str, images: list[str], max_tokens: int) -> str:
        return self._call_local_result(prompt, images, max_tokens).text

    def _call_local_result(
        self,
        prompt: str,
        images: list[str],
        max_tokens: int,
    ) -> VLMCallResult:
        import torch

        pil_images = []
        try:
            content: list[dict[str, Any]] = []
            for img_path in images:
                with Image.open(img_path) as source:
                    image = source.convert("RGB").copy()
                pil_images.append(image)
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            trimmed = [
                output[len(input_ids) :]
                for input_ids, output in zip(inputs.input_ids, generated)
            ]
            result = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            self.last_error = ""
            return VLMCallResult(
                text=result[0] if result else "",
                finish_reason="stop",
                prompt_tokens=int(inputs.input_ids.shape[-1]),
                completion_tokens=int(trimmed[0].shape[-1]) if trimmed else 0,
            )
        except Exception as exc:  # noqa: BLE001 - optional local stack varies
            self.last_error = str(exc)
            return VLMCallResult(error=self.last_error)
        finally:
            for image in pil_images:
                image.close()

    @staticmethod
    def _mock_response(prompt: str) -> str:
        if "Report only confidently visible issues" in prompt:
            return (
                '{"issues":['
                '{"type":"Blurs","severity":"slight","evidence":"Soft fine edges."},'
                '{"type":"Noise","severity":"moderate","evidence":"Visible residual grain."}'
                "]}"
            )
        if '"dimension_scores"' in prompt:
            return (
                '{"quality_score":61.7,'
                '"dimension_scores":{"sharpness":64,"noise_cleanliness":52,'
                '"exposure":72,"color_fidelity":68,"artifact_free":57},'
                '"primary_distortion":"Noise",'
                '"reasoning":"Fine detail is usable but residual grain is visible."}'
            )
        return "{}"


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a model response."""

    if not isinstance(text, str):
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}
