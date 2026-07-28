import asyncio
import base64
import json
import re
import time
from pathlib import Path
from openai import OpenAI, AsyncOpenAI


def _encode_image(img_path: str) -> str:
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _image_content(img_path: str) -> dict:
    ext = Path(img_path).suffix.lower()
    mime = 'image/png' if ext == '.png' else 'image/jpeg'
    b64 = _encode_image(img_path)
    return {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}}


class VLMClient:
    def __init__(self, mode: str = 'api', model_name: str = 'qwen-vl-max',
                 api_key: str = None, base_url: str = None):
        self.mode = mode
        self.model_name = model_name

        if mode == 'api':
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        elif mode == 'mock':
            self.client = None
            self.async_client = None
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'api' or 'mock'.")

    def call(self, prompt: str, images: list = None, max_tokens: int = 512,
             retries: int = 3) -> str:
        if self.mode == 'mock':
            return self._mock_response(prompt)

        content = []
        if images:
            for img_path in images:
                content.append(_image_content(img_path))
        content.append({'type': 'text', 'text': prompt})

        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': content}],
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"[VLMClient] API call failed after {retries} attempts: {e}")
                    return ''
        return ''

    async def call_async(self, prompt: str, images: list = None, max_tokens: int = 512,
                         retries: int = 3) -> str:
        if self.mode == 'mock':
            return self._mock_response(prompt)

        content = []
        if images:
            for img_path in images:
                content.append(_image_content(img_path))
        content.append({'type': 'text', 'text': prompt})

        for attempt in range(retries):
            try:
                resp = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': content}],
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"[VLMClient] async call failed after {retries} attempts: {e}")
                    return ''
        return ''

    def _mock_response(self, prompt: str) -> str:
        if 'distortion_set' in prompt or 'distortion types' in prompt.lower():
            return '{"distortion_set": {"Global": ["Blurs", "Noise"]}}'
        elif 'distortion_analysis' in prompt or 'severity' in prompt.lower():
            return '{"distortion_analysis": [{"type": "Blurs", "severity": "slight", "explanation": "Minor blurring visible in edges."}, {"type": "Noise", "severity": "moderate", "explanation": "Visible grain in uniform areas."}]}'
        elif 'quality_score' in prompt or 'quality scale' in prompt.lower():
            return '{"quality_score": 3.5, "primary_distortion": "Noise", "reasoning": "The image shows moderate noise with slight blurring. Overall acceptable quality for casual use."}'
        elif 'Image A' in prompt or 'Image B' in prompt:
            return 'B'
        else:
            return '{"quality_score": 3.0, "primary_distortion": "none", "reasoning": "Mock response."}'


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}
