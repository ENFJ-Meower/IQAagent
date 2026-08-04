import tempfile
import unittest
from pathlib import Path

from PIL import Image

from skills.vlm_client import VLMClient, _image_content, extract_json
from tools.cache import image_content_key


class VLMAndCacheTests(unittest.TestCase):
    def test_extract_json_from_fence(self):
        parsed = extract_json('```json\n{"quality_score":61.2}\n```')
        self.assertEqual(parsed["quality_score"], 61.2)

    def test_mock_scoring_has_dimensions(self):
        client = VLMClient(mode="mock")
        parsed = extract_json(client.call('Return "dimension_scores" as JSON'))
        self.assertIn("dimension_scores", parsed)

    def test_server_image_payload_has_no_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forbidden_identity.jpg"
            Image.new("RGB", (32, 32), "white").save(path)
            payload = _image_content(str(path))
            url = payload["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
            self.assertNotIn(path.name, url)

    def test_server_payload_bounds_large_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.png"
            Image.new("RGB", (1200, 900), "white").save(path)
            payload = _image_content(str(path), max_pixels=512 * 512)
            self.assertTrue(payload["image_url"]["url"].startswith("data:image/png"))

    def test_cache_key_depends_on_content_not_name(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jpg"
            second = Path(directory) / "second.jpg"
            first.write_bytes(b"same-content")
            second.write_bytes(b"same-content")
            self.assertEqual(
                image_content_key(str(first)), image_content_key(str(second))
            )


if __name__ == "__main__":
    unittest.main()
