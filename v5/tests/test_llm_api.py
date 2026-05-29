import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.llm_api import call_openai_compatible_chat_with_usage


class _StubResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class LlmApiTests(unittest.TestCase):
    def test_openai_content_part_list_is_joined_and_usage_normalized(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "안녕"},
                            {"type": "text", "text": "하세요"},
                            {"type": "image", "ignored": True},
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }

        with patch("urllib.request.urlopen", return_value=_StubResponse(payload)):
            text, usage = call_openai_compatible_chat_with_usage(
                base_url="http://localhost:8000/v1",
                model_name="test-model",
                prompt="테스트",
                temperature=0.0,
                max_tokens=16,
                timeout_sec=1,
            )

        self.assertEqual(text, "안녕하세요")
        self.assertEqual(usage, {"input_tokens": 3, "output_tokens": 5})

    def test_openai_refusal_fallback_is_returned_when_content_is_empty(self) -> None:
        payload = {
            "choices": [{"message": {"content": "", "refusal": "응답할 수 없습니다."}}],
            "usage": {"input_tokens": 2, "output_tokens": 4},
        }

        with patch("urllib.request.urlopen", return_value=_StubResponse(payload)):
            text, usage = call_openai_compatible_chat_with_usage(
                base_url="http://localhost:8000/v1",
                model_name="test-model",
                prompt="테스트",
                temperature=0.0,
                max_tokens=16,
                timeout_sec=1,
            )

        self.assertEqual(text, "응답할 수 없습니다.")
        self.assertEqual(usage, {"input_tokens": 2, "output_tokens": 4})


if __name__ == "__main__":
    unittest.main()
