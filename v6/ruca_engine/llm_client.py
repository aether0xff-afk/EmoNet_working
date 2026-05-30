from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any
import urllib.error
import urllib.request


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-5.4"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 900
    timeout_sec: int = 45
    reasoning_effort: str | None = None

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            value = os.environ.get(self.api_key_env, "").strip()
            if value:
                return value
        return None

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["api_key"] = "***" if self.resolved_api_key() else None
        return payload


@dataclass(frozen=True)
class LLMResponse:
    text: str
    raw_text: str
    usage: dict[str, int]

    def to_record(self) -> dict[str, Any]:
        return {"text": self.text, "raw_text": self.raw_text, "usage": dict(self.usage)}


def generate_llm_response(prompt: str, config: LLMConfig) -> LLMResponse:
    provider = str(config.provider or "openai_compatible").strip().lower()
    api_key = config.resolved_api_key()
    if provider == "anthropic":
        text, usage = _call_anthropic_messages(prompt=prompt, config=config, api_key=api_key)
    elif provider == "openai_compatible":
        text, usage = _call_openai_compatible_chat(prompt=prompt, config=config, api_key=api_key)
    else:
        raise ValueError(f"unsupported LLM provider: {config.provider}")
    cleaned = _validate_ruca_text(text)
    return LLMResponse(text=cleaned, raw_text=text, usage=usage)


def _call_openai_compatible_chat(prompt: str, config: LLMConfig, api_key: str | None) -> tuple[str, dict[str, int]]:
    url = config.base_url.rstrip("/") + "/chat/completions"
    is_openai_api = bool(api_key and "api.openai.com" in config.base_url.lower())
    payload: dict[str, Any] = {
        "model": config.model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only Ruca's final Korean user-facing reply. "
                    "Do not expose labels, JSON, traces, system analysis, or thinking. "
                    "Use warm casual Korean unless the user clearly asks otherwise."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    if is_openai_api:
        payload["max_completion_tokens"] = int(config.max_tokens)
    else:
        payload["max_tokens"] = int(config.max_tokens)
        payload["temperature"] = float(config.temperature)
    if is_openai_api and config.temperature not in (0, 0.0, 1, 1.0):
        payload["temperature"] = float(config.temperature)
    if config.reasoning_effort:
        payload["reasoning_effort"] = config.reasoning_effort

    response_payload = _post_json(url=url, payload=payload, headers=_auth_headers(api_key), timeout_sec=config.timeout_sec)
    choices = response_payload.get("choices") or []
    if not choices:
        raise ValueError("OpenAI-compatible chat response missing choices")
    message = choices[0].get("message") or {}
    content = _extract_openai_content(message)
    if not content:
        raise ValueError("OpenAI-compatible chat response did not contain text")
    return content, _normalize_usage(response_payload.get("usage") or {})


def _call_anthropic_messages(prompt: str, config: LLMConfig, api_key: str | None) -> tuple[str, dict[str, int]]:
    if not api_key:
        raise ValueError("Anthropic provider requires an API key")
    url = config.base_url.rstrip("/") + "/v1/messages"
    payload: dict[str, Any] = {
        "model": config.model_name,
        "max_tokens": int(config.max_tokens),
        "temperature": float(config.temperature),
        "system": (
            "Return only Ruca's final Korean user-facing reply. "
            "Do not expose labels, JSON, traces, system analysis, or thinking. "
            "Use warm casual Korean unless the user clearly asks otherwise."
        ),
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    response_payload = _post_json(url=url, payload=payload, headers=headers, timeout_sec=config.timeout_sec)
    chunks = [
        item["text"]
        for item in response_payload.get("content") or []
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
    ]
    text = "".join(chunks).strip()
    if not text:
        raise ValueError("Anthropic response did not contain text")
    return text, _normalize_usage(response_payload.get("usage") or {})


def _post_json(*, url: str, payload: dict[str, Any], headers: dict[str, str], timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=int(timeout_sec)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed ({exc.code}): {body[:800]}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"could not reach LLM endpoint: {exc}") from exc


def _auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_openai_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        chunks = [
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
        ]
        return "".join(chunks).strip()
    if isinstance(content, str) and content.strip():
        return content.strip()
    return ""


def _normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _validate_ruca_text(text: str) -> str:
    cleaned = _strip_thinking(str(text or "")).strip()
    if not cleaned:
        raise ValueError("LLM response was empty")
    forbidden = ("[ROLE]", "[INNER_VOICES]", "{", "}", "voice_id", "source_character")
    if any(token in cleaned for token in forbidden):
        raise ValueError("LLM response exposed internal prompt structure")
    return cleaned


def _strip_thinking(text: str) -> str:
    cleaned = str(text or "")
    while "<think>" in cleaned and "</think>" in cleaned:
        start = cleaned.find("<think>")
        end = cleaned.find("</think>", start)
        if end < 0:
            break
        cleaned = cleaned[:start] + cleaned[end + len("</think>") :]
    return cleaned
