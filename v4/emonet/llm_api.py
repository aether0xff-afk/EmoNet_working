from __future__ import annotations

import json
from typing import Any, Callable
import urllib.error
import urllib.request


def extract_json_block(text: str) -> dict:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    candidate = stripped[start : end + 1]
    return json.loads(candidate)


def call_openai_compatible_chat(
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    system_prompt: str = "Return JSON only.",
    api_key: str | None = None,
    response_format: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    is_openai_api = bool(api_key and "api.openai.com" in str(base_url).lower())
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    if not is_openai_api:
        payload["temperature"] = temperature
    elif temperature not in (0, 0.0, 1, 1.0):
        payload["temperature"] = temperature
    if is_openai_api:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    if response_format is not None:
        payload["response_format"] = response_format
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible chat failed ({exc.code}): {body[:800]}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"could not reach OpenAI-compatible chat endpoint: {exc}") from exc

    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("OpenAI-compatible chat response missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text_chunks: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text_chunks.append(item["text"])
        content = "".join(text_chunks)
    if isinstance(content, str) and content.strip():
        return content.strip()

    for field_name in ("reasoning", "reasoning_content", "refusal"):
        fallback = message.get(field_name)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
    raise ValueError("chat response did not contain text content")


def request_json_response(
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
    validator: Callable[[dict], object] | None = None,
    retry_instruction: str | None = None,
    api_key: str | None = None,
    response_format: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
) -> tuple[object, str]:
    last_raw = ""
    last_error = ""
    for attempt in range(max_retries + 1):
        retry_suffix = ""
        if attempt > 0:
            retry_suffix = (
                "\n\n[RETRY_INSTRUCTION]\n"
                + (
                    retry_instruction
                    or "직전 응답은 JSON 형식이 아니었다. 설명 없이 JSON object 하나만 다시 출력하라."
                )
            )
        raw = call_openai_compatible_chat(
            base_url=base_url,
            model_name=model_name,
            prompt=prompt + retry_suffix,
            temperature=temperature if attempt == 0 else 0.0,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            api_key=api_key,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        last_raw = raw
        try:
            payload = extract_json_block(raw)
            if not isinstance(payload, dict):
                raise ValueError("model output must be a JSON object")
            if validator is not None:
                return validator(payload), raw
            return payload, raw
        except Exception as exc:
            last_error = str(exc)
            continue
    raise ValueError(f"no JSON object found in model output after retries: {last_error}. raw={last_raw[:500]}")


def request_plain_text_response(
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
    validator: Callable[[str], str] | None = None,
    retry_instruction: str | None = None,
    system_prompt: str = "Return a plain Korean response only. Do not return JSON.",
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, str, dict[str, object]]:
    last_raw = ""
    validation_errors: list[str] = []
    for attempt in range(max_retries + 1):
        retry_suffix = ""
        if attempt > 0:
            retry_reason = validation_errors[-1] if validation_errors else "직전 응답이 형식 검증을 통과하지 못했다."
            retry_suffix = (
                "\n\n[RETRY_INSTRUCTION]\n"
                + (
                    retry_instruction
                    or "직전 응답은 plain Korean response 규칙을 어겼다. 같은 문장 반복, 미완성 문장, bullet/JSON을 피하고 자연스러운 한국어 평문으로만 다시 출력하라."
                )
                + f"\n- 직전 문제: {retry_reason}"
            )
        raw = call_openai_compatible_chat(
            base_url=base_url,
            model_name=model_name,
            prompt=prompt + retry_suffix,
            temperature=temperature if attempt == 0 else 0.0,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            system_prompt=system_prompt,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
        last_raw = raw
        try:
            validated = validator(raw) if validator is not None else str(raw).strip()
            return (
                validated,
                raw,
                {
                    "attempt_count": int(attempt + 1),
                    "retry_count": int(attempt),
                    "validation_errors": list(validation_errors),
                },
            )
        except Exception as exc:
            validation_errors.append(str(exc))
            continue
    last_error = validation_errors[-1] if validation_errors else "unknown validation error"
    raise ValueError(f"invalid plain-text response after retries: {last_error}. raw={last_raw[:500]}")
