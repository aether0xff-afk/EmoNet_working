"""Lazy LM Studio connector using its OpenAI-compatible API."""

from __future__ import annotations


class LMStudioClient:
    """Small client boundary for later internal-thought generation."""

    def __init__(self, *, base_url: str, model: str, api_key: str = "lm-studio") -> None:
        normalized = base_url.rstrip("/")
        self.base_url = normalized if normalized.endswith("/v1") else f"{normalized}/v1"
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the optional 'llm' dependency to use LM Studio") from exc
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.7) -> str:
        result = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        content = result.choices[0].message.content
        if content is None:
            raise RuntimeError("LM Studio returned an empty response")
        return content

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        result = self._get_client().embeddings.create(
            model=model or self.model,
            input=texts,
        )
        return [item.embedding for item in result.data]
