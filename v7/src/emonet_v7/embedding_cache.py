"""Persistent embedding cache for local semantic training runs."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

import torch


class CachedTextEncoder:
    """Wrap a text encoder and cache normalized embeddings on disk."""

    def __init__(self, encoder, cache_path: str | Path) -> None:
        self.encoder = encoder
        self.output_dim = int(encoder.output_dim)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        if self.cache_path.exists():
            loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("embedding cache must contain a JSON object")
            self._cache = {str(key): list(value) for key, value in loaded.items()}

    @staticmethod
    def _key(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        for text in texts:
            key = self._key(text)
            if key not in self._cache:
                missing_texts.append(text)
                missing_keys.append(key)
        if missing_texts:
            embeddings = self.encoder.encode(missing_texts).detach().cpu().to(torch.float32)
            if embeddings.ndim != 2 or embeddings.shape[1] != self.output_dim:
                raise RuntimeError("wrapped encoder returned invalid embedding shape")
            for key, row in zip(missing_keys, embeddings, strict=True):
                self._cache[key] = [float(value) for value in row]
            self.flush()
        rows = [self._cache[self._key(text)] for text in texts]
        return torch.tensor(rows, dtype=torch.float32)

    def flush(self) -> None:
        """Persist current cache contents."""

        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
