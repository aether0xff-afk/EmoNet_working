from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Protocol
from urllib import request

import numpy as np


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


class TextEncoder(Protocol):
    """Frozen text representation interface used by the v5 core."""

    @property
    def output_dim(self) -> int:
        ...

    def encode(self, text: str) -> np.ndarray:
        ...


@dataclass
class HashingTextEncoder:
    """Deterministic non-affective encoder for tests and smoke runs only.

    This encoder intentionally has no emotion dictionary, appraisal mapping, or
    hand-authored affect rule. It is not a semantic benchmark encoder and must
    not be used to make representation-quality claims.
    """

    dimension: int = 96

    @property
    def output_dim(self) -> int:
        return self.dimension

    def encode(self, text: str) -> np.ndarray:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = _TOKEN_RE.findall(str(text).lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            weight = 1.0 / math.sqrt(max(1, len(token)))
            vector[index] += sign * weight
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector


@dataclass
class LMStudioEmbeddingEncoder:
    """OpenAI-compatible frozen embedding backend, e.g. LM Studio.

    `dimension` is explicit so the EmoNet substrate can be initialized without
    probing the endpoint during construction. The returned embedding is checked
    against this dimension on every call.
    """

    model: str
    dimension: int
    base_url: str = "http://127.0.0.1:1234/v1"
    timeout_sec: int = 60

    @property
    def output_dim(self) -> int:
        return self.dimension

    def encode(self, text: str) -> np.ndarray:
        payload = json.dumps(
            {"model": self.model, "input": str(text)},
            ensure_ascii=False,
        ).encode("utf-8")
        req = request.Request(
            self.base_url.rstrip("/") + "/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_sec) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        try:
            values = decoded["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("embedding endpoint returned an unexpected payload") from exc
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"expected embedding dimension {self.dimension}, got {vector.shape}"
            )
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector
