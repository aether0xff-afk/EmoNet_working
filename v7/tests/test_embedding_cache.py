from __future__ import annotations

from pathlib import Path

import torch

from emonet_v7.embedding_cache import CachedTextEncoder


class CountingEncoder:
    output_dim = 3

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return torch.tensor([[1.0, 2.0, 3.0] for _ in texts], dtype=torch.float32)


def test_cached_encoder_avoids_duplicate_calls(tmp_path: Path) -> None:
    base = CountingEncoder()
    cache = CachedTextEncoder(base, tmp_path / "cache.json")
    first = cache.encode(["hello", "world"])
    second = cache.encode(["hello", "world"])
    assert base.calls == 1
    assert torch.allclose(first, second)
    assert (tmp_path / "cache.json").exists()


def test_cached_encoder_reloads_persisted_values(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    first_base = CountingEncoder()
    CachedTextEncoder(first_base, path).encode(["hello"])

    second_base = CountingEncoder()
    reloaded = CachedTextEncoder(second_base, path)
    result = reloaded.encode(["hello"])
    assert second_base.calls == 0
    assert result.shape == (1, 3)
