from __future__ import annotations

import torch

from emonet_v7.text_encoder import LMStudioEmbeddingTextEncoder


class FakeEmbeddingClient:
    def embed(self, texts, *, model=None):
        assert model == "local-embedding-model"
        rows = []
        for index, _ in enumerate(texts):
            rows.append([1.0 + index, 2.0, 3.0])
        return rows


def test_lmstudio_embedding_encoder_probes_dimension_and_normalizes() -> None:
    encoder = LMStudioEmbeddingTextEncoder(FakeEmbeddingClient(), "local-embedding-model")
    assert encoder.output_dim == 3
    embeddings = encoder.encode(["a", "b"])
    assert embeddings.shape == (2, 3)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(2))
