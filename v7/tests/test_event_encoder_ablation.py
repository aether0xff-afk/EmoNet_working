from __future__ import annotations

import torch

from emonet_v7.event_encoder import EventEncoder
from emonet_v7.schemas import Event


def test_text_only_ablation_ignores_event_kind_and_speaker() -> None:
    torch.manual_seed(11)
    encoder = EventEncoder(
        text_embedding_dim=8,
        num_neurons=4,
        include_event_kind=False,
        include_speaker=False,
        include_elapsed_time=False,
    )
    embedding = torch.randn(1, 8)
    user_event = Event("1", "user_message", "same", "human", elapsed_seconds=0.0)
    thought_event = Event("2", "internal_thought", "same", "module_0", elapsed_seconds=99.0)
    user_current = encoder(embedding, [user_event])
    thought_current = encoder(embedding, [thought_event])
    assert torch.allclose(user_current, thought_current)


def test_metadata_enabled_changes_current_for_same_text() -> None:
    torch.manual_seed(11)
    encoder = EventEncoder(
        text_embedding_dim=8,
        num_neurons=4,
        include_event_kind=True,
        include_speaker=True,
        include_elapsed_time=False,
    )
    embedding = torch.randn(1, 8)
    user_event = Event("1", "user_message", "same", "human")
    thought_event = Event("2", "internal_thought", "same", "module_0")
    user_current = encoder(embedding, [user_event])
    thought_current = encoder(embedding, [thought_event])
    assert not torch.allclose(user_current, thought_current)
