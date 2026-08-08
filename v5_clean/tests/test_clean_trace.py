from __future__ import annotations

import numpy as np

from emonet_v5 import (
    DynamicsConfig,
    EmoNetV5Clean,
    HashingTextEncoder,
    build_controls,
    run_context_probe,
)


def build_model() -> EmoNetV5Clean:
    return EmoNetV5Clean(
        encoder=HashingTextEncoder(dimension=64),
        config=DynamicsConfig(
            num_neurons=64,
            recurrent_density=0.12,
            event_ticks=12,
            stimulation_ticks=5,
            seed=7,
        ),
    )


def test_same_seed_and_sequence_are_deterministic() -> None:
    model = build_model()
    model.reset_all()
    first = model.consume_sequence(["alpha context", "same final"])[-1]
    topology_a = model.topology_fingerprint

    model.reset_all()
    second = model.consume_sequence(["alpha context", "same final"])[-1]
    topology_b = model.topology_fingerprint

    assert topology_a == topology_b
    assert first.fingerprint() == second.fingerprint()
    assert np.array_equal(first.states, second.states)


def test_history_changes_same_final_trace_but_reset_removes_difference() -> None:
    model = build_model()
    result = run_context_probe(
        model,
        name="history_gate",
        context_a=["the previous event was calm and successful"],
        context_b=["the previous event was chaotic and unresolved"],
        final_text="the same final event arrived",
    )

    assert result.history_distance > 1e-8
    assert result.trace_a_fingerprint != result.trace_b_fingerprint
    assert result.reset_distance < 1e-8
    assert result.reset_a_fingerprint == result.reset_b_fingerprint


def test_transient_reset_preserves_recurrent_history() -> None:
    model = build_model()
    model.reset_all()
    model.consume_event("history event")
    before = model.dynamics.snapshot().state
    model.reset_transient()
    after = model.dynamics.snapshot().state

    assert np.array_equal(before, after)
    assert len(model.captured_traces) == 0


def test_canonical_controls_preserve_shapes() -> None:
    model = build_model()
    model.reset_all()
    traces = model.consume_sequence(["one", "two", "three"])
    controls = build_controls(traces, seed=123)

    assert set(controls) == {"real", "temporal_shuffle", "wrong_sample"}
    for key, rows in controls.items():
        assert len(rows) == len(traces), key
        for original, controlled in zip(traces, rows, strict=True):
            assert original.states.shape == controlled.states.shape

    assert any(
        original.fingerprint() != shuffled.fingerprint()
        for original, shuffled in zip(traces, controls["temporal_shuffle"], strict=True)
    )
    assert all(
        original.fingerprint() != wrong.fingerprint()
        for original, wrong in zip(traces, controls["wrong_sample"], strict=True)
    )
