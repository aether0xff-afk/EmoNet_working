from __future__ import annotations

from pathlib import Path

from emonet_v7.episode_dataset import iter_transitions, load_episodes, select_split


def test_load_episodes_and_transitions(tmp_path: Path) -> None:
    fixture = tmp_path / "episodes.yaml"
    fixture.write_text(
        """
episodes:
  - id: train_demo
    split: train
    events:
      - text: first
      - kind: internal_thought
        speaker_id: module_0
        text: second
      - kind: module_message
        speaker_id: module_0
        text: third
  - id: validation_demo
    split: validation
    events:
      - text: alpha
      - text: beta
""".strip(),
        encoding="utf-8",
    )
    episodes = load_episodes(fixture)
    assert len(episodes) == 2
    assert [episode.episode_id for episode in select_split(episodes, "train")] == ["train_demo"]
    transitions = list(iter_transitions(episodes[0]))
    assert len(transitions) == 2
    assert transitions[0].current.text == "first"
    assert transitions[0].target.text == "second"
    assert transitions[1].current.kind == "internal_thought"
    assert transitions[1].target.kind == "module_message"
