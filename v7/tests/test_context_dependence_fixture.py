from __future__ import annotations

from pathlib import Path

import yaml

from emonet_v7.episode_dataset import load_episodes


def test_context_fixture_pairs_share_current_text_and_differ_in_history() -> None:
    fixture = Path("fixtures/context_dependence_episodes.yaml")
    episodes = {episode.episode_id: episode for episode in load_episodes(fixture)}
    data = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    pairs = data["contrast_pairs"]
    assert pairs

    for pair in pairs:
        left = episodes[pair["left"]]
        right = episodes[pair["right"]]
        step_index = int(pair["step_index"])
        assert left.events[step_index].text == right.events[step_index].text
        assert left.events[0].text != right.events[0].text
        assert left.events[step_index + 1].text != right.events[step_index + 1].text
