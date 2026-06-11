"""Episode dataset utilities for semantic-dynamics training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .schemas import Event


@dataclass(frozen=True)
class Episode:
    """Ordered events that share one persistent SNN state."""

    episode_id: str
    split: str
    events: tuple[Event, ...]


@dataclass(frozen=True)
class Transition:
    """One current-event to next-event prediction target."""

    episode_id: str
    step_index: int
    current: Event
    target: Event


def load_episodes(path: str | Path) -> list[Episode]:
    """Load and validate YAML episodes."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("episodes"), list):
        raise ValueError("episode fixture must contain an episodes list")

    episodes: list[Episode] = []
    seen_ids: set[str] = set()
    for item in data["episodes"]:
        if not isinstance(item, dict):
            raise ValueError("each episode must be an object")
        episode_id = str(item.get("id", "")).strip()
        split = str(item.get("split", "")).strip()
        raw_events = item.get("events")
        if not episode_id:
            raise ValueError("episode id must not be empty")
        if episode_id in seen_ids:
            raise ValueError(f"duplicate episode id: {episode_id}")
        if split not in {"train", "validation"}:
            raise ValueError(f"unsupported split for {episode_id}: {split}")
        if not isinstance(raw_events, list) or len(raw_events) < 2:
            raise ValueError(f"episode {episode_id} must contain at least two events")
        seen_ids.add(episode_id)
        events: list[Event] = []
        for index, raw_event in enumerate(raw_events):
            if not isinstance(raw_event, dict):
                raise ValueError(f"episode {episode_id} event {index} must be an object")
            events.append(
                Event(
                    event_id=f"{episode_id}:{index}",
                    kind=raw_event.get("kind", "user_message"),
                    text=str(raw_event.get("text", "")).strip(),
                    speaker_id=str(raw_event.get("speaker_id", "human")).strip(),
                    elapsed_seconds=float(raw_event.get("elapsed_seconds", 0.0)),
                )
            )
            if not events[-1].text:
                raise ValueError(f"episode {episode_id} event {index} text must not be empty")
        episodes.append(Episode(episode_id=episode_id, split=split, events=tuple(events)))
    return episodes


def select_split(episodes: Iterable[Episode], split: str) -> list[Episode]:
    """Return episodes for one split."""

    if split not in {"train", "validation"}:
        raise ValueError(f"unsupported split: {split}")
    return [episode for episode in episodes if episode.split == split]


def iter_transitions(episode: Episode) -> Iterable[Transition]:
    """Yield ordered next-event prediction pairs inside one episode."""

    for index in range(len(episode.events) - 1):
        yield Transition(
            episode_id=episode.episode_id,
            step_index=index,
            current=episode.events[index],
            target=episode.events[index + 1],
        )
