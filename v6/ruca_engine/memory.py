from __future__ import annotations

import json
import re
from pathlib import Path

from .emotion import InputSignals
from .models import EmotionState, MemoryItem, utc_now_iso


class MemoryStore:
    def __init__(self, path: Path | None = None, max_short_term: int = 12) -> None:
        self.path = path
        self.max_short_term = max(1, int(max_short_term))
        self._items: list[MemoryItem] = []
        if path is not None and path.exists():
            self._items = self._read(path)

    @classmethod
    def from_items(cls, items: list[MemoryItem] | None = None, max_short_term: int = 12) -> "MemoryStore":
        store = cls(path=None, max_short_term=max_short_term)
        store._items = list(items or [])
        return store

    def all_items(self) -> tuple[MemoryItem, ...]:
        return tuple(self._items)

    def retrieve(self, text: str, limit: int = 5) -> tuple[MemoryItem, ...]:
        query_terms = {part.lower() for part in (text or "").split() if len(part) >= 2}
        scored: list[tuple[float, MemoryItem]] = []
        now = utc_now_iso()
        updated_items: list[MemoryItem] = []
        returned_ids: set[str] = set()
        for item in self._items:
            haystack = f"{item.summary} {item.source_event}".lower()
            overlap = sum(1 for term in query_terms if term in haystack)
            score = item.importance + overlap * 0.15
            if item.memory_type == "relationship":
                score += 0.10
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = tuple(item for _, item in scored[: max(0, int(limit))])
        returned_ids.update(item.memory_id for item in selected)
        for item in self._items:
            if item.memory_id in returned_ids:
                updated_items.append(MemoryItem(**{**item.to_record(), "last_accessed_at": now}))
            else:
                updated_items.append(item)
        if returned_ids:
            self._items = updated_items
            self.save()
        return tuple(item for item in self._items if item.memory_id in returned_ids)

    def observe_turn(
        self,
        *,
        user_text: str,
        assistant_text: str,
        emotion_state: EmotionState,
        signals: InputSignals,
    ) -> MemoryItem | None:
        importance = max(signals.alarm, signals.warmth * 0.8, signals.action_pressure * 0.7, signals.intensity)
        if importance < 0.24 and not user_text.strip():
            return None
        memory_type = "short_term"
        if importance >= 0.62:
            memory_type = "relationship" if signals.warmth >= signals.alarm else "long_term"
        summary = _summarize_event(user_text=user_text, assistant_text=assistant_text, signals=signals)
        item = MemoryItem(
            memory_id=self._next_memory_id(),
            memory_type=memory_type,
            summary=summary,
            source_event=user_text.strip(),
            emotion_snapshot=emotion_state.to_record(),
            importance=round(float(importance), 3),
        )
        self._items.append(item)
        self._trim_short_term()
        self.save()
        return item

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [item.to_record() for item in self._items]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _trim_short_term(self) -> None:
        short_items = [item for item in self._items if item.memory_type == "short_term"]
        if len(short_items) <= self.max_short_term:
            return
        keep_short_ids = {item.memory_id for item in short_items[-self.max_short_term :]}
        self._items = [item for item in self._items if item.memory_type != "short_term" or item.memory_id in keep_short_ids]

    def _next_memory_id(self) -> str:
        max_seen = 0
        for item in self._items:
            match = re.fullmatch(r"mem-(\d+)", item.memory_id)
            if match:
                max_seen = max(max_seen, int(match.group(1)))
        return f"mem-{max_seen + 1:04d}"

    @staticmethod
    def _read(path: Path) -> list[MemoryItem]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"memory file must contain a list: {path}")
        return [MemoryItem.from_mapping(item) for item in payload if isinstance(item, dict)]


def _summarize_event(*, user_text: str, assistant_text: str, signals: InputSignals) -> str:
    text = " ".join((user_text or "").strip().split())
    if len(text) > 96:
        text = text[:93].rstrip() + "..."
    if signals.alarm >= 0.55:
        prefix = "사용자가 불안/위험 신호를 보냄"
    elif signals.warmth >= 0.50:
        prefix = "사용자가 긍정적 관계 신호를 보냄"
    elif signals.action_pressure >= 0.50:
        prefix = "사용자가 행동 또는 구현을 요청함"
    else:
        prefix = "최근 상호작용"
    return f"{prefix}: {text}" if text else prefix
