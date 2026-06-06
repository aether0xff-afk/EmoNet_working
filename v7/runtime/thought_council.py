from __future__ import annotations

from dataclasses import dataclass

from .always_on import AlwaysOnEmoNetRuntime, RuntimeEvent, RuntimeSnapshot


@dataclass(frozen=True)
class ThoughtLine:
    tick_index: int
    speaker_id: str
    speaker_name: str
    target_id: str | None
    text: str
    intensity: float
    dominant_cluster_id: int | None

    def to_record(self) -> dict[str, float | int | str | None]:
        return {
            "tick_index": self.tick_index,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "target_id": self.target_id,
            "text": self.text,
            "intensity": self.intensity,
            "dominant_cluster_id": self.dominant_cluster_id,
        }


@dataclass
class ThoughtAgent:
    agent_id: str
    name: str
    focus: str
    runtime: AlwaysOnEmoNetRuntime
    last_snapshot: RuntimeSnapshot | None = None


class ThoughtCouncil:
    """A small sidecar made of multiple persistent EmoNet-like thought cells."""

    def __init__(self, *, max_lines: int = 80) -> None:
        self.tick_index = 0
        self.max_lines = max(8, int(max_lines))
        self.agents = [
            ThoughtAgent("feeling", "감각", "정서의 온도와 흔들림", AlwaysOnEmoNetRuntime(neuron_count=32, cluster_count=4, decay=0.88)),
            ThoughtAgent("guard", "경계", "위험과 부담", AlwaysOnEmoNetRuntime(neuron_count=32, cluster_count=4, decay=0.84)),
            ThoughtAgent("planner", "정리", "다음 행동과 구조", AlwaysOnEmoNetRuntime(neuron_count=32, cluster_count=4, decay=0.90)),
            ThoughtAgent("memory", "기억", "남는 흔적과 반복", AlwaysOnEmoNetRuntime(neuron_count=32, cluster_count=4, decay=0.92)),
        ]
        self.lines: list[ThoughtLine] = []

    def tick(self, *, event_kind: str, text: str = "", elapsed_seconds: float = 0.0) -> list[ThoughtLine]:
        event = RuntimeEvent(kind=event_kind, text=text, elapsed_seconds=elapsed_seconds)
        snapshots: list[tuple[ThoughtAgent, RuntimeSnapshot]] = []
        for agent in self.agents:
            snapshot = agent.runtime.tick(event)
            agent.last_snapshot = snapshot
            snapshots.append((agent, snapshot))

        new_lines = self._compose_exchange(event, snapshots)
        self.lines.extend(new_lines)
        if len(self.lines) > self.max_lines:
            del self.lines[: len(self.lines) - self.max_lines]
        self.tick_index += 1
        return new_lines

    def to_records(self, *, limit: int = 24) -> list[dict[str, float | int | str | None]]:
        return [line.to_record() for line in self.lines[-max(1, int(limit)) :]]

    def snapshot_records(self) -> list[dict[str, float | int | str | None]]:
        records: list[dict[str, float | int | str | None]] = []
        for agent in self.agents:
            snapshot = agent.last_snapshot
            if snapshot is None:
                records.append(
                    {
                        "agent_id": agent.agent_id,
                        "name": agent.name,
                        "focus": agent.focus,
                        "tick_index": None,
                        "dominant_cluster_id": None,
                        "mean_activity": 0.0,
                        "memory_load": 0.0,
                    }
                )
                continue
            activities = [abs(neuron.activation) for neuron in snapshot.neurons]
            records.append(
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "focus": agent.focus,
                    "tick_index": snapshot.tick_index,
                    "dominant_cluster_id": snapshot.dominant_cluster_id,
                    "mean_activity": sum(activities) / max(len(activities), 1),
                    "memory_load": sum(len(neuron.local_memory) for neuron in snapshot.neurons),
                }
            )
        return records

    def _compose_exchange(
        self,
        event: RuntimeEvent,
        snapshots: list[tuple[ThoughtAgent, RuntimeSnapshot]],
    ) -> list[ThoughtLine]:
        ranked = sorted(
            snapshots,
            key=lambda item: self._snapshot_intensity(item[1]),
            reverse=True,
        )
        if not ranked:
            return []

        primary_agent, primary_snapshot = ranked[0]
        secondary_agent = ranked[1][0] if len(ranked) > 1 else None
        lines: list[ThoughtLine] = [
            ThoughtLine(
                tick_index=self.tick_index,
                speaker_id=primary_agent.agent_id,
                speaker_name=primary_agent.name,
                target_id=secondary_agent.agent_id if secondary_agent else None,
                text=self._line_for_agent(primary_agent, primary_snapshot, event, secondary_agent),
                intensity=self._snapshot_intensity(primary_snapshot),
                dominant_cluster_id=primary_snapshot.dominant_cluster_id,
            )
        ]

        if event.kind in {"user_message", "delayed_speech"} and secondary_agent is not None:
            secondary_snapshot = ranked[1][1]
            lines.append(
                ThoughtLine(
                    tick_index=self.tick_index,
                    speaker_id=secondary_agent.agent_id,
                    speaker_name=secondary_agent.name,
                    target_id=primary_agent.agent_id,
                    text=self._reply_for_agent(secondary_agent, primary_agent, secondary_snapshot, event),
                    intensity=self._snapshot_intensity(secondary_snapshot),
                    dominant_cluster_id=secondary_snapshot.dominant_cluster_id,
                )
            )
        return lines

    def _snapshot_intensity(self, snapshot: RuntimeSnapshot) -> float:
        activity = sum(abs(neuron.activation) for neuron in snapshot.neurons) / max(len(snapshot.neurons), 1)
        memory = sum(len(neuron.local_memory) for neuron in snapshot.neurons) / max(len(snapshot.neurons), 1)
        return min(1.0, activity + memory * 0.08)

    def _line_for_agent(
        self,
        agent: ThoughtAgent,
        snapshot: RuntimeSnapshot,
        event: RuntimeEvent,
        target: ThoughtAgent | None,
    ) -> str:
        target_name = target.name if target else "다른 생각"
        if agent.agent_id == "feeling":
            return f"{target_name}, 지금 들어온 자극의 온도가 먼저 움직였어."
        if agent.agent_id == "guard":
            return f"{target_name}, 너무 빨리 밖으로 말하지 말고 부담을 한번 확인하자."
        if agent.agent_id == "planner":
            return f"{target_name}, 말이 나가려면 구조를 먼저 짧게 잡아야 해."
        if agent.agent_id == "memory":
            if snapshot.clusters and max(cluster.memory_load for cluster in snapshot.clusters) > 0:
                return f"{target_name}, 이건 남을 가능성이 있어. 같은 결을 표시해 둘게."
            return f"{target_name}, 아직 기억으로 굳힐 만큼 강하지는 않아."
        if event.kind == "idle":
            return f"{target_name}, 조용한 tick도 상태를 조금 바꾸고 있어."
        return f"{target_name}, 이 자극을 내부에서 더 돌려 보자."

    def _reply_for_agent(
        self,
        agent: ThoughtAgent,
        source: ThoughtAgent,
        snapshot: RuntimeSnapshot,
        event: RuntimeEvent,
    ) -> str:
        source_name = source.name
        if agent.agent_id == "feeling":
            return f"{source_name}, 맞아. 말보다 먼저 느낌의 압력이 생겼어."
        if agent.agent_id == "guard":
            return f"{source_name}, 그래도 바로 튀어나가면 어색해질 수 있어."
        if agent.agent_id == "planner":
            return f"{source_name}, 그러면 한 박자 늦추고 짧은 문장으로 내보내자."
        if agent.agent_id == "memory":
            memory_load = sum(len(neuron.local_memory) for neuron in snapshot.neurons)
            return f"{source_name}, 기억 부하는 {memory_load}야. 아직은 관찰로 두자."
        return f"{source_name}, 동의해."
