from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TASKS = ("alternation", "palindrome", "repeat_gap", "repeat_position")
PAIR_COUNT = 120
TRAIN_PAIRS = 80
CALIBRATION_TRAIN_PAIRS = 60
CALIBRATION_VALIDATION_PAIRS = 20

SYLLABLES = (
    "ka", "zu", "mi", "tor", "vel", "shi", "na", "dor", "pel", "rin",
    "qua", "fen", "lum", "bar", "cek", "ivo", "ryn", "sol", "tek", "uma",
)


@dataclass(frozen=True)
class TemporalCase:
    task: str
    pair_id: int
    class0: tuple[str, ...]
    class1: tuple[str, ...]
    current: str
    identities: tuple[str, ...]


def token_name(global_index: int, side: int) -> str:
    a = SYLLABLES[(global_index * 7 + side * 3) % len(SYLLABLES)]
    b = SYLLABLES[(global_index * 11 + side * 5 + 1) % len(SYLLABLES)]
    c = SYLLABLES[(global_index * 13 + side * 7 + 2) % len(SYLLABLES)]
    return f"{a}{b}{c}-{global_index:05d}-{side}"


def build_case(task: str, pair_id: int) -> TemporalCase:
    if task not in TASKS:
        raise ValueError(task)
    if not 0 <= pair_id < PAIR_COUNT:
        raise ValueError(pair_id)

    task_index = TASKS.index(task)
    global_index = task_index * 1000 + pair_id
    names = tuple(token_name(global_index, side) for side in range(3))
    a = f"The transient marker {names[0]} appeared."
    b = f"The transient marker {names[1]} appeared."
    c = f"The transient marker {names[2]} appeared."

    patterns: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "alternation": ((a, b, a, b), (a, a, b, b)),
        "palindrome": ((a, b, b, a), (a, a, b, b)),
        "repeat_gap": ((a, b, c, a), (a, a, b, c)),
        "repeat_position": ((a, b, c, a), (a, b, a, c)),
    }
    pattern0, pattern1 = patterns[task]
    prefix = f"Temporal structure case {task}-{pair_id:03d} begins with a neutral marker."
    suffix = "The same neutral separator appears after the four transient markers."
    current = "The identical current observation is now presented."
    class0 = (prefix, *pattern0, suffix)
    class1 = (prefix, *pattern1, suffix)
    return TemporalCase(
        task=task,
        pair_id=pair_id,
        class0=class0,
        class1=class1,
        current=current,
        identities=names,
    )


def split_name(pair_id: int) -> str:
    if pair_id < CALIBRATION_TRAIN_PAIRS:
        return "calibration_train"
    if pair_id < TRAIN_PAIRS:
        return "calibration_validation"
    return "test"


def relational_features(sequence: tuple[str, ...] | list[str], encoder) -> np.ndarray:
    transient = list(sequence)[1:5]
    vectors = np.stack([encoder.encode(text) for text in transient])
    values: list[float] = []
    for i in range(4):
        for j in range(i + 1, 4):
            values.append(float(np.dot(vectors[i], vectors[j])))
    return np.asarray(values, dtype=np.float32)
