from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence

import torch

TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_']+")


def tokenize(text: str, max_tokens: int = 96) -> List[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        tokens = ["<empty>"]
    return tokens[:max_tokens]


def hash_tokens(tokens: Sequence[str], vocab_size: int) -> torch.Tensor:
    ids = [abs(hash(tok)) % vocab_size for tok in tokens]
    return torch.tensor(ids, dtype=torch.long)


def cosine_similarity_list(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / math.sqrt(len(sa) * len(sb))


def longest_common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def pad_or_truncate(xs: Sequence[float], length: int, pad_value: float = 0.0) -> List[float]:
    out = list(xs[:length])
    if len(out) < length:
        out += [pad_value] * (length - len(out))
    return out


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    if not values:
        return default
    return float(sum(values) / len(values))
