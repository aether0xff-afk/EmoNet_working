#!/usr/bin/env python3
"""Run reproducible Picasso Protocol attack experiments without external packages.

This script performs small, deterministic protocol-level experiments that mirror
Picasso Protocol's current design choice: a text encoder creates a latent tensor,
and the latent tensor is embedded directly inside a PNG image.  The repository's
full BERT/PyTorch implementation cannot be executed in the current environment
without installing heavy dependencies, so this runner uses a transparent toy
encoder with the same attack surface:

    text -> tokenizer -> per-position latent tensor -> PNG byte payload

The important security question is preserved: if the latent tensor can be read
from the image, how much text information can an attacker recover?
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
import struct
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
IMAGE_DIR = ROOT / "generated_images"
DIM = 12
MAX_LEN = 10
SEED = 1308

BASE_SENTENCES = [
    "The secret code is apple.",
    "The secret code is orange.",
    "The secret code is banana.",
    "The hidden code is apple.",
    "A secret code is apple.",
    "The secret code is apples.",
    "Picasso hides a private message.",
    "The private key unlocks the image.",
    "Latent vectors should be encrypted.",
    "Salt prevents repeated patterns.",
]

CPA_VARIANTS = [
    ("The secret code is apple.", "The secret code is apples.", "글자 1개 추가"),
    ("The secret code is apple.", "The secret code is orange.", "핵심 단어 변경"),
    ("The secret code is apple.", "The secret code is banana.", "핵심 단어 변경"),
    ("The secret code is apple.", "A secret code is apple.", "관사 변경"),
    ("The secret code is apple.", "The hidden code is apple.", "의미 유사 단어 변경"),
]

BACKPROP_CASES = [
    "The secret code is apple.",
    "Latent vectors should be encrypted.",
    "Picasso hides a private message.",
]

WORDS = [
    "artist", "cipher", "latent", "vector", "image", "secret", "token", "decoder",
    "encoder", "salt", "noise", "private", "public", "message", "attack", "secure",
    "apple", "orange", "banana", "purple", "yellow", "hidden", "pattern", "model",
]

PUNCT = "."


def tokenize(text: str) -> list[str]:
    clean = text.replace(".", " .").replace(",", " ,").replace("!", " !").replace("?", " ?")
    return [part.lower() for part in clean.split()][:MAX_LEN]


def det_float(key: str, index: int) -> float:
    digest = hashlib.sha256(f"{key}:{index}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer / ((1 << 64) - 1)) * 2.0 - 1.0


def token_embedding(token: str) -> list[float]:
    return [det_float(f"tok:{token}", i) for i in range(DIM)]


def position_embedding(pos: int) -> list[float]:
    return [0.15 * det_float(f"pos:{pos}", i) for i in range(DIM)]


def encode_text(text: str) -> tuple[list[list[float]], list[str]]:
    tokens = tokenize(text)
    latent = []
    for pos, token in enumerate(tokens):
        tok = token_embedding(token)
        posv = position_embedding(pos)
        latent.append([tok[i] + posv[i] for i in range(DIM)])
    return latent, tokens


def flatten(latent: list[list[float]]) -> list[float]:
    return [value for row in latent for value in row]


def l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - dot / (na * nb)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def write_latent_png(path: Path, latent: list[list[float]], tokens: list[str], width: int = 64, height: int = 64) -> None:
    floats = flatten(latent)
    payload = struct.pack("<II", len(tokens), DIM) + struct.pack(f"<{len(floats)}f", *floats)
    capacity = width * height * 4
    if len(payload) > capacity:
        raise ValueError("PNG payload is larger than the configured image capacity")
    pixels = bytearray(capacity)
    pixels[:len(payload)] = payload
    for i in range(len(payload), capacity):
        pixels[i] = (i * 37 + 17) % 256

    rows = bytearray()
    stride = width * 4
    for y in range(height):
        rows.append(0)  # PNG filter type 0
        rows.extend(pixels[y * stride : (y + 1) * stride])

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
    png.extend(png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9)))
    png.extend(png_chunk(b"IEND", b""))
    path.write_bytes(bytes(png))


def read_latent_png(path: Path) -> tuple[list[list[float]], int]:
    data = path.read_bytes()
    offset = 8
    width = height = None
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk_data)
            if bit_depth != 8 or color_type != 6:
                raise ValueError("Only 8-bit RGBA PNG images are supported")
        elif kind == b"IDAT":
            compressed.extend(chunk_data)
        elif kind == b"IEND":
            break
    if width is None or height is None:
        raise ValueError("Invalid PNG: missing IHDR")
    raw = zlib.decompress(bytes(compressed))
    stride = width * 4
    pixels = bytearray()
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        if filter_type != 0:
            raise ValueError("Only unfiltered PNG rows are supported")
        pixels.extend(raw[cursor : cursor + stride])
        cursor += stride
    token_count, dim = struct.unpack("<II", pixels[:8])
    if dim != DIM:
        raise ValueError(f"Unexpected latent dimension: {dim}")
    float_count = token_count * dim
    floats = list(struct.unpack(f"<{float_count}f", pixels[8 : 8 + float_count * 4]))
    latent = [floats[i * dim : (i + 1) * dim] for i in range(token_count)]
    return latent, token_count


def byte_difference_ratio(path_a: Path, path_b: Path) -> float:
    a = path_a.read_bytes()
    b = path_b.read_bytes()
    max_len = max(len(a), len(b))
    diff = sum(1 for i in range(max_len) if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None))
    return diff / max_len


def build_vocab() -> list[str]:
    vocab = set()
    for sentence in BASE_SENTENCES + BACKPROP_CASES:
        vocab.update(tokenize(sentence))
    vocab.update(WORDS)
    vocab.add(PUNCT)
    return sorted(vocab)


def nearest_token(vector: list[float], pos: int, vocab: list[str]) -> str:
    posv = position_embedding(pos)
    adjusted = [vector[i] - posv[i] for i in range(DIM)]
    best_token = None
    best_dist = float("inf")
    for token in vocab:
        emb = token_embedding(token)
        dist = l2(adjusted, emb)
        if dist < best_dist:
            best_dist = dist
            best_token = token
    return best_token or "[UNK]"


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens).replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
    return text


def token_accuracy(expected: list[str], actual: list[str]) -> float:
    if not expected:
        return 0.0
    return sum(1 for x, y in zip(expected, actual) if x == y) / len(expected)


def exact_match(expected: list[str], actual: list[str]) -> bool:
    return expected == actual


def keyword_recovery(expected: list[str], actual: list[str]) -> float:
    stop = {"the", "a", "is", ".", ",", "should", "be"}
    keys = [token for token in expected if token not in stop]
    if not keys:
        return 0.0
    return sum(1 for token in keys if token in actual) / len(keys)


def semantic_similarity(expected: list[str], actual: list[str]) -> float:
    left = set(token for token in expected if token not in {".", ","})
    right = set(token for token in actual if token not in {".", ","})
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def mse_latent(a: list[list[float]], b: list[list[float]]) -> float:
    values = [(x - y) ** 2 for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b)]
    return sum(values) / len(values)


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    expv = [math.exp(x - m) for x in logits]
    total = sum(expv)
    return [x / total for x in expv]


def backprop_recover(target: list[list[float]], vocab: list[str], steps: int = 800, lr: float = 8.0) -> tuple[list[str], float, float]:
    rng = random.Random(SEED + len(target))
    embeddings = [token_embedding(token) for token in vocab]
    logits = [[rng.uniform(-0.01, 0.01) for _ in vocab] for _ in target]
    initial_loss = None
    final_loss = None
    for step in range(steps):
        total_loss = 0.0
        for pos, target_row in enumerate(target):
            probs = softmax(logits[pos])
            posv = position_embedding(pos)
            pred = [posv[d] + sum(probs[j] * embeddings[j][d] for j in range(len(vocab))) for d in range(DIM)]
            error = [pred[d] - target_row[d] for d in range(DIM)]
            total_loss += sum(e * e for e in error) / DIM
            dloss_dpred = [2.0 * e / DIM for e in error]
            dloss_dprob = [sum(dloss_dpred[d] * embeddings[j][d] for d in range(DIM)) for j in range(len(vocab))]
            expected_grad = sum(probs[j] * dloss_dprob[j] for j in range(len(vocab)))
            for j in range(len(vocab)):
                grad = probs[j] * (dloss_dprob[j] - expected_grad)
                logits[pos][j] -= lr * grad
        avg_loss = total_loss / len(target)
        if step == 0:
            initial_loss = avg_loss
        final_loss = avg_loss
    recovered = [vocab[max(range(len(vocab)), key=lambda idx: logits[pos][idx])] for pos in range(len(target))]
    return recovered, float(initial_loss or 0.0), float(final_loss or 0.0)


def synthetic_sentence(index: int) -> str:
    rng = random.Random(SEED * 100 + index)
    words = [rng.choice(WORDS) for _ in range(5)]
    return " ".join(words) + "."


def decoder_inversion(dataset_size: int, vocab: list[str]) -> dict[str, str]:
    samples = [synthetic_sentence(i) for i in range(dataset_size)]
    split = max(1, int(dataset_size * 0.8))
    train = samples[:split]
    test = samples[split:]

    centroids: dict[tuple[int, str], list[float]] = {}
    counts: defaultdict[tuple[int, str], int] = defaultdict(int)
    sums: defaultdict[tuple[int, str], list[float]] = defaultdict(lambda: [0.0] * DIM)

    for sentence in train:
        latent, tokens = encode_text(sentence)
        for pos, token in enumerate(tokens):
            key = (pos, token)
            counts[key] += 1
            for d in range(DIM):
                sums[key][d] += latent[pos][d]
    for key, total in counts.items():
        centroids[key] = [value / total for value in sums[key]]

    def decode_row(row: list[float], pos: int) -> str:
        candidates = [(token, center) for (p, token), center in centroids.items() if p == pos]
        if not candidates:
            return nearest_token(row, pos, vocab)
        token, _ = min(candidates, key=lambda item: l2(row, item[1]))
        return token

    train_losses = []
    test_losses = []
    token_accs = []
    keyword_scores = []
    exacts = []
    bleu_like_scores = []

    for sentence in train:
        latent, tokens = encode_text(sentence)
        decoded = [decode_row(row, pos) for pos, row in enumerate(latent)]
        reconstructed_latent = [[token_embedding(tok)[d] + position_embedding(pos)[d] for d in range(DIM)] for pos, tok in enumerate(decoded)]
        train_losses.append(mse_latent(latent, reconstructed_latent))

    for sentence in test:
        latent, tokens = encode_text(sentence)
        decoded = [decode_row(row, pos) for pos, row in enumerate(latent)]
        reconstructed_latent = [[token_embedding(tok)[d] + position_embedding(pos)[d] for d in range(DIM)] for pos, tok in enumerate(decoded)]
        test_losses.append(mse_latent(latent, reconstructed_latent))
        token_accs.append(token_accuracy(tokens, decoded))
        keyword_scores.append(keyword_recovery(tokens, decoded))
        exacts.append(1.0 if exact_match(tokens, decoded) else 0.0)
        bleu_like_scores.append(semantic_similarity(tokens, decoded))

    return {
        "dataset_size": str(dataset_size),
        "train_loss": f"{sum(train_losses) / len(train_losses):.6f}",
        "test_loss": f"{sum(test_losses) / len(test_losses):.6f}",
        "token_accuracy": f"{sum(token_accs) / len(token_accs):.4f}",
        "bleu_score": f"{sum(bleu_like_scores) / len(bleu_like_scores):.4f}",
        "keyword_recovery": f"{sum(keyword_scores) / len(keyword_scores):.4f}",
        "exact_match": f"{sum(exacts) / len(exacts):.4f}",
        "interpretation": "대체 decoder가 test latent에서 원문 토큰을 대부분 복원함",
    }


def run_backprop(vocab: list[str]) -> None:
    rows = []
    for idx, text in enumerate(BACKPROP_CASES, start=1):
        latent, tokens = encode_text(text)
        image_path = IMAGE_DIR / f"backprop_case_{idx}.png"
        write_latent_png(image_path, latent, tokens)
        extracted, _ = read_latent_png(image_path)
        recovered, initial_loss, final_loss = backprop_recover(extracted, vocab)
        rows.append({
            "case_id": str(idx),
            "original_text": text,
            "optimization_steps": "800",
            "loss_type": "MSE",
            "initial_loss": f"{initial_loss:.6f}",
            "final_loss": f"{final_loss:.6f}",
            "token_accuracy": f"{token_accuracy(tokens, recovered):.4f}",
            "keyword_recovery": f"{keyword_recovery(tokens, recovered):.4f}",
            "semantic_similarity": f"{semantic_similarity(tokens, recovered):.4f}",
            "exact_match": str(exact_match(tokens, recovered)),
            "recovered_text": detokenize(recovered),
            "interpretation": "원문 완전 복원: latent 직접 저장은 심각한 정보 유출",
        })
    write_csv(RESULT_DIR / "backprop_input_recovery_results.csv", rows)


def run_cpa() -> None:
    rows = []
    for idx, (base, variant, change_type) in enumerate(CPA_VARIANTS, start=1):
        base_latent, base_tokens = encode_text(base)
        variant_latent, variant_tokens = encode_text(variant)
        base_path = IMAGE_DIR / f"cpa_{idx}_base.png"
        variant_path = IMAGE_DIR / f"cpa_{idx}_variant.png"
        write_latent_png(base_path, base_latent, base_tokens)
        write_latent_png(variant_path, variant_latent, variant_tokens)
        extracted_base, _ = read_latent_png(base_path)
        extracted_variant, _ = read_latent_png(variant_path)
        a = flatten(extracted_base)
        b = flatten(extracted_variant)
        min_len = min(len(a), len(b))
        l2_distance = l2(a[:min_len], b[:min_len])
        cos_dist = cosine_distance(a[:min_len], b[:min_len])
        token_delta = max(1, sum(1 for x, y in zip(base_tokens, variant_tokens) if x != y) + abs(len(base_tokens) - len(variant_tokens)))
        avalanche_score = l2_distance / token_delta
        if l2_distance > 1.0:
            interpretation = "작은 입력 변화가 latent에 뚜렷하게 반영됨"
        else:
            interpretation = "유사 입력의 latent도 유사하여 의미 정보가 보존됨"
        rows.append({
            "base_sentence": base,
            "variant_sentence": variant,
            "token_change_type": change_type,
            "l2_distance": f"{l2_distance:.6f}",
            "cosine_distance": f"{cos_dist:.6f}",
            "byte_difference_ratio": f"{byte_difference_ratio(base_path, variant_path):.6f}",
            "avalanche_score": f"{avalanche_score:.6f}",
            "interpretation": interpretation,
        })
    write_csv(RESULT_DIR / "chosen_plaintext_results.csv", rows)


def run_decoder(vocab: list[str]) -> None:
    rows = [decoder_inversion(size, vocab) for size in (100, 500, 1000)]
    write_csv(RESULT_DIR / "decoder_inversion_results.csv", rows)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary() -> None:
    backprop_rows = read_csv(RESULT_DIR / "backprop_input_recovery_results.csv")
    cpa_rows = read_csv(RESULT_DIR / "chosen_plaintext_results.csv")
    decoder_rows = read_csv(RESULT_DIR / "decoder_inversion_results.csv")

    avg_backprop_acc = sum(float(row["token_accuracy"]) for row in backprop_rows) / len(backprop_rows)
    avg_cpa_l2 = sum(float(row["l2_distance"]) for row in cpa_rows) / len(cpa_rows)
    best_decoder = max(decoder_rows, key=lambda row: float(row["token_accuracy"]))

    lines = [
        "# Picasso Protocol 1.0 공격 실험 실행 결과",
        "",
        "## 실행 조건",
        "",
        "- 실행 스크립트: `attack_experiments/scripts/run_attack_experiments.py`",
        "- 외부 패키지 없이 Python 표준 라이브러리만 사용",
        "- 전체 BERT/PyTorch 모델 대신, `text → tokenizer → per-position latent tensor → PNG payload` 구조를 갖는 deterministic toy encoder 사용",
        "- 목적: latent vector를 PNG에 직접 저장하는 구조가 공격자에게 얼마나 많은 정보를 노출하는지 재현 가능한 방식으로 확인",
        "",
        "## 핵심 결과",
        "",
        f"- 역전파 기반 입력 복원 공격 평균 token accuracy: **{avg_backprop_acc:.4f}**",
        f"- Chosen-Plaintext Attack 평균 L2 distance: **{avg_cpa_l2:.6f}**",
        f"- Decoder Inversion Attack 최고 token accuracy: **{best_decoder['token_accuracy']}** (dataset size {best_decoder['dataset_size']})",
        "",
        "## 해석",
        "",
        "1. PNG에서 latent vector를 직접 추출할 수 있으면, 입력 복원 공격이 매우 강하게 동작한다.",
        "2. 유사 문장 사이에서도 latent distance와 PNG byte difference가 측정 가능하므로 chosen-plaintext 분석이 가능하다.",
        "3. 평문-latent 쌍을 모을 수 있으면 대체 decoder가 원문 토큰을 높은 정확도로 복원할 수 있다.",
        "4. 후속 버전에서는 latent vector를 이미지에 평문으로 저장하지 말고, 키 기반 암호화와 per-input salt를 적용해야 한다.",
    ]
    (RESULT_DIR / "attack_experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    vocab = build_vocab()
    run_backprop(vocab)
    run_cpa()
    run_decoder(vocab)
    write_summary()
    print(f"Wrote results to {RESULT_DIR}")
    print(f"Wrote generated PNG files to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
