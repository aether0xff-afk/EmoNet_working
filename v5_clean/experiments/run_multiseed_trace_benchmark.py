from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from emonet_v5 import DynamicsConfig, EmoNetV5Clean, HashingTextEncoder
from emonet_v5.evaluation import build_controls, normalized_l2, run_context_probe, trace_distance


SEEDS = [7, 13, 21, 42, 100]

CONTEXT_PAIRS = [
    {
        "name": "praise_vs_failure",
        "context_a": ["발표가 잘 끝났다", "선생님이 칭찬했다"],
        "context_b": ["발표에서 크게 실수했다", "친구들이 웃었다"],
        "final_text": "선생님이 나를 불렀다",
    },
    {
        "name": "support_vs_conflict",
        "context_a": ["친구가 내 편을 들어줬다", "같이 해결하자고 했다"],
        "context_b": ["친구와 크게 다퉜다", "서로 말을 끊었다"],
        "final_text": "친구에게서 메시지가 왔다",
    },
    {
        "name": "success_vs_uncertainty",
        "context_a": ["실험이 예상대로 성공했다", "결과도 재현됐다"],
        "context_b": ["실험 결과가 계속 흔들렸다", "원인을 아직 못 찾았다"],
        "final_text": "새로운 결과가 하나 나왔다",
    },
    {
        "name": "acceptance_vs_rejection",
        "context_a": ["지원한 곳에서 좋은 반응이 왔다", "면담 분위기도 좋았다"],
        "context_b": ["지원한 곳에서 거절 메일이 왔다", "이유는 적혀 있지 않았다"],
        "final_text": "메일 알림이 하나 더 왔다",
    },
    {
        "name": "safety_vs_threat",
        "context_a": ["문제가 해결됐다는 연락을 받았다", "주변도 조용해졌다"],
        "context_b": ["경고 알림이 반복해서 떴다", "문제가 더 커질 수 있다고 했다"],
        "final_text": "새 알림이 화면에 떴다",
    },
    {
        "name": "trust_vs_betrayal",
        "context_a": ["비밀을 맡긴 친구가 약속을 지켰다", "내 편이 되어줬다"],
        "context_b": ["비밀을 맡긴 친구가 다른 사람에게 말했다", "내 말을 부정했다"],
        "final_text": "그 친구가 다시 말을 걸었다",
    },
    {
        "name": "progress_vs_stall",
        "context_a": ["코드가 하나씩 정상 동작했다", "테스트도 계속 통과했다"],
        "context_b": ["같은 오류가 계속 반복됐다", "수정해도 결과가 바뀌지 않았다"],
        "final_text": "테스트 결과가 새로 나왔다",
    },
    {
        "name": "welcome_vs_exclusion",
        "context_a": ["모임에서 먼저 자리를 만들어줬다", "다 같이 이야기하자고 했다"],
        "context_b": ["모임에서 내 자리만 빠져 있었다", "대화에도 끼워주지 않았다"],
        "final_text": "단체 채팅에 새 메시지가 올라왔다",
    },
    {
        "name": "relief_vs_pressure",
        "context_a": ["마감이 연장됐다는 공지가 왔다", "해야 할 일이 줄었다"],
        "context_b": ["마감이 오늘로 당겨졌다", "추가 작업도 생겼다"],
        "final_text": "새 공지가 하나 올라왔다",
    },
    {
        "name": "recovery_vs_exhaustion",
        "context_a": ["충분히 쉬고 다시 시작했다", "몸 상태도 괜찮아졌다"],
        "context_b": ["밤새 거의 쉬지 못했다", "계속 일을 이어갔다"],
        "final_text": "다음 작업을 시작할 시간이 됐다",
    },
    {
        "name": "cooperation_vs_opposition",
        "context_a": ["팀원이 내 제안에 동의했다", "역할도 나눠 맡았다"],
        "context_b": ["팀원이 내 제안을 계속 반대했다", "역할 분담도 거부했다"],
        "final_text": "팀 회의가 다시 시작됐다",
    },
    {
        "name": "clarity_vs_ambiguity",
        "context_a": ["문제 원인을 정확히 찾았다", "재현 조건도 확인했다"],
        "context_b": ["문제 원인이 여러 개로 보였다", "재현 조건도 일정하지 않았다"],
        "final_text": "새 로그가 하나 추가됐다",
    },
]


def make_model(seed: int) -> EmoNetV5Clean:
    encoder = HashingTextEncoder(dimension=96)
    config = DynamicsConfig(seed=seed)
    return EmoNetV5Clean(encoder=encoder, config=config)


def collect_final_traces(model: EmoNetV5Clean) -> list:
    traces = []
    for fixture in CONTEXT_PAIRS:
        model.reset_all()
        model.consume_sequence(fixture["context_a"])
        traces.append(model.consume_event(fixture["final_text"]))
    return traces


def main() -> None:
    out_dir = Path("outputs/multiseed_trace_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        model = make_model(seed)
        for fixture in CONTEXT_PAIRS:
            result = run_context_probe(
                model,
                name=fixture["name"],
                context_a=fixture["context_a"],
                context_b=fixture["context_b"],
                final_text=fixture["final_text"],
            )
            rows.append({"seed": seed, **result.to_dict()})

        final_traces = collect_final_traces(model)
        controls = build_controls(final_traces, seed=seed)
        real = controls["real"]
        shuffled = controls["temporal_shuffle"]
        wrong = controls["wrong_sample"]
        for index, fixture in enumerate(CONTEXT_PAIRS):
            control_rows.append(
                {
                    "seed": seed,
                    "name": fixture["name"],
                    "real_vs_temporal_shuffle": trace_distance(real[index], shuffled[index]),
                    "real_vs_wrong_sample": trace_distance(real[index], wrong[index]),
                    "real_final_state_rms": float(np.sqrt(np.mean(real[index].final_state ** 2))),
                    "real_trace_fingerprint": real[index].fingerprint(),
                    "shuffled_trace_fingerprint": shuffled[index].fingerprint(),
                    "wrong_trace_fingerprint": wrong[index].fingerprint(),
                }
            )

    history = np.asarray([float(row["history_distance"]) for row in rows], dtype=np.float64)
    reset = np.asarray([float(row["reset_distance"]) for row in rows], dtype=np.float64)
    shuffle_dist = np.asarray(
        [float(row["real_vs_temporal_shuffle"]) for row in control_rows], dtype=np.float64
    )
    wrong_dist = np.asarray(
        [float(row["real_vs_wrong_sample"]) for row in control_rows], dtype=np.float64
    )

    summary = {
        "purpose": "recurrent trace mechanism benchmark; not a semantic or affect claim",
        "encoder": "HashingTextEncoder",
        "seeds": SEEDS,
        "fixture_count": len(CONTEXT_PAIRS),
        "runs": len(rows),
        "history_distance": {
            "mean": float(history.mean()),
            "std": float(history.std()),
            "min": float(history.min()),
            "max": float(history.max()),
            "positive_fraction": float(np.mean(history > 1e-8)),
        },
        "reset_distance": {
            "mean": float(reset.mean()),
            "max": float(reset.max()),
            "zero_fraction": float(np.mean(reset <= 1e-8)),
        },
        "controls": {
            "real_vs_temporal_shuffle_mean": float(shuffle_dist.mean()),
            "real_vs_wrong_sample_mean": float(wrong_dist.mean()),
        },
        "acceptance": {
            "all_history_pairs_change_trace": bool(np.all(history > 1e-8)),
            "all_reset_pairs_remove_difference": bool(np.all(reset <= 1e-8)),
            "controls_are_nonidentical": bool(
                np.all(shuffle_dist > 1e-8) and np.all(wrong_dist > 1e-8)
            ),
        },
    }

    with (out_dir / "context_probe_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "control_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(control_rows[0].keys()))
        writer.writeheader()
        writer.writerows(control_rows)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not all(summary["acceptance"].values()):
        raise SystemExit("multiseed trace benchmark acceptance failed")


if __name__ == "__main__":
    main()
