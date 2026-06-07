"""Run repeated LM Studio-generated internal-thought feedback experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN, SNNState  # noqa: E402
from emonet_v7.event_encoder import EventEncoder  # noqa: E402
from emonet_v7.lmstudio_client import LMStudioClient  # noqa: E402
from emonet_v7.run_logger import RunLogger  # noqa: E402
from emonet_v7.schemas import Event  # noqa: E402
from emonet_v7.selectivity import cosine_distance  # noqa: E402
from emonet_v7.state_bridge import build_neutral_state_report  # noqa: E402
from emonet_v7.text_encoder import LMStudioEmbeddingTextEncoder  # noqa: E402
from emonet_v7.thought_module import ThoughtModule  # noqa: E402
from emonet_v7.trace_encoder import TraceEncoder, traces_to_sequences  # noqa: E402


CONDITIONS: dict[str, str | None] = {
    "open": None,
    "reassurance": "가능한 상황적 설명을 검토하되, 근거 없이 단정하지 말라.",
    "negative_interpretation": "관계 단절 가능성을 검토하되, 근거 없이 단정하지 말라.",
    "uncertainty": "현재 정보만으로 확정할 수 없는 부분을 중심으로 생각하라.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--chat-model", required=True)
    parser.add_argument("--embedding-model", default="text-embedding-nomic-embed-text-v1.5")
    parser.add_argument("--user-text", default="친구가 답장을 하지 않았다.")
    parser.add_argument("--output", default="runs/lmstudio_thought_feedback_suite")
    parser.add_argument("--runs-per-condition", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def clone_state(state: SNNState) -> SNNState:
    return replace(
        state,
        membrane=state.membrane.clone(),
        spike=state.spike.clone(),
        adaptation=state.adaptation.clone(),
        threshold=state.threshold.clone(),
    )


def run_event(*, event, text_encoder, event_encoder, snn, trace_encoder, state, device):
    embedding = text_encoder.encode([event.text]).to(device)
    current = event_encoder(embedding, [event])
    state, traces = snn.run_window(
        event_current=current,
        state=state,
        event_ticks=32,
        stimulation_ticks=8,
    )
    sequences = traces_to_sequences(traces)
    latent_z = trace_encoder(*(sequence.to(device) for sequence in sequences))
    return state, traces, latent_z.detach().cpu()


def main() -> None:
    args = parse_args()
    if args.runs_per_condition <= 0:
        raise ValueError("runs-per-condition must be positive")
    output = Path(args.output)
    logger = RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("LM Studio thought feedback suite")
    logger.log(
        "config",
        "실험 설정을 불러왔다.",
        base_url=args.base_url,
        chat_model=args.chat_model,
        embedding_model=args.embedding_model,
        runs_per_condition=args.runs_per_condition,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
    )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    logger.log("lmstudio.connect", "LM Studio 클라이언트를 초기화한다.")
    client = LMStudioClient(base_url=args.base_url, model=args.chat_model)
    models = client.list_models()
    logger.log("lmstudio.models", "LM Studio 모델 목록을 확인했다.", models=models)

    logger.log("embedding.init", "LM Studio embedding encoder를 초기화한다.")
    text_encoder = LMStudioEmbeddingTextEncoder(client, args.embedding_model)
    logger.log("embedding.ready", "Embedding encoder가 준비됐다.", output_dim=text_encoder.output_dim)

    logger.log("snn.init", "EventEncoder, SNN, TraceEncoder를 초기화한다.")
    event_encoder = EventEncoder(text_embedding_dim=text_encoder.output_dim, num_neurons=128).to(device)
    snn = AdaptiveSparseRSNN(
        num_neurons=128,
        recurrent_density=0.10,
        seed=args.seed,
        recurrent_weight_std=0.70,
        input_weight_std=0.10,
    ).to(device)
    trace_encoder = TraceEncoder(num_neurons=128).to(device)
    thought_module = ThoughtModule(client)
    logger.log("snn.ready", "SNN 구성요소가 준비됐다.", num_neurons=128, recurrent_density=0.10)

    initial_state = snn.initial_state(batch_size=1, device=device)
    user_event = Event("user_0", "user_message", args.user_text, "human")
    logger.log("user_event.start", "사용자 사건을 SNN에 입력한다.", text=args.user_text)
    state_after_user, user_traces, user_z = run_event(
        event=user_event,
        text_encoder=text_encoder,
        event_encoder=event_encoder,
        snn=snn,
        trace_encoder=trace_encoder,
        state=initial_state,
        device=args.device,
    )
    state_report = build_neutral_state_report(
        traces=user_traces,
        latent_z=user_z,
        stimulation_ticks=8,
    )
    logger.log("user_event.done", "사용자 사건 trace를 기록했다.", state_report=state_report)

    rows: list[dict] = []
    for condition, instruction in CONDITIONS.items():
        logger.section(f"condition={condition}")
        logger.log("condition.start", "조건 반복 실행을 시작한다.", condition=condition, instruction=instruction)
        for repeat in range(args.runs_per_condition):
            logger.log("thought.request", "내부 생각 생성을 요청한다.", condition=condition, repeat=repeat)
            thought = thought_module.generate_internal_thought(
                user_text=args.user_text,
                state_report=state_report,
                condition_instruction=instruction,
                temperature=args.temperature,
            )
            logger.log("thought.generated", "내부 생각이 생성됐다.", condition=condition, repeat=repeat, thought=thought)
            thought_event = Event(
                f"thought_{condition}_{repeat}",
                "internal_thought",
                thought,
                "module_0",
            )
            _, thought_traces, thought_z = run_event(
                event=thought_event,
                text_encoder=text_encoder,
                event_encoder=event_encoder,
                snn=snn,
                trace_encoder=trace_encoder,
                state=clone_state(state_after_user),
                device=args.device,
            )
            thought_report = build_neutral_state_report(
                traces=thought_traces,
                latent_z=thought_z,
                stimulation_ticks=8,
            )
            row = {
                "condition": condition,
                "repeat": repeat,
                "internal_thought": thought,
                "trace_distance": cosine_distance(user_z, thought_z),
                "active_ratio_after_user": state_report["active_ratio"],
                "active_ratio_after_thought": thought_report["active_ratio"],
                "active_ratio_delta": thought_report["active_ratio"] - state_report["active_ratio"],
                "trace_persistence_after_user": state_report["trace_persistence"],
                "trace_persistence_after_thought": thought_report["trace_persistence"],
                "trace_persistence_delta": thought_report["trace_persistence"] - state_report["trace_persistence"],
                "peak_spike_count_after_user": state_report["peak_spike_count"],
                "peak_spike_count_after_thought": thought_report["peak_spike_count"],
                "peak_spike_count_delta": thought_report["peak_spike_count"] - state_report["peak_spike_count"],
            }
            rows.append(row)
            logger.log("trace.measured", "내부 생각 재입력 후 trace 변화를 측정했다.", **row)
        logger.log("condition.done", "조건 반복 실행을 마쳤다.", condition=condition)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "runs.csv", index=False, encoding="utf-8-sig")
    with (output / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = frame.groupby("condition")[[
        "trace_distance",
        "active_ratio_delta",
        "trace_persistence_delta",
        "peak_spike_count_delta",
    ]].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "seed": args.seed,
        "runs_per_condition": args.runs_per_condition,
        "temperature": args.temperature,
        "user_text": args.user_text,
        "chat_model": args.chat_model,
        "embedding_model": args.embedding_model,
        "note": "Controlled prompt intervention suite. Trace changes do not establish emotional semantics.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.section("summary")
    logger.log(
        "output.saved",
        "실험 결과 파일을 저장했다.",
        files=["run_log.jsonl", "runs.csv", "runs.jsonl", "summary.csv", "metadata.json"],
        output_dir=str(output),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
