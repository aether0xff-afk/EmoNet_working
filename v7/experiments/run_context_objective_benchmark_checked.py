"""Run the context benchmark and evaluate each model from its best validation checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

import run_context_objective_benchmark as base


def load_state_dict_for(model_type: str, model, checkpoint: dict[str, Any]) -> None:
    """Restore one benchmark model from a saved checkpoint payload."""

    if model_type.startswith("snn_"):
        model.event_encoder.load_state_dict(checkpoint["event_encoder"])
        model.snn.load_state_dict(checkpoint["snn"])
        model.trace_encoder.load_state_dict(checkpoint["trace_encoder"])
        model.predictor.load_state_dict(checkpoint["predictor"])
    else:
        model.load_state_dict(checkpoint["model"])


def train_one_checked(
    *,
    model_type: str,
    seed: int,
    train_pairs,
    validation_pairs,
    episode_by_id,
    text_encoder,
    args,
    device: torch.device,
    output: Path,
    logger,
) -> dict[str, Any]:
    """Train one model and replay the validation-best checkpoint for final evaluation."""

    model = base.build_model(
        model_type,
        text_dim=text_encoder.output_dim,
        num_neurons=args.num_neurons,
        seed=seed,
        device=device,
    )
    parameters = base.parameters_for(model_type, model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    model_output = output / f"seed_{seed}" / model_type
    model_output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_output / "best_checkpoint.pt"
    best_validation = float("inf")
    best_metrics: dict[str, float] | None = None
    best_epoch = -1
    history: list[dict[str, float | int]] = []

    logger.log(
        "model.start",
        "모델 학습을 시작한다.",
        seed=seed,
        model_type=model_type,
        parameter_count=sum(parameter.numel() for parameter in parameters),
    )
    for epoch in range(args.epochs):
        base.set_mode(model_type, model, training=True)
        epoch_rows = []
        for pair in train_pairs:
            optimizer.zero_grad()
            left_episode = episode_by_id[pair.left_episode_id]
            right_episode = episode_by_id[pair.right_episode_id]
            left = base.predict(
                model_type=model_type,
                model=model,
                episode=left_episode,
                step_index=pair.step_index,
                text_encoder=text_encoder,
                args=args,
                device=device,
            )
            right = base.predict(
                model_type=model_type,
                model=model,
                episode=right_episode,
                step_index=pair.step_index,
                text_encoder=text_encoder,
                args=args,
                device=device,
            )
            total, metrics = base.pair_objective(
                model_type=model_type,
                left=left,
                right=right,
                context_weight=args.context_weight,
                ranking_margin=args.context_margin,
            )
            total.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            epoch_rows.append(metrics)

        train_metrics = base.aggregate(epoch_rows)
        validation_metrics = base.evaluate(
            model_type=model_type,
            model=model,
            pairs=validation_pairs,
            episode_by_id=episode_by_id,
            text_encoder=text_encoder,
            args=args,
            device=device,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        logger.log("epoch.done", "모델 epoch를 마쳤다.", seed=seed, model_type=model_type, **row)
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            best_metrics = validation_metrics
            best_epoch = epoch
            torch.save(
                {
                    "seed": seed,
                    "model_type": model_type,
                    "args": vars(args),
                    "epoch": epoch,
                    "validation": validation_metrics,
                    **base.state_dict_for(model_type, model),
                },
                checkpoint_path,
            )

    pd.DataFrame(history).to_csv(model_output / "history.csv", index=False, encoding="utf-8-sig")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_state_dict_for(model_type, model, checkpoint)
    logger.log(
        "checkpoint.loaded",
        "Validation 기준 최고 checkpoint를 다시 불러왔다.",
        seed=seed,
        model_type=model_type,
        best_epoch=best_epoch,
        best_validation_total=best_validation,
    )

    validation_real = base.evaluate(
        model_type=model_type,
        model=model,
        pairs=validation_pairs,
        episode_by_id=episode_by_id,
        text_encoder=text_encoder,
        args=args,
        device=device,
    )
    validation_shuffled = None
    if model_type.startswith("snn_") or model_type == "gru_context_contrastive":
        validation_shuffled = base.evaluate(
            model_type=model_type,
            model=model,
            pairs=validation_pairs,
            episode_by_id=episode_by_id,
            text_encoder=text_encoder,
            args=args,
            device=device,
            shuffle_history=True,
        )

    result = {
        "seed": seed,
        "model_type": model_type,
        "evaluated_checkpoint": "best_validation",
        "best_epoch": best_epoch,
        "best_validation_total": best_validation,
        "best_validation": best_metrics,
        "final_validation_real": validation_real,
        "final_validation_shuffled": validation_shuffled,
    }
    (model_output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.log("model.done", "최고 checkpoint 기준 평가를 마쳤다.", **result)
    return result


def main() -> None:
    base.train_one = train_one_checked
    base.main()


if __name__ == "__main__":
    main()
