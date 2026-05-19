from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGET_COLUMNS = [
    "record_id",
    "text",
    "selection_bucket",
    "selection_reason",
    "episode_label",
    "valence",
    "arousal",
    "target",
    "control_state",
    "social_orientation",
    "preserve",
    "avoid",
    "action_tendency",
]


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _load_text_lookup(scored_csv: Path) -> dict[str, str]:
    if not scored_csv.exists():
        return {}
    df = pd.read_csv(scored_csv)
    if "record_id" not in df.columns or "text" not in df.columns:
        return {}
    rows = df[["record_id", "text"]].dropna(subset=["record_id"]).drop_duplicates("record_id")
    return {str(row.record_id): str(row.text) for row in rows.itertuples(index=False)}


def _load_failure_ids(paired_examples_csv: Path | None, max_failures: int) -> list[str]:
    if paired_examples_csv is None or not paired_examples_csv.exists() or max_failures <= 0:
        return []
    df = pd.read_csv(paired_examples_csv)
    required = {"condition", "record_id", "delta_mean_total"}
    if not required.issubset(df.columns):
        return []
    episode = df[df["condition"].astype(str) == "episode_trace"].copy()
    episode["delta_mean_total"] = pd.to_numeric(episode["delta_mean_total"], errors="coerce")
    episode = episode.dropna(subset=["delta_mean_total"])
    episode = episode.sort_values("delta_mean_total", ascending=True)
    return episode["record_id"].astype(str).head(max_failures).tolist()


def _candidate_rows(episode_df: pd.DataFrame, text_lookup: dict[str, str]) -> pd.DataFrame:
    df = episode_df.copy()
    if "sample_id" not in df.columns:
        raise ValueError("episode summary must contain sample_id")
    df["record_id"] = df["sample_id"].astype(str)
    if "text" not in df.columns:
        df["text"] = df["record_id"].map(text_lookup).fillna("")
    else:
        df["text"] = df["text"].fillna(df["record_id"].map(text_lookup)).fillna("")
    for column in TARGET_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def _pick_bucket(
    source: pd.DataFrame,
    *,
    used_ids: set[str],
    mask: pd.Series,
    bucket: str,
    reason: str,
    count: int,
    seed: int,
) -> pd.DataFrame:
    candidates = source[mask & ~source["record_id"].astype(str).isin(used_ids)].copy()
    if candidates.empty or count <= 0:
        return pd.DataFrame(columns=source.columns)
    if len(candidates) > count:
        candidates = candidates.sample(n=count, random_state=seed)
    candidates["selection_bucket"] = bucket
    candidates["selection_reason"] = reason
    used_ids.update(candidates["record_id"].astype(str).tolist())
    return candidates


def build_targeted_set(
    *,
    episode_summary_csv: Path,
    scored_csv: Path,
    paired_examples_csv: Path | None,
    output_csv: Path,
    manifest_json: Path,
    target_size: int,
    seed: int,
) -> dict[str, object]:
    episode_df = pd.read_csv(episode_summary_csv)
    source = _candidate_rows(episode_df, _load_text_lookup(scored_csv))
    used_ids: set[str] = set()
    buckets: list[pd.DataFrame] = []

    bucket_specs = [
        (
            "target_other",
            source["target"].astype(str).str.lower() == "other",
            "episode target is another person or group",
            30,
        ),
        (
            "social_mixed",
            source["social_orientation"].astype(str).str.lower() == "mixed",
            "mixed social orientation needs appraisal-sensitive response",
            20,
        ),
        (
            "guilt_self_blame",
            source["episode_label"].astype(str).str.contains("죄책|미안|자기비난|수치|부담", regex=True, na=False)
            | source["action_tendency"].astype(str).str.contains("사과|자기비난|부담|만회", regex=True, na=False),
            "guilt, shame, self-blame, or repair tendency",
            15,
        ),
    ]
    for offset, (bucket, mask, reason, count) in enumerate(bucket_specs):
        buckets.append(
            _pick_bucket(
                source,
                used_ids=used_ids,
                mask=mask,
                bucket=bucket,
                reason=reason,
                count=count,
                seed=seed + offset,
            )
        )

    failure_ids = _load_failure_ids(paired_examples_csv, max_failures=15)
    failure_mask = source["record_id"].astype(str).isin(failure_ids)
    buckets.append(
        _pick_bucket(
            source,
            used_ids=used_ids,
            mask=failure_mask,
            bucket="strong_failure_cases",
            reason="large current episode_trace loss; use as hard regression case",
            count=15,
            seed=seed + 10,
        )
    )

    selected = pd.concat([bucket for bucket in buckets if not bucket.empty], ignore_index=True)
    if len(selected) < target_size:
        remaining = source[~source["record_id"].astype(str).isin(used_ids)].copy()
        needed = min(int(target_size - len(selected)), len(remaining))
        if needed > 0:
            filler = remaining.sample(n=needed, random_state=seed + 99) if len(remaining) > needed else remaining
            filler["selection_bucket"] = "balanced_filler"
            filler["selection_reason"] = "fills targeted set to requested size without duplicating records"
            selected = pd.concat([selected, filler], ignore_index=True)

    selected = selected[TARGET_COLUMNS].copy()
    for column in TARGET_COLUMNS:
        selected[column] = selected[column].map(_clean)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False, encoding="utf-8-sig")

    counts = selected["selection_bucket"].value_counts().to_dict()
    manifest = {
        "episode_summary_csv": str(episode_summary_csv),
        "scored_csv": str(scored_csv),
        "paired_examples_csv": str(paired_examples_csv) if paired_examples_csv else "",
        "output_csv": str(output_csv),
        "rows": int(len(selected)),
        "target_size": int(target_size),
        "seed": int(seed),
        "bucket_counts": {str(key): int(value) for key, value in counts.items()},
        "columns": TARGET_COLUMNS,
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a targeted superiority-evaluation dataset.")
    parser.add_argument(
        "--episode-summary-csv",
        default="outputs/research/trajectory_batch_matrix120_v1_gpt54/episode_summary.csv",
    )
    parser.add_argument(
        "--scored-csv",
        default="outputs/experiments/paper_matrix_current_episode_v2_scored.csv",
    )
    parser.add_argument(
        "--paired-examples-csv",
        default="outputs/experiments/paired_superiority_episode_v2/paired_examples.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/experiments/superiority_targeted_v1/targeted_records.csv",
    )
    parser.add_argument(
        "--manifest-json",
        default="outputs/experiments/superiority_targeted_v1/targeted_records_manifest.json",
    )
    parser.add_argument("--target-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_targeted_set(
        episode_summary_csv=Path(args.episode_summary_csv),
        scored_csv=Path(args.scored_csv),
        paired_examples_csv=Path(args.paired_examples_csv) if args.paired_examples_csv else None,
        output_csv=Path(args.output_csv),
        manifest_json=Path(args.manifest_json),
        target_size=args.target_size,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
