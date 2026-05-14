from __future__ import annotations

from pathlib import Path


BENCHMARK_RESULTS_FILENAME = "benchmark_results_20260305_180830.csv"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def benchmark_dir() -> Path:
    return project_root() / "data" / "benchmark"


def default_benchmark_csv() -> Path:
    return benchmark_dir() / BENCHMARK_RESULTS_FILENAME


def default_stim_dataset_csv() -> Path:
    candidates = [
        project_root() / "outputs" / "z" / "out_z_training_extended40_calref_v1.csv",
        project_root() / "outputs" / "z" / "out_z_training_extended40.csv",
        benchmark_dir() / "dataset_for_regression.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
