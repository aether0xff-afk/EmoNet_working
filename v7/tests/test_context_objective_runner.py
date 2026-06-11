from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_checked_runner():
    script = Path("experiments/run_context_objective_benchmark_checked.py").resolve()
    spec = importlib.util.spec_from_file_location("checked_context_runner", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script.parent))
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(script.parent))
    return module


def test_checked_context_runner_writes_hash_smoke_outputs(tmp_path, monkeypatch) -> None:
    runner = load_checked_runner()
    output = tmp_path / "context_checked"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_context_objective_benchmark_checked.py",
            "--encoder",
            "hash",
            "--epochs",
            "1",
            "--seeds",
            "7",
            "--num-neurons",
            "8",
            "--event-ticks",
            "2",
            "--stimulation-ticks",
            "1",
            "--output",
            str(output),
            "--quiet",
        ],
    )

    runner.main()

    assert (output / "metadata.json").exists()
    assert (output / "by_seed_model.csv").exists()
    assert (output / "summary_by_model.csv").exists()
    assert (output / "seed_7" / "snn_context_contrastive" / "summary.json").exists()
