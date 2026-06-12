# Benchmark Fixture Policy

Status date: 2026-06-12

This document fixes the current v7 fixture hierarchy for regression and
exploratory work. It separates the primary long-run benchmark fixture from fast
CI guardrails and response-surface experiments.

## Decision

Primary long-run regression fixture:

```text
fixtures/semantic_alignment_episodes.yaml
```

Fast CI/context guardrail fixture:

```text
fixtures/context_dependence_episodes.yaml
```

Secondary exploratory response-influence fixture:

```text
fixtures/response_conditioning_cases.yaml
```

Starter trainability fixture, not a regression target:

```text
fixtures/semantic_training_episodes.yaml
```

## Rationale

`semantic_alignment_episodes.yaml` is the primary regression fixture because it
is the only current fixture that supports the full next substrate decision path:

- 40 episodes: 24 train and 16 validation
- 20 contrast pairs across train and validation
- 40 evaluation-only semantic labels
- four coarse axes: `valence`, `arousal`, `certainty`, and `social_distance`
- existing support in trace-alignment, memory-threshold, history-reconstruction,
  and activity-guided rewiring runners

Those labels are evaluation probes only. They must not be used to train the SNN,
rewire topology, discover communities, or claim ground-truth emotions.

`context_dependence_episodes.yaml` remains important, but it is too small and
too narrow to be the primary fixture for substrate decisions:

- 8 episodes: 4 train and 4 validation
- 4 contrast pairs
- identical-current-text controls with different prior histories
- no semantic labels

It is the correct fast guardrail for checking whether history matters and
whether reset/shuffle controls remain wired correctly.

`response_conditioning_cases.yaml` is a response-surface fixture. It tests
whether neutral trace reports change generated replies under direct, report,
masked-report, and shuffled-report conditions. It is not a substrate regression
fixture until its reports are generated from validated trace runs rather than
small fixed cases.

## CI Smoke

CI should stay small and deterministic. Use hash encoders and fixture integrity
tests. These checks verify wiring and guard against accidental fixture drift;
they are not semantic evidence.

```powershell
python -m pytest -q `
  tests/test_context_dependence_fixture.py `
  tests/test_context_objective.py `
  tests/test_context_objective_runner.py
```

Optional local smoke for the context objective:

```powershell
python experiments/run_context_objective_benchmark_checked.py `
  --fixture fixtures/context_dependence_episodes.yaml `
  --encoder hash `
  --epochs 1 `
  --seeds 7 `
  --output runs/ci_context_objective_hash_smoke `
  --quiet
```

## Long-Run Regression

Long-run evidence should use `semantic_alignment_episodes.yaml` with LM Studio
embeddings and multi-seed runs. Generated outputs stay under `runs/` and are
promoted into docs only through reviewed summaries, following
`docs/result_artifact_policy.md`.

Canonical memory-threshold sweep:

```powershell
python experiments/run_memory_threshold_parameter_sweep.py `
  --fixture fixtures/semantic_alignment_episodes.yaml `
  --encoder lmstudio `
  --base-url <lmstudio-base-url> `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --seeds 7 13 21 42 100 `
  --device cuda `
  --output runs/memory_threshold_parameter_sweep_lmstudio

python experiments/summarize_memory_threshold_parameter_sweep.py `
  --input runs/memory_threshold_parameter_sweep_lmstudio `
  --baseline runs/trace_semantic_alignment_benchmark_lmstudio
```

Canonical activity-guided rewiring pipeline:

```powershell
python experiments/run_activity_guided_rewiring_pipeline.py `
  --fixture fixtures/semantic_alignment_episodes.yaml `
  --encoder lmstudio `
  --base-url <lmstudio-base-url> `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --seeds 7 13 21 42 100 `
  --null-permutations 64 `
  --device cuda `
  --output runs/activity_guided_rewiring_pipeline_lmstudio `
  --skip-baseline-auto-create
```

## Secondary Exploratory Work

Use `response_conditioning_cases.yaml` for report-conditioned response behavior
only:

```powershell
python experiments/run_response_conditioning.py `
  --mode scripted `
  --fixture fixtures/response_conditioning_cases.yaml `
  --output runs/response_conditioning_scripted

python experiments/summarize_response_conditioning.py `
  --input runs/response_conditioning_scripted/runs.jsonl
```

These results may show that neutral reports affect response surfaces. They do
not establish that the SNN state is emotional, that the model feels, or that
response text is a ground-truth readout of internal affect.

## Promotion Criteria

Do not promote a new fixture to primary regression until it has:

- train and validation splits
- explicit contrast pairs
- reset/shuffle or equivalent negative controls
- deterministic CI integrity tests
- long-run LM Studio command coverage
- clear claim boundaries
- at least one multi-seed evidence run summarized in docs

The next likely upgrade is a larger emotion-trajectory fixture, but it should
enter as exploratory until it satisfies those criteria.
