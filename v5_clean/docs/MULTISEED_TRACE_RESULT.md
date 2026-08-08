# EmoNet v5 Clean Multi-seed Trace Benchmark

검증일: 2026-08-09 KST

## 실행

- branch: `feature/v5-clean-trace-rebuild`
- verified head commit: `223c73cef76e9f609df4a50718e032ef4db5d47b`
- GitHub Actions workflow: `v5-clean-ci`
- workflow run id: `31264433732`
- job id: `93120186915`
- Python: 3.11
- encoder: deterministic `HashingTextEncoder`
- seeds: `7, 13, 21, 42, 100`
- context fixtures: 12
- total context probes: 60

이 실험은 recurrent trace mechanism의 multi-seed sanity benchmark다. semantic representation 또는 affect representation 성능 실험이 아니다.

## 결과

### History distance

- mean: `0.2552781653`
- std: `0.0841495417`
- min: `0.0786530897`
- max: `0.4438913465`
- positive fraction: `1.0`

즉 5개 seed × 12개 fixture의 60개 비교 전부에서 서로 다른 history가 동일 final event의 trace를 바꿨다.

### Reset distance

- mean: `0.0`
- max: `0.0`
- zero fraction: `1.0`

동일 final event 직전에 `reset_episode()`를 수행하면 60/60 비교에서 history 차이가 완전히 사라졌다.

### Controls

- mean real vs temporal-shuffled trace distance: `0.2824241598`
- mean real vs wrong-sample trace distance: `0.5787148972`

### Acceptance

```text
all_history_pairs_change_trace = true
all_reset_pairs_remove_difference = true
controls_are_nonidentical = true
```

GitHub Actions 결과:

```text
Unit tests: 4 passed
Context smoke: success
Multi-seed trace benchmark: success
Artifact upload: success
```

Artifact name:

```text
v5-clean-multiseed-trace-benchmark
```

Artifact ID: `9023724515`

파일:

- `context_probe_rows.csv`
- `control_rows.csv`
- `summary.json`

## 해석 경계

현재 결과가 직접 지지하는 것은 다음뿐이다.

> fixed recurrent baseline은 seed가 달라져도 과거 입력의 영향을 동일 현재 입력의 raw trace에 남기며, 그 차이는 recurrent state reset으로 제거된다.

현재 결과만으로 다음은 주장하지 않는다.

- trace가 semantic context를 올바르게 표현한다.
- trace가 affect/emotion representation이다.
- trace가 text-only representation보다 유용하다.
- temporal order가 downstream task에 유용하다.

`HashingTextEncoder`는 의미 성능용 encoder가 아니므로 다음 단계는 semantic encoder 또는 task-controlled synthetic representation을 사용해 trace의 incremental information을 직접 측정해야 한다.
