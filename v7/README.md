# EmoNet v7

v7은 EmoNet을 다시 정리한 SNN 중심 연구 라인입니다. 목표는 감정을 라벨로 맞히는 것이 아니라, **텍스트 사건의 시간적 맥락이 SNN 내부 상태 변화와 trace에 어떻게 남는지**를 검증하는 것입니다.

## 한 줄 요약

```text
텍스트 사건 -> SNN 내부 동역학 -> trace/latent/state report -> 예측, 응답 조건화, substrate, rewiring ablation
```

v7은 아직 “감정을 느끼는 AI”가 아닙니다. 현재 단계는 **emotion-related dynamics 후보를 통제 fixture에서 검증하는 연구 프로토타입**입니다.

## 현재 구현된 것

- Adaptive sparse recurrent SNN heartbeat
- Text event schema
- frozen embedding adapter
- `EventEncoder`
- differentiable SNN training window
- `TraceEncoder`
- persistent semantic dynamics training
- context objective benchmark
- neutral trace report
- response-conditioning runner와 summarizer
- two-module thought runtime
- thought runtime budget/termination policy
- `NeuronMemoryThresholdRSNN` ablation substrate
- activity-guided rewiring ablation
- CPU/CUDA device policy
- LM Studio embedding integration

## 현재 결정 상태

| 주제 | 결정 |
| --- | --- |
| Primary substrate | `AdaptiveSparseRSNN` 유지 |
| Memory-threshold substrate | primary 승격 보류, ablation/testbed로 유지 |
| Activity-guided rewiring | final rule 아님, controlled ablation/search heuristic |
| Primary long-run fixture | `fixtures/semantic_alignment_episodes.yaml` |
| Fast CI/context fixture | `fixtures/context_dependence_episodes.yaml` |
| Response influence fixture | `fixtures/response_conditioning_cases.yaml` |
| 실험 산출물 | `runs/`는 커밋하지 않고 docs에 요약만 승격 |

## 설치

```powershell
cd v7
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

LM Studio를 쓰려면:

```powershell
pip install -e .[llm]
```

모든 optional dependency:

```powershell
pip install -e .[all]
```

## 테스트

```powershell
py -3.11 -m pytest -q
```

최근 기준:

```text
47 passed
```

빠른 context guardrail:

```powershell
python -m pytest -q `
  tests/test_context_dependence_fixture.py `
  tests/test_context_objective.py `
  tests/test_context_objective_runner.py
```

## LM Studio 확인

```powershell
python experiments/check_lmstudio.py `
  --base-url http://127.0.0.1:1234
```

사용 가능한 model id를 확인한 뒤 benchmark command에 넣습니다.

## 주요 실험

### 1. Context Objective

작고 강한 history-control fixture입니다. CI guardrail과 context-memory 회귀 확인에 씁니다.

```powershell
python experiments/run_context_objective_benchmark_checked.py `
  --fixture fixtures/context_dependence_episodes.yaml `
  --encoder lmstudio `
  --base-url http://127.0.0.1:1234 `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --seeds 7 13 21 42 100 `
  --output runs/context_objective_benchmark_lmstudio
```

### 2. Semantic Alignment / Memory-Threshold

v7의 primary long-run regression fixture입니다.

```powershell
python experiments/run_memory_threshold_parameter_sweep.py `
  --fixture fixtures/semantic_alignment_episodes.yaml `
  --encoder lmstudio `
  --base-url http://127.0.0.1:1234 `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --seeds 7 13 21 42 100 `
  --device cuda `
  --output runs/memory_threshold_parameter_sweep_lmstudio
```

요약:

```powershell
python experiments/summarize_memory_threshold_parameter_sweep.py `
  --input runs/memory_threshold_parameter_sweep_lmstudio `
  --baseline runs/trace_semantic_alignment_benchmark_lmstudio
```

### 3. Activity-Guided Rewiring

semantic-preserving topology change와 community evidence를 분리해서 평가합니다.

```powershell
python experiments/run_activity_guided_rewiring_pipeline.py `
  --fixture fixtures/semantic_alignment_episodes.yaml `
  --encoder lmstudio `
  --base-url http://127.0.0.1:1234 `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --seeds 7 13 21 42 100 `
  --null-permutations 64 `
  --device cuda `
  --output runs/activity_guided_rewiring_pipeline_lmstudio `
  --skip-baseline-auto-create
```

figure 생성:

```powershell
python experiments/visualize_activity_guided_rewiring_clusters.py `
  --input runs/activity_guided_rewiring_pipeline_lmstudio/rewired_cluster
```

### 4. Response Conditioning

neutral trace report가 응답 표면에 영향을 주는지 봅니다. substrate evidence가 아니라 response influence evidence입니다.

```powershell
python experiments/run_response_conditioning.py `
  --mode scripted `
  --fixture fixtures/response_conditioning_cases.yaml `
  --output runs/response_conditioning_scripted

python experiments/summarize_response_conditioning.py `
  --input runs/response_conditioning_scripted/runs.jsonl
```

## Fixture 정책

| Fixture | 용도 |
| --- | --- |
| `semantic_alignment_episodes.yaml` | primary long-run regression |
| `context_dependence_episodes.yaml` | fast CI/context guardrail |
| `response_conditioning_cases.yaml` | response influence exploratory |
| `semantic_training_episodes.yaml` | starter trainability |

자세한 기준은 `docs/benchmark_fixture_policy.md`를 봅니다.

## 산출물 정책

`runs/` 아래 결과, checkpoint, embedding cache, raw log, bulk CSV/PNG는 기본적으로 커밋하지 않습니다. 결과는 검토된 요약만 docs에 남깁니다.

자세한 기준은 `docs/result_artifact_policy.md`를 봅니다.

## 중요 문서

| 문서 | 내용 |
| --- | --- |
| `docs/implementation_spec_and_decision_log.md` | v7 설계와 결정 로그 |
| `docs/v7_baseline_release_note.md` | baseline release note |
| `docs/context_objective_benchmark.md` | context objective 결과 |
| `docs/semantic_dynamics_training.md` | semantic dynamics와 CUDA 기록 |
| `docs/activity_guided_rewiring_experiment_design.md` | rewiring 실험 설계와 AET-29 결과 |
| `docs/benchmark_fixture_policy.md` | fixture hierarchy |
| `docs/result_artifact_policy.md` | 산출물 보존 정책 |
| `docs/trace_meaning_and_response_evaluation.md` | trace report와 response influence 평가 |

## 해석 경계

v7이 현재 말할 수 있는 것:

- controlled fixture에서 context가 trace와 예측에 영향을 준다.
- memory-threshold substrate는 유망한 ablation 후보이다.
- activity-guided rewiring은 semantic-preserving region을 찾을 수 있다.
- neutral trace report는 응답 표면에 영향을 줄 수 있다.

v7이 아직 말할 수 없는 것:

- 감정을 느낀다.
- 내부 상태가 검증된 감정이다.
- semantic label이 ground truth emotion이다.
- neuron community가 감정 cluster다.
- rewiring rule이 최종 학습 규칙이다.
- fixture 결과가 넓은 현실 일반화를 보장한다.
