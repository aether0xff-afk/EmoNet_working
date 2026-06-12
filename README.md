# EmoNet

EmoNet은 감정을 단일 라벨로 맞히는 모델이 아니라, **시간에 따라 변하는 내부 상태와 trace가 감정 관련 맥락을 어떻게 담는지** 검증하는 연구 프로토타입입니다.

현재 중심 작업은 `v7`입니다. v7은 텍스트 사건을 SNN 내부 동역학으로 변환하고, 그 상태가 다음 사건 예측, context 유지, response conditioning, memory-threshold substrate, activity-guided rewiring에 어떤 영향을 주는지 실험합니다.

## 지금 이 프로젝트를 뭐라고 부를 수 있나

현재 가장 정확한 표현은 다음입니다.

```text
SNN 기반 affective dynamics 연구 프로토타입
```

조금 더 풀면:

```text
텍스트 사건의 시간적 맥락을 SNN 내부 상태 변화로 보존하고,
그 변화가 예측, 응답 조건화, memory substrate, rewiring ablation에
검증 가능한 영향을 주는지 확인하는 실험 시스템
```

아직 **감정을 느끼는 AI**라고 말할 단계는 아닙니다. 지금까지의 결과는 “감정 관련 내부 상태 후보”와 “맥락 의존적 trace dynamics”에 대한 통제 실험 근거입니다.

## 현재 상태

최근 v7 후속 작업에서 `AET-19`부터 `AET-30`까지 완료했습니다.

- v7 baseline 정리
- two-module thought runtime 구현
- thought runtime 비용/라운드 제한 구현
- neutral trace report 기반 response-conditioning runner 구현
- response influence summarizer 구현
- LM Studio multi-seed context objective 실행
- SSH GPU 호스트에서 CUDA 장기 실행 및 CPU 비교
- 실험 산출물 보존 정책 정리
- memory-threshold substrate 승격 여부 결정
- activity-guided rewiring 본실험 실행
- 다음 benchmark fixture hierarchy 선정

현재 `main`은 최신 작업이 푸시된 상태이며, 마지막 검증 기준 테스트는 `47 passed`입니다.

## 핵심 결론

### 1. Context objective

LM Studio embedding 기반 multi-seed 실험에서 contrastive SNN은 prior context를 사용하는 신호를 보였습니다. 다만 GRU도 경쟁력이 있었기 때문에, 이것만으로 SNN 고유 우위나 감정 의미를 주장하지 않습니다.

### 2. CUDA 실행

원격 GPU 호스트 `DESKTOP-MMLRCFK`의 RTX 4090에서 strict CUDA 실행을 확인했습니다. 작은 fixture에서는 CPU가 더 빠른 결과도 있었으므로, CUDA 결과는 성능 우위가 아니라 **device path 검증**으로 해석합니다.

### 3. Memory-threshold substrate

`NeuronMemoryThresholdRSNN`은 기존 contrastive SNN과 GRU보다 일부 semantic/context 지표가 좋았습니다. 하지만 community evidence가 아직 약해서 primary substrate로 승격하지 않고, ablation substrate 및 rewiring testbed로 보류했습니다.

### 4. Activity-guided rewiring

semantic-preserving rewiring region은 찾았습니다. 그러나 rewired adjacency community evidence는 확립되지 않았습니다. 따라서 현재 rewiring rule은 final rule이 아니라 controlled ablation/search heuristic입니다.

### 5. Benchmark fixture

현재 fixture hierarchy는 다음으로 고정했습니다.

| 역할 | Fixture |
| --- | --- |
| Primary long-run regression | `v7/fixtures/semantic_alignment_episodes.yaml` |
| Fast CI/context guardrail | `v7/fixtures/context_dependence_episodes.yaml` |
| Secondary response influence | `v7/fixtures/response_conditioning_cases.yaml` |
| Starter trainability only | `v7/fixtures/semantic_training_episodes.yaml` |

## 빠른 시작

```powershell
git clone <repository-url>
cd EmoNet_working
cd v7

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

LM Studio나 response-conditioning 실험까지 실행하려면:

```powershell
pip install -e .[llm]
```

모든 optional dependency를 설치하려면:

```powershell
pip install -e .[all]
```

## 테스트

v7 전체 테스트:

```powershell
cd v7
py -3.11 -m pytest -q
```

최근 기준:

```text
47 passed
```

빠른 context guardrail만 확인:

```powershell
python -m pytest -q `
  tests/test_context_dependence_fixture.py `
  tests/test_context_objective.py `
  tests/test_context_objective_runner.py
```

## 주요 실행 예시

### LM Studio 연결 확인

```powershell
python experiments/check_lmstudio.py `
  --base-url http://127.0.0.1:1234
```

### Context objective benchmark

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

### Memory-threshold long-run regression

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

### Activity-guided rewiring pipeline

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

## 중요한 문서

| 문서 | 내용 |
| --- | --- |
| `v7/docs/implementation_spec_and_decision_log.md` | v7 전체 설계, claim boundary, 주요 결정 로그 |
| `v7/docs/v7_baseline_release_note.md` | v7 baseline release note |
| `v7/docs/context_objective_benchmark.md` | context objective 실험과 AET-25 결과 |
| `v7/docs/semantic_dynamics_training.md` | semantic dynamics training, CUDA 기록 |
| `v7/docs/activity_guided_rewiring_experiment_design.md` | rewiring 실험 설계와 AET-29 결과 |
| `v7/docs/benchmark_fixture_policy.md` | primary/secondary fixture hierarchy |
| `v7/docs/result_artifact_policy.md` | 실험 산출물 보존/커밋 정책 |
| `v7/docs/trace_meaning_and_response_evaluation.md` | trace report와 response influence 평가 경계 |

## 저장소 구조

```text
.
  v7/                  현재 중심 연구 라인
  v6/                  Ruca/Rookie 자율 캐릭터 런타임
  v5/                  EmoNet trace -> character-chat MVP
  v4/                  논문, 평가, trajectory 분석, local GUI
  v3.1/                trace-as-emotion representation 연구
  v3/, v2/, v1/        이전 실험 라인
  src/                 Minecraft RL Agent MVP
  docs/                루트 문서
  Dataset/             공유 데이터셋
  blueprints/          설계 노트
  outputs/, output/    생성 산출물
```

## 산출물 정책

`runs/`, checkpoint, embedding cache, raw log, bulk CSV/PNG는 기본적으로 커밋하지 않습니다. 실험 결과는 다음만 문서로 승격합니다.

- 실행 command
- code commit hash
- seed와 핵심 hyperparameter
- backend/model id
- 핵심 metric
- 해석 경계
- full output 위치

자세한 기준은 `v7/docs/result_artifact_policy.md`를 봅니다.

## Claim Boundary

현재 EmoNet v7이 말할 수 있는 것:

- 텍스트 사건을 SNN 내부 상태 변화로 변환할 수 있다.
- 일부 fixture에서 prior context가 trace와 예측에 영향을 준다.
- memory-threshold substrate는 의미 있는 ablation 후보이다.
- activity-guided rewiring은 semantic-preserving topology change를 찾을 수 있지만 community evidence는 아직 부족하다.
- neutral trace report는 response surface에 영향을 줄 수 있다.

현재 말하면 안 되는 것:

- 시스템이 감정을 느낀다.
- 내부 상태가 검증된 감정이다.
- neuron cluster가 감정 cluster다.
- rewiring rule이 최종 생물학적 규칙이다.
- fixture 결과가 넓은 현실 일반화를 보장한다.

## License

MIT License. 자세한 내용은 `LICENSE`를 참고하세요.
