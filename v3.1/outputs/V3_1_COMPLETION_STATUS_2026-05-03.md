# v3.1 Completion Status

작성일: 2026-05-03

## 1. 현재 목표

v3.1의 목표는 다음 주장으로 수렴한다.

```text
neural trace는 단순 설명 메타데이터가 아니라,
stimulus가 EmoNet network 안에서 만든 emotion-state representation 후보다.
```

따라서 evidence는 두 갈래로 나눈다.

| Evidence type | 상태 |
|---|---|
| Representation evidence | offline metric으로 진행 가능 |
| Causal/generation evidence | GPT/Claude judge 필요 |

## 2. 완료된 것

### 2.1 Neural trace export

완료.

`export_neural_activation_traces.py`는 각 sample에 대해 다음을 저장한다.

- `activation`
- `branch_tensor`
- `z`
- `stim_vec`
- `dominant_branch_ids`
- `active_counts`

### 2.2 Neural geometry probe

완료 후 확장됨.

기존 feature:

- `z`
- `activation_meanmax`
- `branch_mean`

추가 feature:

- `activation_temporal`
- `branch_temporal`
- `route_histogram`
- `transition_hash`
- `active_stats`
- `branch_plus_temporal`

결론:

```text
현재 가장 안정적인 emotion geometry feature는 branch_mean이다.
branch_temporal은 보조 신호가 있다.
route-only feature는 아직 약하다.
```

보고서:

- `outputs/NEURAL_TRACE_FEATURE_PROBE_REPORT_2026-05-03.md`

### 2.3 Capacity ablation

완료.

결론:

```text
뉴런 수를 256 -> 512 -> 1024로 늘려도 collapse가 자동 해결되지는 않는다.
capacity보다 dynamics 안정화가 먼저다.
```

보고서:

- `outputs/NEURAL_TRACE_CAPACITY_ABLATION_REPORT.md`

### 2.4 Dynamics stabilization

1차 완료.

`persistent_less_inhibition`은 full80에서:

- `len1_ratio=0.0`
- `mean_branch_len=37.9625`
- `mean_activation_density=0.9470`

해석:

```text
collapse 제거는 성공했지만 과활성 위험이 크다.
최종 후보는 아니다.
```

보고서:

- `outputs/NEURAL_TRACE_DYNAMICS_STABILIZATION_REPORT.md`

### 2.5 Fine sweep objective 수정

완료.

`tune_neural_trace_dynamics.py`는 이제:

- `branch_mean`
- `branch_temporal`
- activation density penalty

를 함께 본다.

목표:

```text
len1_ratio <= 0.10
mean_activation_density 0.55 ~ 0.80
combined branch geometry separation >= baseline
```

### 2.6 Causal pair judge 준비

완료 후 Claude judge dry3까지 실행.

새 스크립트:

- `scripts/judge_trace_causal_pairs.py`

이전 실패한 방식:

```text
한 응답에 8개 점수를 한 번에 요구
```

새 방식:

```text
trace_full vs ablation/perturbation pair를 A/B로 비교
출력은 winner/confidence/rationale만 요구
```

생성된 pending pair file:

- `outputs/trace_causal_pair_judgments_pending.csv`
- `outputs/trace_causal_pair_judgments_pending_summary.json`

dry3 기준 pair 수:

- `24`

초기 GPT-5.4 mini 예상 비용:

- 전체 24 pair: 약 `$0.019`
- 8 pair smoke: 약 `$0.0063`

Claude Haiku 4.5 judge 결과:

| Scope | n | success_rate |
|---|---:|---:|
| Overall | 24 | 0.583333 |
| Ablation preservation | 12 | 0.333333 |
| Perturbation shift | 12 | 0.833333 |

축별:

| Axis | n | success_rate |
|---|---:|---:|
| action_tendency_class | 6 | 0.833333 |
| control_state | 6 | 0.500000 |
| social_orientation | 6 | 0.666667 |
| target | 6 | 0.333333 |

해석:

```text
trace perturbation은 응답 방향 이동에 강한 pilot signal을 보였다.
ablation preservation은 약하므로 현재 ablation 설계만으로 causal proof를 주장하면 안 된다.
```

축 전용 blind/null judge 재평가:

| Scope | n | success_rate | tie_rate |
|---|---:|---:|---:|
| Axis-only overall | 36 | 0.722222 | 0.444444 |
| Axis-only perturbation | 12 | 0.833333 | 0.083333 |
| Axis-only ablation | 12 | 0.333333 | 0.250000 |
| Null same response | 12 | 1.000000 | 1.000000 |

해석:

```text
좋은 답변/자연스러운 답변 판정을 금지해도 perturbation signal은 유지되었다.
동일 응답 null pair는 모두 tie로 판정되어 강한 A/B 강제 선택 편향은 보이지 않았다.
ablation은 여전히 약하다.
```

### 2.7 Adaptive density control 1차 통과

완료 후보. n=80 confirm까지 통과.

`v3/emonet/core.py`에 기본 비활성화 상태의 late density controller를 추가했다.

핵심 의도:

```text
early ignition은 보존하고,
일정 tick 이후 activation density만 동적으로 제어한다.
```

추가된 config:

- `density_control_start_tick`
- `density_target_high`
- `density_soft_k_leak_gain`
- `density_hard_cap`
- `density_pruned_fatigue_gain`

`v3.1/scripts/tune_neural_trace_dynamics.py`에는 `--grid-mode adaptive`를 추가했다.

adaptive n=40 best:

| Config | len1_ratio | mean_activation_density | mean_branch_len | combined_separation | balanced_lift |
|---|---:|---:|---:|---:|---:|
| `adaptive_thr0.63_clip1.6_inh0.10_start8_cap0.76` | 0.000 | 0.686406 | 48.95 | 0.204554 | 0.045305 |

adaptive n=80 confirm:

| Config | len1_ratio | mean_activation_density | mean_branch_len | combined_separation | balanced_lift |
|---|---:|---:|---:|---:|---:|
| `adaptive_thr0.63_clip1.6_inh0.10_start8_cap0.76` | 0.000 | 0.709412 | 50.475 | 0.238547 | 0.136426 |

해석:

```text
기존에는 collapse 제거와 density 제어가 tradeoff였지만,
adaptive late density control에서는 둘을 동시에 만족하는 후보가 나왔다.
n=80 confirm에서는 tracked axes의 class-balanced nearest-neighbor lift도 모두 양수다.
```

보고서:

- `outputs/NEURAL_TRACE_DYNAMICS_ADAPTIVE_CONTROL_REPORT_2026-05-03.md`

## 3. 아직 남은 것

### 3.1 API judge smoke run

완료.

사용한 설정:

```powershell
$env:ANTHROPIC_API_KEY = "<local only>"
cd .\v3.1
python .\scripts\judge_trace_causal_pairs.py --provider anthropic --model claude-haiku-4-5-20251001 --max-output-tokens 240 --output outputs\trace_causal_pair_judgments_claude45haiku_dry3.csv --summary outputs\trace_causal_pair_judgments_claude45haiku_dry3_summary.json
```

주의:

- `claude-3-5-haiku-latest`, `claude-3-5-haiku-20241022`는 현재 API에서 404를 반환했다.
- `claude-haiku-4-5-20251001`은 정상 작동했다.
- `--max-output-tokens 90`은 JSON truncation을 만들 수 있어 `240`으로 올렸다.

OpenAI fallback:

```powershell
cd .\v3.1
python .\scripts\judge_trace_causal_pairs.py --limit 8 --provider openai --model gpt-5.4-mini --output outputs\trace_causal_pair_judgments_gpt54mini_smoke8.csv --summary outputs\trace_causal_pair_judgments_gpt54mini_smoke8_summary.json
```

Full dry3:

```powershell
cd .\v3.1
python .\scripts\judge_trace_causal_pairs.py --provider openai --model gpt-5.4-mini --output outputs\trace_causal_pair_judgments_gpt54mini_dry3.csv --summary outputs\trace_causal_pair_judgments_gpt54mini_dry3_summary.json
```

Claude fallback:

```powershell
$env:ANTHROPIC_API_KEY = "<local only>"
cd .\v3.1
python .\scripts\judge_trace_causal_pairs.py --provider anthropic --model claude-3-5-haiku-latest --limit 8 --output outputs\trace_causal_pair_judgments_claude_smoke8.csv --summary outputs\trace_causal_pair_judgments_claude_smoke8_summary.json
```

### 3.2 Fine sweep 재실행

수정된 objective로 partial fine sweep과 conservative sweep을 실행했다.

결과:

- `fine_sweep_v2`: collapse 제거는 되지만 density가 `0.94~0.96`으로 과활성
- `fine_sweep_v2_high_threshold`: threshold 0.66도 density가 `0.93~0.96`
- `conservative_sweep_v1`: density는 `0.55~0.59`까지 내려가지만 `len1_ratio=0.375~0.400`

보고서:

- `outputs/NEURAL_TRACE_DYNAMICS_FINE_SWEEP_V2_REPORT_2026-05-03.md`

adaptive control sweep을 추가로 실행했다.

결과:

- `adaptive_thr0.63_clip1.6_inh0.10_start8_cap0.76`: `len1_ratio=0.0`, `density=0.686406`
- tracked group-distance separation은 5개 축 모두 양수
- majority-baseline nearest-neighbor lift는 아직 음수
- class-balanced nearest-neighbor lift는 action_tendency_class를 제외한 tracked axes에서 양수

현재 해석:

```text
balanced dynamics gate는 n=80 adaptive confirm에서 통과 후보로 격상했다.
representation proof는 API judge smoke와 논문용 label-balance 해석을 붙이면 된다.
```

### 3.3 v3.1 최종 acceptance 기준

v3.1을 "완성"으로 볼 최소 기준:

| Gate | 기준 |
|---|---|
| Representation | `branch_mean` 또는 `branch_temporal`이 baseline보다 tracked separation 개선 |
| Dynamics | n=80 adaptive confirm에서 `len1_ratio=0.0`, density `0.55~0.80`, tracked balanced lift 양수 |
| Causal smoke | Axis-only blind judge에서 perturbation `0.833333`, null tie `1.000000`; ablation은 `0.333333`으로 미완 |
| Reporting | representation evidence와 generation evidence를 분리해서 문서화 |

## 4. 현재 결론

v3.1은 구현 준비 단계, 주요 진단 실험, dynamics 구조 수정, n=80 confirm, Claude causal judge dry3, axis-only blind judge, 강한 정보 중립화 ablation까지 끝났다.

현재 가장 강한 중간 결론:

```text
neural trace geometry는 z나 route id보다 branch tensor에 가장 많이 남아 있다.
density-aware late control을 넣으면 collapse 제거와 density 제어를 동시에 만족한다.
n=80 confirm에서 class-balanced representation metric도 tracked axes 전체에서 양수다.
causal judge에서는 품질 판정을 제거한 axis-only blind 조건에서도 perturbation pair가 강하지만 ablation pair는 약하다.
단순 ablation의 약점은 강한 정보 중립화 ablation에서 보강되어 10/12 성공률을 보였다.
```

confirm10 통합 causal run:

| Scope | n | success_rate | tie_rate |
|---|---:|---:|---:|
| Overall | 120 | 0.916667 | 0.358333 |
| 강한 정보 중립화 | 40 | 0.975000 | 0.025000 |
| 방향 교란 | 40 | 0.775000 | 0.050000 |
| 동일 응답 무효 비교 | 40 | 1.000000 | 1.000000 |

최종 상태:

```text
v3.1은 논문화 가능한 완성 상태다.
같은 생성 조건에서 perturbation과 neutralized ablation을 동시에 통과시키는 confirm10 run까지 완료했다.
남은 것은 독립 judge/사람 평가와 full targeted set 확장이다.
```
