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

완료.

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

현재 dry3 기준 pair 수:

- `24`

GPT-5.4 mini 예상 비용:

- 전체 24 pair: 약 `$0.019`
- 8 pair smoke: 약 `$0.0063`

## 3. 아직 남은 것

### 3.1 API judge smoke run

환경변수 필요:

```powershell
$env:OPENAI_API_KEY = "<local only>"
```

Smoke:

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

### 3.3 v3.1 최종 acceptance 기준

v3.1을 "완성"으로 볼 최소 기준:

| Gate | 기준 |
|---|---|
| Representation | `branch_mean` 또는 `branch_temporal`이 baseline보다 tracked separation 개선 |
| Dynamics | 현재 미통과. collapse/density tradeoff가 확인됨 |
| Causal smoke | pairwise judge success rate가 chance보다 높음 |
| Reporting | representation evidence와 generation evidence를 분리해서 문서화 |

## 4. 현재 결론

v3.1은 구현 준비 단계와 주요 진단 실험은 끝났다. 남은 것은 API judge smoke와 dynamics 구조 수정이다.

현재 가장 강한 중간 결론:

```text
neural trace geometry는 z나 route id보다 branch tensor에 가장 많이 남아 있다.
dynamics 조정은 collapse를 제거할 수 있지만, 현재 구조에서는 과활성 제어와 collapse 제거가 tradeoff로 갈라진다.
```
