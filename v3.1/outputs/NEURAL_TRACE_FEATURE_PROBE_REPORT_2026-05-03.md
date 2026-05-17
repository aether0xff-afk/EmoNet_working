# Neural Trace Feature Probe Report

작성일: 2026-05-03

## 1. 목적

이번 작업은 LLM judge 없이 `v3.1`의 trace-as-emotion 연구를 이어가기 위한 offline representation probe다.

기존 `probe_neural_trace_geometry.py`는 세 feature만 비교했다.

- `z`
- `activation_meanmax`
- `branch_mean`

이번에는 저장된 neural trace `.npz`를 다시 읽어서 temporal/route 계열 feature를 추가했다.

## 2. 추가한 feature

새로 추가한 feature kind:

| Feature | 설명 |
|---|---|
| `activation_temporal` | activation matrix를 4개 시간 구간으로 나누고 각 구간 mean/max를 연결 |
| `branch_temporal` | branch tensor를 4개 시간 구간으로 나누고 각 구간 mean/max를 연결 |
| `route_histogram` | dominant route neuron id histogram |
| `transition_hash` | dominant route transition을 fixed 256-bin hash histogram으로 요약 |
| `active_stats` | active neuron count의 평균, 표준편차, max, 변화량 등 |
| `branch_plus_temporal` | branch mean/max + branch temporal + active stats + route/transition feature |

수정 파일:

- `scripts/probe_neural_trace_geometry.py`

생성 산출물:

- `outputs/feature_probe_2026-05-03/*.json`

## 3. 비교 대상

두 full80 trace set을 비교했다.

| Dataset | 의미 |
|---|---|
| `baseline_full80` | 기존 neural trace probe full80 |
| `best_persistent_less_inhibition_full80` | dynamics sweep에서 collapse를 제거한 후보 |

Tracked axes:

- `valence`
- `social_orientation`
- `action_tendency_class`
- `appraisal_family`
- `control_state`

## 4. 결과 요약

| Dataset | Feature | len1 ratio | density | tracked lift mean | tracked separation mean |
|---|---|---:|---:|---:|---:|
| baseline | `branch_mean` | 0.4625 | 0.4947 | +0.0675 | +0.2063 |
| baseline | `branch_temporal` | 0.4625 | 0.4947 | +0.0250 | +0.1593 |
| baseline | `branch_plus_temporal` | 0.4625 | 0.4947 | +0.0275 | +0.0513 |
| baseline | `z` | 0.4625 | 0.4947 | -0.0475 | +0.0090 |
| best full80 | `branch_mean` | 0.0000 | 0.9470 | +0.0750 | +0.2631 |
| best full80 | `branch_temporal` | 0.0000 | 0.9470 | +0.0450 | +0.2400 |
| best full80 | `branch_plus_temporal` | 0.0000 | 0.9470 | +0.0050 | +0.0705 |
| best full80 | `z` | 0.0000 | 0.9470 | -0.1200 | -0.0132 |

Full table은 각 JSON에 들어 있다.

## 5. 해석

### 5.1 `branch_mean`은 여전히 가장 강하다

새 feature를 추가해도 가장 안정적인 representation은 `branch_mean`이다.

Baseline full80:

- tracked lift mean: `+0.0675`
- tracked separation mean: `+0.2063`

Best full80:

- tracked lift mean: `+0.0750`
- tracked separation mean: `+0.2631`

따라서 현재 neural trace-as-emotion proof의 주 feature는 아직 `branch_mean`이 맞다.

### 5.2 temporal feature는 보조 신호다

`branch_temporal`은 `branch_mean`보다 약하지만 같은 방향의 신호를 보인다.

Best full80:

- `branch_mean` separation: `+0.2631`
- `branch_temporal` separation: `+0.2400`

이는 감정 geometry가 단순 평균값에만 있는 것이 아니라 시간 구간별 branch trajectory에도 일부 남아 있음을 시사한다.

다만 nearest-neighbor lift는 `branch_mean`보다 낮다.

### 5.3 route-only feature는 현재 쓸모가 약하다

`route_histogram`과 `transition_hash`는 separation이 사실상 `0.0`이고, 여러 label axis에서 majority baseline보다 낮다.

해석:

- 현재 `dominant_branch_ids`는 감정별 reusable route를 충분히 만들지 못한다.
- 또는 route id 자체보다 branch tensor의 signal/value 정보가 더 중요하다.
- `-1` route id가 많고, high-density setting에서는 거의 모든 노드가 켜지는 문제가 route feature를 약하게 만든다.

### 5.4 collapse 제거는 필요하지만 충분하지 않다

Best full80은 `len1_ratio=0.0`까지 collapse를 제거했다.

좋은 점:

- `branch_mean` separation이 `+0.2063 -> +0.2631`로 개선됐다.
- `branch_temporal` separation도 꽤 높다.

나쁜 점:

- activation density가 `0.9470`으로 너무 높다.
- `z`, activation feature, active stats는 오히려 약하다.

따라서 현재 결론은 다음이다.

```text
collapse 제거는 representation evidence를 일부 개선하지만,
현재 best setting은 과활성 때문에 emotion-specific neural geometry로 보기에는 아직 위험하다.
```

## 6. 현재 방어 가능한 주장

가능:

- neural trace extraction pipeline은 작동한다.
- branch tensor 기반 feature는 valence/appraisal/social/action 축에서 weak-to-moderate geometry signal을 보인다.
- dynamics 안정화는 branch collapse를 제거하고 group distance separation을 개선할 수 있다.

불가능:

- `z` embedding이 emotion state를 잘 보존한다고 말하기 어렵다.
- route id 자체가 감정별 reusable path를 형성한다고 말하기 어렵다.
- high-density best setting을 최종 dynamics로 채택하기에는 과활성 위험이 크다.

## 7. 다음 작업

1. fine sweep을 다시 돌린다.
   - 목표: `len1_ratio <= 0.10`
   - 목표: `mean_activation_density 0.55 ~ 0.80`
   - 목표: `branch_mean` 또는 `branch_temporal` separation이 baseline 이상

2. scoring objective를 수정한다.
   - `branch_mean`만 보지 말고 `branch_temporal`도 함께 본다.
   - activation density penalty를 더 강하게 둔다.

3. route feature는 단독 주 feature에서 내린다.
   - 후속에는 route id보다 branch tensor transition, branch slope, early/mid/late phase shift를 본다.

4. GPT API는 causal A/B judge에 사용한다.
   - representation proof는 offline metric으로 계속 진행한다.
   - generation/casual proof만 API judge를 붙인다.

## 8. 한 줄 결론

이번 feature probe는 `branch_mean` 중심 주장을 강화했다. Neural trace geometry는 branch tensor에 가장 많이 남아 있고, dynamics 안정화는 도움이 되지만 과활성을 제어해야 최종 trace-as-emotion 주장으로 갈 수 있다.
