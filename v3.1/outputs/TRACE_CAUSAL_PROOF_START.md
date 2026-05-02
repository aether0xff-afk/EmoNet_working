# Trace Causal Proof Start

작성일: 2026-05-02

## 1. 목적

이전 단계에서는 trace space가 emotion-state-like structure를 보인다는 representation-level evidence를 얻었다.

이번 단계부터는 causal proof를 시작한다.

핵심 질문:

> trace 축을 제거하거나 바꾸면 응답의 감정 방향이 예측 가능하게 무너지거나 바뀌는가?

이 질문이 중요한 이유는 단순 군집화보다 강한 증거이기 때문이다. 군집화는 "비슷한 trace가 가까이 있다"를 보여준다. causal proof는 "trace가 실제로 감정 해석과 응답 방향을 조절한다"를 보여준다.

## 2. 생성한 causal probe set

입력:

`outputs/targeted_records_trace_normalized.csv`

출력:

`outputs/trace_causal_probe_set.csv`

Manifest:

`outputs/trace_causal_probe_manifest.json`

생성 결과:

| 항목 | 값 |
|---|---:|
| Base records | 24 |
| Total rows | 216 |
| Control rows | 24 |
| Ablation rows | 96 |
| Perturbation rows | 96 |

## 3. 조건별 구성

| Causal condition | Count |
|---|---:|
| `trace_full` | 24 |
| `ablate_target` | 24 |
| `ablate_social_orientation` | 24 |
| `ablate_control_state` | 24 |
| `ablate_action_tendency_class` | 24 |
| `perturb_target` | 24 |
| `perturb_social_orientation` | 24 |
| `perturb_control_state` | 24 |
| `perturb_action_tendency_class` | 24 |

축별로 균형 있게 구성되었다.

| Manipulated axis | Count |
|---|---:|
| `target` | 48 |
| `social_orientation` | 48 |
| `control_state` | 48 |
| `action_tendency_class` | 48 |
| `none` | 24 |

## 4. 실험 논리

각 base record는 다음 9개 조건으로 확장된다.

```text
trace_full
ablate_target
ablate_social_orientation
ablate_control_state
ablate_action_tendency_class
perturb_target
perturb_social_orientation
perturb_control_state
perturb_action_tendency_class
```

예상되는 효과:

| 축 | 제거 시 예상 | 교란 시 예상 |
|---|---|---|
| `target` | blame 방향이 흐려짐 | self/other/situation 방향이 바뀜 |
| `social_orientation` | defend/approach/withdraw 톤이 약해짐 | 사회적 접근 방향이 바뀜 |
| `control_state` | 무력감/통제감/계획성이 흐려짐 | agency 표현이 바뀜 |
| `action_tendency_class` | 응답의 행동 방향이 일반화됨 | 행동 제안 또는 충동 방향이 바뀜 |

## 5. 이 단계의 의미

현재까지는 causal proof의 실험 재료를 만든 단계다. 아직 응답 생성과 judge 평가까지 수행한 것은 아니다.

하지만 중요한 진전은 있다.

> 같은 record 안에서 trace만 조작하는 paired causal experiment가 가능해졌다.

이제부터는 외부 예시 차이나 문장 난이도 차이가 아니라, 같은 입력에서 trace 조작만으로 응답 변화가 생기는지를 볼 수 있다.

## 6. 다음 단계

다음 구현은 두 가지다.

1. Causal generation
   - `trace_causal_probe_set.csv`의 각 row로 응답을 생성한다.
   - full/ablated/perturbed trace를 prompt payload에 반영한다.

2. Causal judge
   - full trace가 ablated trace보다 matching metric에서 높은지 본다.
   - perturbed trace가 새 target/action/social/control 방향으로 응답을 이동시키는지 본다.

성공 기준:

| 비교 | 성공 신호 |
|---|---|
| `trace_full` vs `ablate_target` | target direction fit 하락 |
| `trace_full` vs `ablate_social_orientation` | social orientation fit 하락 |
| `trace_full` vs `ablate_control_state` | control state fit 하락 |
| `trace_full` vs `ablate_action_tendency_class` | action tendency fit 하락 |
| perturbation | 응답 방향이 new_value 쪽으로 이동 |

## 7. 한 줄 결론

causal proof가 시작되었다. 이제 trace는 단순히 "비슷한 것끼리 모이는가"가 아니라, "바꾸면 감정 응답도 바뀌는가"로 검증된다.

