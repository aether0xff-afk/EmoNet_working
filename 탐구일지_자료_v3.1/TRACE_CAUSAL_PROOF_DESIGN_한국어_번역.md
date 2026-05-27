# Trace Causal Proof 설계

## 1. 목적

Representation-level evidence는 normalized trace axis가 감정과 비슷한 구조를 형성한다는 점을 보여주었다. 다음 질문은 causal question이다.

> trace axis를 제거하거나 바꾸면, 응답의 감정 방향이 예측 가능한 방식으로 바뀌는가?

이 증명은 clustering보다 강하다. clustering은 비슷한 trace가 서로 가깝다는 것을 말한다. causal manipulation은 trace field가 실제로 appraisal과 response direction을 통제하는지 묻는다.

## 2. 조건

각 base record는 아홉 개의 causal condition으로 확장된다.

| Condition | Type | Meaning |
|---|---|---|
| `trace_full` | control | 모든 trace field 보존 |
| `ablate_target` | ablation | emotion target 제거 |
| `ablate_social_orientation` | ablation | social orientation 제거 |
| `ablate_control_state` | ablation | control/agency state 제거 |
| `ablate_action_tendency_class` | ablation | canonical action tendency 제거 |
| `perturb_target` | perturbation | target을 대조 값으로 변경 |
| `perturb_social_orientation` | perturbation | social orientation 변경 |
| `perturb_control_state` | perturbation | control state 변경 |
| `perturb_action_tendency_class` | perturbation | action tendency class 변경 |

## 3. 예상 효과

| 조작한 축 | 예상되는 실패 또는 변화 |
|---|---|
| `target` | blame/self/other 방향이 약해지거나 뒤집혀야 함 |
| `social_orientation` | defend/approach/withdraw tone이 이동해야 함 |
| `control_state` | helplessness, agency, planning tone이 이동해야 함 |
| `action_tendency_class` | 제안되는 행동 방향이 바뀌어야 함 |

## 4. 증거 기준

causal effect는 다음 조건을 만족할 때 지지된다.

1. Full trace가 해당 metric에서 ablated trace보다 높은 점수를 얻는다.
2. Perturbed trace가 응답 방향을 새로 조작한 값 쪽으로 이동시킨다.
3. 결과가 naturalness만으로 설명되지 않는다.
4. 효과가 서로 다른 example 사이에서만 나타나는 것이 아니라 같은 `record_id` 안에서도 나타난다.

## 5. 현재 probe set

첫 causal probe set은 24개의 base record를 사용하고, 216개의 row를 만든다.

```text
24 trace_full
96 ablation rows
96 perturbation rows
```

각 manipulated axis는 48개의 row를 가진다.

```text
24 ablations + 24 perturbations
```

## 6. 다음 실행 단계

다음 script는 v4 `episode_trace_v3`와 같은 generation backend를 사용해 모든 causal row의 응답을 생성해야 한다. 단, trace payload는 조작된 값을 사용한다.

그다음 causal judge는 다음을 평가해야 한다.

- appraisal fidelity
- target direction fit
- social orientation fit
- control state fit
- action tendency fit
- raw affect preservation
- naturalness

분석은 `record_id` 내부의 paired comparison으로 수행해야 한다.

```text
trace_full - ablated_axis
perturbed_axis direction match
```
