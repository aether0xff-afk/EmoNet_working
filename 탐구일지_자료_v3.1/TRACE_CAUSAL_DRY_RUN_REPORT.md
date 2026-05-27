# Trace Causal Dry Run Report

작성일: 2026-05-02

## 1. 목적

이번 dry run의 목적은 causal proof pipeline이 실제로 작동 가능한지 확인하는 것이다.

검증 대상:

1. `trace_causal_probe_set.csv`에서 조작된 trace 조건을 읽을 수 있는가
2. 각 조건별 응답을 생성할 수 있는가
3. causal judge가 full/ablation/perturbation 조건을 안정적으로 채점할 수 있는가

## 2. 생성 결과

입력:

`outputs/trace_causal_probe_set.csv`

출력:

`outputs/trace_causal_responses_dry3.csv`

Dry run 범위:

| 항목 | 값 |
|---|---:|
| Base records | 3 |
| Causal rows | 27 |
| Successful generations | 27 |
| Failed generations | 0 |

생성은 성공했다. 즉 조작된 trace payload를 이용해 full/ablation/perturbation 조건별 응답을 만드는 단계는 작동한다.

## 3. 생성 조건

각 base record는 다음 9개 조건으로 확장되었다.

| Condition | Rows |
|---|---:|
| `trace_full` | 3 |
| `ablate_target` | 3 |
| `ablate_social_orientation` | 3 |
| `ablate_control_state` | 3 |
| `ablate_action_tendency_class` | 3 |
| `perturb_target` | 3 |
| `perturb_social_orientation` | 3 |
| `perturb_control_state` | 3 |
| `perturb_action_tendency_class` | 3 |

## 4. Judge 결과

입력:

`outputs/trace_causal_responses_dry3.csv`

출력:

`outputs/trace_causal_responses_dry3_scored.csv`

Summary:

`outputs/trace_causal_responses_dry3_scored_summary.json`

현재 judge 결과:

| 항목 | 값 |
|---|---:|
| Rows submitted to judge | 27 |
| Successful judge rows | 3 |
| Failed judge rows | 24 |

실패 원인:

```text
compact response did not contain eight scores
```

즉 응답 생성이 실패한 것이 아니라, local LLM judge가 복잡한 causal scoring format을 안정적으로 따르지 못했다.

## 5. 현재 해석

이번 dry run에서 확인된 것은 두 가지다.

성공:

> causal generation pipeline은 작동한다. 같은 record에서 trace만 제거하거나 바꾼 응답을 생성할 수 있다.

미해결:

> multi-metric causal judge는 현재 local LLM에서 안정적이지 않다.

이 실패는 trace-as-emotion 가설의 실패가 아니다. 평가기 설계 문제다. 현재 judge는 한 번에 8개 점수를 JSON 또는 comma score로 요구하는데, local model이 빈 응답이나 형식 불일치 출력을 자주 낸다.

## 6. 다음 수정 방향

다음 judge는 더 단순해야 한다.

현재 방식:

```text
한 응답에 대해 8개 metric을 한 번에 점수화
```

개선 방식:

```text
축별 binary/A-B judge
```

예:

| 실험 | Judge 질문 |
|---|---|
| `trace_full` vs `ablate_target` | 어느 응답이 target/blame 방향을 더 잘 보존하는가 |
| `trace_full` vs `ablate_social_orientation` | 어느 응답이 defend/approach/withdraw 방향을 더 잘 보존하는가 |
| `trace_full` vs `ablate_action_tendency_class` | 어느 응답이 행동 경향을 더 잘 반영하는가 |
| `perturb_target` | 응답이 original target보다 new target 쪽으로 이동했는가 |

출력도 숫자 8개가 아니라 다음처럼 단순화한다.

```text
A
B
tie
```

또는:

```json
{"winner":"A","confidence":3}
```

## 7. 결론

causal proof는 한 단계 더 전진했다.

이번 단계에서 증명된 것:

> trace 조작 조건을 만들고, 그 조건별 응답을 생성하는 pipeline은 성공했다.

아직 증명되지 않은 것:

> trace 조작이 응답 방향을 통계적으로 바꾼다는 judge 기반 causal evidence.

다음 단계는 judge를 축별 A/B 평가로 단순화하는 것이다. 그 다음에야 full 24-record causal set을 안정적으로 채점할 수 있다.

