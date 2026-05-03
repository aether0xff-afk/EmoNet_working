# Trace Causal Pair Judge Report

작성일: 2026-05-03

## 1. 목적

v3.1의 representation evidence는 neural trace가 감정 label과 정렬되는지를 본다. 하지만 trace-as-emotion 주장을 강화하려면 trace 조작이 응답 방향에도 영향을 주는지 별도로 확인해야 한다.

이번 실험은 `trace_full` 응답과 ablation/perturbation 응답을 pairwise로 비교해, judge가 예상 방향의 응답을 고르는지 확인한 causal smoke다.

## 2. 실행 설정

Provider/model:

- `anthropic`
- `claude-haiku-4-5-20251001`

중요한 실행상 발견:

- `claude-3-5-haiku-latest` 및 `claude-3-5-haiku-20241022`는 현재 환경에서 HTTP 404를 반환했다.
- `claude-haiku-4-5-20251001`은 정상 호출되었다.
- 기본 `--max-output-tokens 90`은 Claude 응답 JSON을 중간에 자를 수 있어 parse error를 만들었다.
- `--max-output-tokens 240`에서는 정상 파싱되었다.

명령:

```powershell
cd .\v3.1
$env:ANTHROPIC_API_KEY = "<local only>"
python .\scripts\judge_trace_causal_pairs.py --provider anthropic --model claude-haiku-4-5-20251001 --max-output-tokens 240 --output outputs\trace_causal_pair_judgments_claude45haiku_dry3.csv --summary outputs\trace_causal_pair_judgments_claude45haiku_dry3_summary.json
```

## 3. Smoke8 결과

| Scope | n | success | success rate |
|---|---:|---:|---:|
| Overall | 8 | 5 | 0.625 |
| Ablation preservation | 4 | 1 | 0.250 |
| Perturbation shift | 4 | 4 | 1.000 |

## 4. Full dry3 결과

| Scope | n | success | success rate |
|---|---:|---:|---:|
| Overall | 24 | 14 | 0.583333 |
| Ablation preservation | 12 | 4 | 0.333333 |
| Perturbation shift | 12 | 10 | 0.833333 |

Axis별 결과:

| Axis | n | success | success rate |
|---|---:|---:|---:|
| action_tendency_class | 6 | 5 | 0.833333 |
| control_state | 6 | 3 | 0.500000 |
| social_orientation | 6 | 4 | 0.666667 |
| target | 6 | 2 | 0.333333 |

Winner distribution:

| Winner | count |
|---|---:|
| A | 6 |
| B | 18 |

## 5. 해석

이 결과는 causal/generation evidence를 완성하지는 않지만, 중요한 방향성을 준다.

```text
Perturbation condition에서는 trace 조작 방향으로 응답이 이동한다는 신호가 강하다.
Ablation condition에서는 trace_full이 항상 원본 감정 방향을 더 잘 보존한다는 신호가 약하다.
```

따라서 v3.1에서 방어 가능한 causal 주장은 다음 정도다.

- trace perturbation은 응답의 감정 방향을 바꾸는 데 효과가 있는 것으로 보인다.
- ablation은 현재 prompt/response setup에서 원본 보존성을 안정적으로 약화시키지 못했다.
- target axis는 특히 약하므로 trace payload, ablation 방식, judge question을 재설계해야 한다.
- action tendency와 social orientation은 가장 유망한 causal axes다.

## 6. 논문 반영

논문에서는 이 결과를 strong proof가 아니라 pilot causal evidence로 둔다.

권장 표현:

```text
In a 24-pair Claude Haiku 4.5 judge pilot, perturbation pairs succeeded in 10/12 cases,
while ablation preservation succeeded in only 4/12 cases. This suggests that trace
perturbations can steer response-level affective interpretation, but ablations do not yet
provide a reliable preservation contrast.
```

## 7. 다음 작업

1. perturbation 중심의 causal experiment를 n을 늘려 재실행한다.
2. ablation은 단순 field removal이 아니라 stronger masking 또는 counterfactual neutralization으로 바꾼다.
3. target axis는 label definition과 judge question을 더 구체화한다.
4. 사람 평가 또는 두 번째 LLM judge로 inter-judge agreement를 확인한다.
