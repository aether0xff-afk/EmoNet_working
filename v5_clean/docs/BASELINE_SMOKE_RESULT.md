# EmoNet v5 Clean Baseline Smoke Result

검증일: 2026-08-08 UTC / 2026-08-09 KST

## 실행 기준

- branch: `feature/v5-clean-trace-rebuild`
- verified head commit: `d73a89639c433338a01bdbe1a0581c3a414bc19a`
- GitHub Actions workflow: `v5-clean-ci`
- workflow run id: `31264249757`
- Python: 3.11
- encoder: deterministic `HashingTextEncoder`
- purpose: CI / mechanism smoke only

## Unit tests

```text
4 passed in 0.14s
```

검증 항목:

1. same seed + same sequence deterministic reproduction
2. different history changes the same final-event trace
3. `reset_episode()` before the final event removes the history difference
4. temporal-shuffle / wrong-sample controls preserve expected trace shape

## Context smoke

| Pair | history distance | reset distance |
| --- | ---: | ---: |
| praise_vs_failure | 0.1908409 | 0.0 |
| support_vs_conflict | 0.1722931 | 0.0 |
| success_vs_uncertainty | 0.2243349 | 0.0 |

Acceptance:

```text
history_changes_trace = true
reset_removes_history_difference = true
```

Topology fingerprint:

```text
828f1f67f21a9e124d23c34f60f137eae360a194d7730851d78c5615f796721b
```

## Interpretation boundary

이 결과는 감정 representation을 증명하지 않는다.

현재 확인된 것은 다음뿐이다.

> fixed recurrent baseline에서 서로 다른 과거 상태는 동일한 현재 입력의 raw trace를 다르게 만들며, 해당 차이는 recurrent state를 reset하면 사라진다.

Hashing encoder는 semantic-performance 실험용이 아니다. 따라서 위 distance 값으로 context semantics, affect semantics, EmoNet superiority를 주장하면 안 된다.

## 다음 검증

다음 단계에서는 frozen semantic embedding backend(LM Studio)를 사용해 multi-seed benchmark를 수행한다.

비교 조건:

- text only
- trace only
- text + real trace
- text + temporal-shuffled trace
- text + wrong-sample trace
- text + reset-history trace

핵심 질문은 단순히 trace가 달라지는지가 아니라, **real trace가 downstream prediction에 현재 text만으로는 얻을 수 없는 incremental information을 제공하는가**이다.
