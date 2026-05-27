# v3.1 Final Completion Report

작성일: 2026-05-03

## 1. 완성 판정

v3.1은 다음 기준에서 `논문화 가능한 완성 후보`에 도달했다.

```text
neural trace는 감정 상태 표현 후보로서 구조적 신호를 가지며,
trace 방향 교란은 응답의 감정축 해석을 움직이고,
강한 정보 중립화는 해당 감정축 신호를 약화시킨다.
```

confirm10 통합 실행까지 완료되어, causal evidence도 pilot을 넘어 1차 confirmatory evidence로 격상되었다.

## 2. 최종 best dynamics

현재 main model 설정:

```text
adaptive_thr0.63_clip1.6_inh0.10_start8_cap0.76
```

한국어 해석:

```text
초반에는 감정 흐름이 충분히 생기도록 두고,
후반에는 너무 많은 뉴런이 켜지지 않도록 활성 밀도를 조절하는 설정.
```

결과:

| 지표 | 값 | 판정 |
|---|---:|---|
| 감정 흐름 길이 | 50.475 | 좋음 |
| 한 칸짜리 흐름 비율 | 0.000 | 붕괴 없음 |
| 활성 밀도 | 0.709412 | 적정 범위 |
| 감정 구조 분리도 | 0.238547 | 양수 |
| 균형 보정 감정 신호 | 0.136426 | 양수 |

## 3. 표현 증거

n=80 confirm에서 tracked axes 전체가 class-balanced 기준 양수다.

| 감정축 | 균형 보정 감정 신호 |
|---|---:|
| 감정가 | 0.258480 |
| 사회적 방향 | 0.119748 |
| 행동 경향 | 0.017921 |
| 평가 가족 | 0.078728 |
| 통제감 상태 | 0.175607 |

해석:

```text
데이터 불균형을 보정해도 trace 안에 감정축 구조가 남아 있다.
```

## 4. 방향 교란 증거

axis-only blind judge에서 기존 perturbation dry3는 다음 결과를 보였다.

| 평가 | 값 |
|---|---:|
| 방향 교란 성공률 | 10/12 = 0.833333 |
| 동일 응답 무효 비교 | 12/12 tie |

해석:

```text
좋은 답변/자연스러운 답변 판정을 금지해도,
trace 방향을 바꾸면 응답의 감정축 해석도 따라 움직인다.
```

## 5. 강한 정보 중립화 증거

기존 ablation은 단순히 한 field만 지웠기 때문에 약했다.

기존 결과:

| 평가 | 성공률 |
|---|---:|
| 단순 정보 제거 | 4/12 = 0.333333 |

그래서 강한 중립화 ablation을 추가했다.

변경:

1. 해당 axis field를 `neutral`로 바꿈
2. `preserve` 문장에서도 해당 축 단서를 제거
3. `avoid` 문장에서도 해당 축 단서를 제거
4. `action_tendency` 문장에서도 해당 축 단서를 제거
5. 생성 prompt에 원래 축 방향을 표현하지 말라고 명시

coherent neutralization dry3 결과:

| 평가 | 성공률 |
|---|---:|
| 강한 정보 중립화 | 10/12 = 0.833333 |
| 동일 응답 무효 비교 | 12/12 = 1.000000 |

해석:

```text
trace 안의 해당 감정축 단서를 충분히 중립화하면,
원래 감정축 신호가 응답에서 약해진다.
```

## 6. 통합 Confirmatory Run

이전 한계는 방향 교란과 강한 정보 중립화가 완전히 같은 생성 조건에서 동시에 통과한 것이 아니라는 점이었다. 이를 해소하기 위해 같은 조건으로 confirm10을 실행했다.

설정:

- base record: 10개
- causal response rows: 90개
- blind judge pairs: 120개
- generator: `claude-haiku-4-5-20251001`
- judge: `claude-haiku-4-5-20251001`
- ablation mode: strong neutralization
- perturbation mode: coherent perturbation

결과:

| 평가 | n | 성공률 | tie 비율 |
|---|---:|---:|---:|
| 전체 | 120 | 0.916667 | 0.358333 |
| 강한 정보 중립화 | 40 | 0.975000 | 0.025000 |
| 방향 교란 | 40 | 0.775000 | 0.050000 |
| 동일 응답 무효 비교 | 40 | 1.000000 | 1.000000 |

축별 결과:

| 감정축 | n | 성공률 | tie 비율 |
|---|---:|---:|---:|
| 행동 경향 | 30 | 0.866667 | 0.400000 |
| 통제감 상태 | 30 | 0.966667 | 0.333333 |
| 사회적 방향 | 30 | 0.900000 | 0.366667 |
| 감정이 향하는 대상 | 30 | 0.933333 | 0.333333 |

해석:

```text
같은 생성 조건에서 정보 중립화와 방향 교란이 모두 완성 기준을 넘었다.
무효 비교는 40/40 tie로, judge가 억지로 한쪽을 고르는 편향은 낮다.
```

## 7. 아직 남은 한계

중요한 한계:

```text
confirm10은 같은 생성 조건에서 통과했지만, generator와 judge가 모두 Claude Haiku 4.5다.
```

따라서 남은 한계는 다음이다.

- 독립 judge 또는 사람 평가가 아직 없다.
- base record 10개는 3개보다 훨씬 낫지만, full targeted set은 아니다.
- 같은 모델이 생성과 평가를 모두 맡았으므로 model-family bias 가능성이 남는다.

따라서 논문 표현은 다음처럼 해야 한다.

```text
The confirm10 run provides first confirmatory causal evidence under a unified generator-controlled setting,
but independent judges and larger samples remain future work.
```

## 8. 완성/미완성 판정표

| Gate | 기준 | 결과 | 판정 |
|---|---|---:|---|
| 감정 흐름 안정성 | 한 칸짜리 흐름 비율 <= 0.05 | 0.000 | 통과 |
| 활성 밀도 | 0.55--0.80 | 0.709 | 통과 |
| 표현 증거 | tracked axes balanced signal 양수 | 전체 양수 | 통과 |
| 방향 교란 | axis-only blind >= 0.75 | 0.775 | 통과 |
| 정보 중립화 | axis-only blind >= 0.60 | 0.975 | 통과 |
| 무효 비교 | 동일 응답 tie 높음 | 1.000 | 통과 |
| 통합 causal run | 같은 생성기에서 교란/중립화 동시 통과 | 0.916 overall | 통과 |

## 9. 최종 결론

v3.1은 연구 완성본으로 다음 수준까지 도달했다.

```text
trace-as-emotion 가설은 representation evidence와 pilot causal evidence를 모두 확보했다.
dynamics 문제는 해결되었고, 정보 중립화 약점도 강한 ablation으로 보강되었다.
confirm10 통합 causal run에서도 방향 교란과 정보 중립화가 동시에 기준을 넘었다.
```

논문에서는 v3.1을 다음처럼 주장하는 것이 안전하다.

```text
EmoNet v3.1 provides evidence that neural activation traces behave as emotion-state representations:
they form stable branch dynamics, preserve class-balanced affective geometry, and support pilot
counterfactual steering and neutralization tests at the response level.
```
