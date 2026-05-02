# v3.1 Trace-As-Emotion Probe Report

작성일: 2026-05-02

## 1. 목적

v3.1의 핵심 가설은 다음이다.

> trace는 감정을 설명하는 부가 정보가 아니라, 감정 상태 그 자체의 표현이다.

이번 첫 probe는 v4의 targeted records 80개를 이용해, 현재 trace field들이 실제로 emotion-state space처럼 작동하는지 확인한다.

입력:

`../v4/outputs/experiments/superiority_targeted_v1/targeted_records.csv`

출력:

`outputs/trace_emotion_probe_summary.json`

## 2. 사용한 trace field

거리 계산에 사용한 field:

- `episode_label`
- `valence`
- `arousal`
- `target`
- `control_state`
- `social_orientation`
- `preserve`
- `avoid`
- `action_tendency`

검증한 label axis:

- `episode_label`
- `valence`
- `arousal`
- `target`
- `control_state`
- `social_orientation`
- `action_tendency`

## 3. Nearest-neighbor consistency

각 record의 trace-space nearest neighbor가 같은 label axis를 공유하는지 측정했다.

| Axis | n | NN consistency | Majority baseline | Lift |
|---|---:|---:|---:|---:|
| `valence` | 80 | 0.9875 | 0.7750 | +0.2125 |
| `arousal` | 80 | 0.9375 | 0.9250 | +0.0125 |
| `target` | 80 | 0.9375 | 0.6875 | +0.2500 |
| `control_state` | 80 | 0.9750 | 0.7000 | +0.2750 |
| `social_orientation` | 80 | 0.9750 | 0.3500 | +0.6250 |
| `episode_label` | 80 | 0.0000 | 0.0125 | -0.0125 |
| `action_tendency` | 80 | 0.0000 | 0.0125 | -0.0125 |

## 4. Intra-group vs inter-group distance

같은 axis 값을 가진 pair의 평균 거리와 다른 axis 값을 가진 pair의 평균 거리를 비교했다.

| Axis | Intra pairs | Inter pairs | Mean intra distance | Mean inter distance | Separation |
|---|---:|---:|---:|---:|---:|
| `valence` | 1979 | 1181 | 0.6147 | 0.7543 | +0.1396 |
| `arousal` | 2711 | 449 | 0.6474 | 0.7843 | +0.1369 |
| `target` | 1599 | 1561 | 0.6178 | 0.7172 | +0.0994 |
| `control_state` | 1721 | 1439 | 0.6070 | 0.7385 | +0.1314 |
| `social_orientation` | 900 | 2260 | 0.5667 | 0.7068 | +0.1401 |

`episode_label`과 `action_tendency`는 현재 80개 record에서 거의 모두 고유한 자연어 값이므로 intra-pair가 없다. 이 둘은 직접 label axis로 쓰기 전에 정규화가 필요하다.

## 5. 해석

첫 probe 결과는 v3.1 방향에 긍정적이다.

현재 trace field는 이미 다음 축에서 emotion-state structure를 보인다.

- `social_orientation`: 가장 강한 nearest-neighbor lift
- `control_state`: 높은 consistency와 의미 있는 separation
- `target`: `self`, `other`, `situation`, `mixed` 축에서 baseline보다 높은 일관성
- `valence`: negative/mixed/positive 구분이 trace 거리와 잘 맞음

다만 다음 축은 아직 그대로 쓰기 어렵다.

- `episode_label`: 각 record마다 거의 고유한 서술형 label이라 군집 평가에 부적합
- `action_tendency`: 긴 자연어 설명이라 exact-match label로는 구조가 드러나지 않음

즉 현재 설계는 다음처럼 볼 수 있다.

```text
trace-as-emotion 가능성 있음:
  target
  control_state
  social_orientation
  valence

정규화가 필요한 축:
  action_tendency
  episode_label
```

## 6. 결론

v3.1을 새 폴더로 분리한 것은 타당하다. v4가 "trace를 써서 응답을 좋게 만드는 앱/평가 축"이라면, v3.1은 "trace 자체가 감정 공간인가"를 검증하는 연구 축이다.

이번 첫 probe는 trace가 단순 prompt metadata가 아니라 emotion-state representation으로 다뤄질 수 있다는 초기 증거를 준다. 특히 `social_orientation`, `control_state`, `target`은 감정 label보다 더 중요한 구조 축이 될 가능성이 있다.

## 7. 다음 작업

1. `action_tendency`를 6-10개 canonical category로 정규화한다.
   - approach
   - defend
   - confront
   - withdraw
   - repair
   - seek_support
   - inhibit
   - plan

2. `episode_label`을 free-text label이 아니라 appraisal cluster label로 재생성한다.

3. 정규화된 axis로 clustering을 다시 수행한다.

4. trace perturbation 실험을 만든다.
   - 같은 stimulus에서 `target=other`를 `target=self`로 바꿨을 때 응답 정서가 바뀌는지 본다.

5. neuron count ablation은 그 다음에 한다.
   - 먼저 trace space가 잘 정의되어야 뉴런 수 증가가 의미 있는지 판단할 수 있다.

