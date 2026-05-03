# v3.1 Performance Comparison

작성일: 2026-05-03

## 1. 비교 기준

v3.1의 목적은 일반적인 LLM 응답 품질이 아니라 `neural trace가 emotion-state representation 후보인가`를 보는 것이다. 따라서 성능 비교는 다음 네 축으로 나눈다.

| Metric | 의미 | 좋은 방향 |
|---|---|---|
| `len1_ratio` | branch collapse 비율 | 낮을수록 좋음 |
| `mean_activation_density` | 전체 뉴런 중 활성 비율 | 0.55--0.80 target |
| `combined_separation_mean` | trace feature가 label group을 분리하는 정도 | 높을수록 좋음 |
| `combined_balanced_lift_mean` | class imbalance를 보정한 nearest-neighbor lift | 양수이고 높을수록 좋음 |

## 2. 주요 후보 비교

| Group | Config | n | branch len | len1 | density | sep | balanced lift | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | `baseline` | 40 | 10.275 | 0.500 | 0.448 | - | - | collapse 심함, density 낮음 |
| Persistent | `persistent_less_inhibition` | 80 | 37.963 | 0.000 | 0.947 | - | - | collapse 제거, 과활성 |
| Conservative | `thr0.74_clip1.2_inh0.20_high_fatigue` | 40 | 19.825 | 0.400 | 0.550 | 0.156 | - | density는 맞지만 collapse 남음 |
| Adaptive sweep | `adaptive_thr0.63_clip1.6_inh0.10_start8_cap0.76` | 40 | 48.950 | 0.000 | 0.686 | 0.205 | 0.045 | 1차 통과 |
| Adaptive confirm | `adaptive_thr0.63_clip1.6_inh0.10_start8_cap0.76` | 80 | 50.475 | 0.000 | 0.709 | 0.239 | 0.136 | 현재 best |

## 3. 개선폭

Best adaptive confirm은 baseline 대비:

| Metric | Baseline | Best adaptive n=80 | 변화 |
|---|---:|---:|---:|
| branch len | 10.275 | 50.475 | +391.2% |
| len1_ratio | 0.500 | 0.000 | collapse 제거 |
| density | 0.448 | 0.709 | target range 진입 |

Best adaptive confirm은 persistent 계열 대비:

| Metric | Persistent | Best adaptive n=80 | 해석 |
|---|---:|---:|---|
| branch len | 37.963 | 50.475 | 더 긴 trace |
| len1_ratio | 0.000 | 0.000 | 둘 다 collapse 제거 |
| density | 0.947 | 0.709 | adaptive만 과활성 회피 |

Best adaptive confirm은 conservative 계열 대비:

| Metric | Conservative | Best adaptive n=80 | 해석 |
|---|---:|---:|---|
| branch len | 19.825 | 50.475 | 훨씬 긴 trace |
| len1_ratio | 0.400 | 0.000 | adaptive만 collapse 제거 |
| density | 0.550 | 0.709 | 둘 다 target range |
| separation | 0.156 | 0.239 | adaptive가 더 분리 잘함 |

## 4. Causal judge 비교

Claude Haiku 4.5 dry3 pairwise judge:

| Scope | n | success | success rate |
|---|---:|---:|---:|
| Overall | 24 | 14 | 0.583333 |
| Ablation preservation | 12 | 4 | 0.333333 |
| Perturbation shift | 12 | 10 | 0.833333 |

축별:

| Axis | n | success rate |
|---|---:|---:|
| action_tendency_class | 6 | 0.833333 |
| social_orientation | 6 | 0.666667 |
| control_state | 6 | 0.500000 |
| target | 6 | 0.333333 |

해석:

```text
Perturbation은 강하다. Ablation은 약하다.
즉 trace를 조작하면 응답 방향은 움직이지만,
현재 ablation 설계는 trace의 causal necessity를 충분히 보여주지 못한다.
```

## 5. 최종 순위

| Rank | Version / Setting | 이유 |
|---:|---|---|
| 1 | `adaptive_thr0.63_clip1.6_inh0.10_start8_cap0.76` n=80 | collapse, density, representation metric을 동시에 만족 |
| 2 | adaptive n=40 sweep candidates | 모두 collapse와 density target을 동시 만족하지만 confirm 규모가 작음 |
| 3 | `persistent_less_inhibition` | collapse는 제거하지만 density 0.947로 과활성 |
| 4 | conservative high-fatigue configs | density는 맞지만 collapse가 크게 남음 |
| 5 | baseline | trace가 짧고 collapse가 큼 |

## 6. 결론

현재 성능 비교상 v3.1 best는 다음이다.

```text
adaptive_thr0.63_clip1.6_inh0.10_start8_cap0.76
```

이 설정은 현재까지 유일하게 다음 조건을 동시에 만족한다.

- `len1_ratio=0.0`
- `mean_activation_density=0.709412`
- `combined_separation_mean=0.238547`
- `combined_balanced_lift_mean=0.136426`
- tracked axes 전체 class-balanced lift 양수

논문에서는 이 설정을 v3.1의 main model로 두고, persistent/conservative/baseline을 ablation 또는 failed tradeoff baseline으로 제시하는 것이 가장 자연스럽다.
