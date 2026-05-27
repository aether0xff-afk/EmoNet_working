# Neural Trace Dynamics Stabilization Report

작성일: 2026-05-03

## 1. 목적

이 실험의 목적은 단순히 뉴런 수를 늘리는 것이 아니라, 자극 벡터가 EmoNet 신경망 안에서 충분히 오래 흐르도록 dynamics를 안정화하는 것이다.

직전 capacity ablation에서 확인된 문제:

```text
뉴런 수 증가 -> branch length 증가
하지만 len1 collapse는 해결되지 않음
```

따라서 이번 실험은 다음 질문에 답한다.

> 어떤 dynamics 조정이 branch collapse를 줄이고 neural trace geometry를 개선하는가?

## 2. 원인 가설

v3 core를 읽은 결과 collapse 원인은 크게 두 갈래로 볼 수 있다.

1. 초기 활성 실패
   - `k_threshold_base`가 높거나
   - 입력 신호가 적게 전달되어
   - 초반 뉴런들이 threshold를 넘지 못함

2. 흐름 조기 소멸
   - lateral inhibition
   - fatigue
   - refractory
   - convergence 조건
   - output fire value attenuation

이번 sweep은 이 둘을 나누어 작은 후보군으로 검증했다.

## 3. Sweep 후보

총 8개 설정을 비교했다.

| Config | Hypothesis |
|---|---|
| `baseline` | current reference |
| `lower_threshold` | weak stimuli가 threshold를 넘도록 도움 |
| `stronger_input` | downstream으로 더 많은 입력 전달 |
| `less_inhibition` | early suppression 완화 |
| `less_fatigue` | route가 펼쳐지기 전 죽는 현상 완화 |
| `persistent_flow` | lower threshold + stronger input + less fatigue |
| `persistent_less_inhibition` | persistent flow + weaker lateral suppression |
| `high_ne_modulation` | norepinephrine-like threshold lowering 강화 |

공통 조건:

```text
n_neurons=256
limit=40
feature=branch_mean
stim_source=auto
```

## 4. Sweep 결과

| Config | Mean branch len | len1 ratio | Activation density | Tracked lift mean | Tracked separation mean |
|---|---:|---:|---:|---:|---:|
| `persistent_less_inhibition` | 32.175 | 0.000 | 0.9328 | 0.000 | 0.2705 |
| `less_fatigue` | 31.125 | 0.000 | 0.9184 | -0.060 | 0.2436 |
| `baseline` | 10.275 | 0.500 | 0.4484 | -0.045 | 0.2160 |
| `high_ne_modulation` | 13.875 | 0.500 | 0.4529 | 0.005 | 0.2035 |
| `persistent_flow` | 34.750 | 0.000 | 0.9342 | -0.055 | 0.1935 |
| `stronger_input` | 20.175 | 0.500 | 0.4663 | -0.170 | 0.1388 |
| `lower_threshold` | 15.000 | 0.450 | 0.4970 | -0.015 | 0.1353 |
| `less_inhibition` | 16.925 | 0.500 | 0.4636 | -0.130 | 0.0761 |

## 5. 핵심 발견

### 5.1 Collapse 제거 성공

다음 세 설정은 first40에서 `len1_ratio=0.0`을 달성했다.

- `less_fatigue`
- `persistent_flow`
- `persistent_less_inhibition`

즉 collapse를 줄이는 데 가장 중요한 축은 단순 threshold나 input 증폭이 아니라 fatigue/persistence 쪽이었다.

### 5.2 Best candidate

가장 좋은 후보는 `persistent_less_inhibition`이다.

설정:

```json
{
  "k_threshold_base": 0.58,
  "k_remem_base": 0.80,
  "input_topk": 4,
  "input_signal_clip": 2.40,
  "fatigue_gain": 0.10,
  "fatigue_threshold_gain": 0.05,
  "fatigue_k_leak": 0.02,
  "inhibitory_suppression_gain": 0.06
}
```

이 후보는:

- len1 collapse를 제거했고
- 평균 branch length를 크게 늘렸으며
- tracked separation mean이 가장 높았다.

## 6. Best candidate full80 검증

`persistent_less_inhibition`을 80개 전체에 적용했다.

| Metric | Baseline full80 | Best full80 |
|---|---:|---:|
| Mean dominant branch len | 19.1875 | 37.9625 |
| len1 count | 37 | 0 |
| len1 ratio | 0.4625 | 0.0000 |
| Mean activation density | 0.4947 | 0.9470 |

결론:

> branch collapse 안정화는 성공했다.

하지만 activation density가 0.947까지 올라갔다. 이는 너무 많은 뉴런이 켜지는 과활성 위험을 의미한다.

## 7. Best full80 geometry

`branch_mean` 기준 nearest-neighbor lift:

| Axis | Lift |
|---|---:|
| `valence` | +0.1625 |
| `social_orientation` | +0.1000 |
| `appraisal_family` | +0.1500 |
| `control_state` | -0.0250 |
| `action_tendency_class` | -0.0125 |
| `target` | -0.1250 |

`branch_mean` 기준 group distance separation:

| Axis | Separation |
|---|---:|
| `valence` | +0.6637 |
| `arousal` | +0.5576 |
| `episode_family` | +0.3988 |
| `action_tendency_class` | +0.3280 |
| `control_state` | +0.1997 |
| `social_orientation` | +0.0939 |
| `appraisal_family` | +0.0304 |
| `target` | -0.0580 |

해석:

- 거리 기반 separation은 baseline보다 좋아졌다.
- `valence`, `arousal`, `action_tendency_class`는 꽤 뚜렷하다.
- `target`은 여전히 neural trace에서 잘 분리되지 않는다.
- `z` embedding은 여전히 약하다. 현재 `z`는 trace-as-emotion proof의 주 representation으로 부족하다.

## 8. 중요한 균형

이번 실험은 좋은 소식과 나쁜 소식을 같이 준다.

좋은 소식:

> dynamics 조정으로 branch collapse를 0까지 낮출 수 있다.

나쁜 소식:

> 현재 best 설정은 activation density가 너무 높아, 감정 특이적 군집이 아니라 전역 과활성일 수 있다.

따라서 다음 목표는 단순히 더 오래 흐르게 하는 것이 아니다.

```text
collapse는 낮게
activation density는 중간으로
geometry separation은 높게
```

이 세 조건을 동시에 만족해야 한다.

## 9. 다음 실험 방향

다음 sweep은 `persistent_less_inhibition` 주변의 미세 조정이다.

목표:

```text
len1_ratio <= 0.10
mean_activation_density 0.55 ~ 0.80
tracked_separation_mean >= baseline
```

조정 후보:

1. threshold를 살짝 올림
   - `k_threshold_base`: 0.60, 0.62, 0.64

2. input clip을 낮춤
   - `input_signal_clip`: 1.8, 2.0, 2.2

3. inhibition을 중간으로 복원
   - `inhibitory_suppression_gain`: 0.08, 0.10, 0.12

4. fatigue를 너무 낮추지 않음
   - `fatigue_gain`: 0.12, 0.15, 0.18

5. trace feature 개선
   - `branch_mean`만 쓰지 말고 route histogram, transition features를 추가한다.

## 10. 한 줄 결론

dynamics 안정화는 성공적으로 시작됐다. branch collapse는 제거했지만, 이제 과활성을 줄이면서 감정 geometry를 유지하는 정밀 조정 단계로 가야 한다.

