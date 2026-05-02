# Neural Trace-As-Emotion Probe Report

작성일: 2026-05-02

## 1. 정정된 trace 정의

사용자가 말한 trace는 symbolic appraisal table이 아니다.

이번 v3.1에서 새로 잡은 trace 정의는 다음이다.

```text
stimulus vector
-> EmoNet neural network
-> tick-by-tick neuron activation
-> dominant branch / branch tensor / z
-> neural activation trace
```

즉 감정의 본체 후보는 `target`, `control_state`, `action_tendency` 같은 해석 필드가 아니라, 자극 벡터가 뉴런망을 지나며 만든 활성 궤적이다.

해석 필드는 이제 평가용 외부 라벨로만 사용한다.

```text
object being tested: neural activation trace
external probes: valence, target, control_state, social_orientation, action_tendency_class
```

## 2. 구현한 것

새 스크립트:

- `scripts/export_neural_activation_traces.py`
- `scripts/probe_neural_trace_geometry.py`

새 설계 문서:

- `docs/NEURAL_TRACE_AS_EMOTION_DESIGN.md`

생성된 neural trace 산출물:

- `outputs/neural_trace_probe_v1_full80/neural_trace_summary.csv`
- `outputs/neural_trace_probe_v1_full80/neural_trace_manifest.json`
- `outputs/neural_trace_probe_v1_full80/traces_npz/*.npz`
- `outputs/neural_trace_probe_v1_full80/neural_trace_geometry_z.json`
- `outputs/neural_trace_probe_v1_full80/neural_trace_geometry_activation.json`
- `outputs/neural_trace_probe_v1_full80/neural_trace_geometry_branch.json`

각 `.npz`는 다음을 포함한다.

| Array | 의미 |
|---|---|
| `activation` | tick x neuron K matrix |
| `branch_tensor` | dominant branch feature tensor |
| `z` | encoded neural trace embedding |
| `stim_vec` | 4D stimulus vector |
| `dominant_branch_ids` | dominant route through network |
| `active_counts` | active neuron count per tick |

## 3. 런타임 조건

현재 환경에는 scikit-learn이 없어 v3 ridge text encoder를 사용할 수 없었다.

따라서 이번 run은 fallback stimulus vector를 사용했다.

```text
stim_source=auto
sklearn_available=false
```

이 fallback은 text/label 기반으로 4D stimulus vector를 만든 뒤, 그 벡터를 v3 EmoNet network에 흘려 activation trace를 얻는다.

중요한 한계:

> 이번 결과는 neural network propagation trace의 첫 probe이지만, v3의 원래 learned text encoder를 사용한 결과는 아니다.

## 4. Full80 export 결과

| 항목 | 값 |
|---|---:|
| Requested rows | 80 |
| OK rows | 80 |
| Error rows | 0 |
| Neurons | 256 |
| z encoder | `stat` |

Neural trace extraction 자체는 성공했다.

## 5. Branch health

Full80 기준 branch 상태:

| Metric | Value |
|---|---:|
| mean dominant branch length | 19.1875 |
| len1 count | 37 |
| len1 ratio | 0.4625 |
| mean activation density | 0.4947 |

해석:

> neural trace는 추출되었지만, branch collapse가 아직 많이 남아 있다.

80개 중 37개가 `dominant_branch_len <= 1`이다. 따라서 현재 neural trace geometry가 약하게 나오는 가장 큰 이유 중 하나는 trace 자체가 충분히 펼쳐지지 못하는 샘플이 많기 때문일 가능성이 크다.

## 6. Nearest-neighbor consistency

세 가지 neural representation을 비교했다.

| Representation | 설명 |
|---|---|
| `z` | encoded trace embedding |
| `activation_meanmax` | 전체 neuron activation의 mean/max 요약 |
| `branch_mean` | dominant branch tensor의 mean/max 요약 |

### 6.1 `z` embedding

| Axis | NN consistency | Majority baseline | Lift |
|---|---:|---:|---:|
| `valence` | 0.7750 | 0.7750 | +0.0000 |
| `social_orientation` | 0.3750 | 0.3500 | +0.0250 |
| `appraisal_family` | 0.3000 | 0.2875 | +0.0125 |
| `action_tendency_class` | 0.2750 | 0.3250 | -0.0500 |
| `target` | 0.4625 | 0.6875 | -0.2250 |
| `control_state` | 0.4750 | 0.7000 | -0.2250 |

`z`는 현재 emotion geometry를 강하게 잡지 못한다.

### 6.2 `activation_meanmax`

| Axis | NN consistency | Majority baseline | Lift |
|---|---:|---:|---:|
| `social_orientation` | 0.4000 | 0.3500 | +0.0500 |
| `arousal` | 0.8875 | 0.9250 | -0.0375 |
| `target` | 0.3250 | 0.6875 | -0.3625 |
| `action_tendency_class` | 0.2250 | 0.3250 | -0.1000 |

전체 activation mean/max도 아직 강하지 않다.

### 6.3 `branch_mean`

| Axis | NN consistency | Majority baseline | Lift |
|---|---:|---:|---:|
| `valence` | 0.9500 | 0.7750 | +0.1750 |
| `social_orientation` | 0.4625 | 0.3500 | +0.1125 |
| `appraisal_family` | 0.3625 | 0.2875 | +0.0750 |
| `action_tendency_class` | 0.3750 | 0.3250 | +0.0500 |
| `target` | 0.4875 | 0.6875 | -0.2000 |
| `control_state` | 0.6250 | 0.7000 | -0.0750 |

현재 세 representation 중 `branch_mean`이 가장 유망하다. 특히 `valence`, `social_orientation`, `action_tendency_class`, `appraisal_family`에서 baseline 이상의 신호가 있다.

## 7. Group distance separation: `branch_mean`

`branch_mean` 기준 intra-group과 inter-group 거리를 비교했다.

| Axis | Mean intra distance | Mean inter distance | Separation |
|---|---:|---:|---:|
| `valence` | 1.0654 | 1.5807 | +0.5153 |
| `arousal` | 1.1904 | 1.6658 | +0.4754 |
| `episode_family` | 1.0332 | 1.4899 | +0.4567 |
| `action_tendency_class` | 1.0764 | 1.2982 | +0.2218 |
| `control_state` | 1.1737 | 1.3587 | +0.1849 |
| `social_orientation` | 1.1963 | 1.2825 | +0.0862 |
| `appraisal_family` | 1.2399 | 1.2630 | +0.0231 |
| `target` | 1.2988 | 1.2161 | -0.0826 |

거리 기반으로는 `valence`, `arousal`, `episode_family`, `action_tendency_class`, `control_state`에서 positive separation이 나온다.

## 8. 현재 결론

이번 neural trace probe는 완전 성공이 아니다. 하지만 중요한 전환은 성공했다.

성공한 것:

> v3.1에서 실제 neural activation trace를 추출하고 저장하는 파이프라인을 만들었다.

초기 긍정 신호:

> `branch_mean` representation에서 valence, social_orientation, action_tendency_class, appraisal_family가 baseline보다 높은 nearest-neighbor lift를 보였다.

약한 지점:

> `z` embedding과 full activation mean/max는 아직 감정 라벨을 강하게 분리하지 못한다.

가장 큰 문제:

> branch collapse가 여전히 높다. full80에서 len1 ratio가 0.4625다.

따라서 현재 주장은 이렇게 제한해야 한다.

```text
neural activation trace를 감정 representation으로 검증하는 파이프라인은 성공했다.
하지만 현재 256-neuron/stat-z/fallback-stim 설정에서는 trace geometry가 아직 약하다.
```

## 9. 다음 단계

1. Branch collapse를 먼저 줄인다.
   - len1 ratio를 0.46에서 0.15 이하로 낮추는 설정이 필요하다.

2. neuron count ablation을 실행한다.
   - 256 / 512 / 1024 비교
   - 각 설정에서 branch health와 neural geometry를 같이 본다.

3. 원래 v3 text encoder 또는 대체 learned encoder를 복구한다.
   - 현재 fallback stim vector는 첫 proof용이다.
   - 최종 주장은 learned stimulus encoder 기반이어야 한다.

4. neural trace feature를 개선한다.
   - 단순 mean/max보다 trajectory-aware feature가 필요하다.
   - 예: active route histogram, transition matrix, temporal pooling, branch edit distance

5. symbolic appraisal labels는 본체가 아니라 probe로 유지한다.

## 10. 한 줄 결론

이제 v3.1은 사용자가 원래 말한 의미의 trace, 즉 neural activation trajectory를 다루기 시작했다. 첫 결과는 아직 강한 우위 증명이 아니라, neural trace 추출과 초기 geometry 검증의 출발점이다.

