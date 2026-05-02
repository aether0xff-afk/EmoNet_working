# Trace-As-Emotion Proof Start

작성일: 2026-05-02

## 1. 증명 목표

v3.1의 증명 목표는 다음이다.

> trace는 감정을 설명하는 보조 정보가 아니라, 감정 상태를 구성하는 representation이다.

따라서 첫 증명은 생성 응답 품질이 아니라 representation-level evidence에서 시작한다.

핵심 질문:

1. 비슷한 감정 구조를 가진 record들이 trace 공간에서 가까운가?
2. trace 축을 정규화하면 action tendency도 군집 구조를 보이는가?
3. trace는 단순 label보다 appraisal/action structure를 더 잘 담는가?

## 2. 수행한 작업

이번 단계에서 새로 추가한 것은 다음이다.

| Artifact | 역할 |
|---|---|
| `scripts/normalize_trace_axes.py` | free-text trace 축을 canonical category로 정규화 |
| `outputs/targeted_records_trace_normalized.csv` | 정규화된 targeted records |
| `outputs/trace_axis_normalization_summary.json` | 정규화 분포 요약 |
| `outputs/trace_emotion_probe_normalized_summary.json` | 정규화 후 trace-space probe 결과 |

정규화된 주요 축:

- `action_tendency_class`
- `episode_family`
- `appraisal_family`
- `trace_emotion_signature`

## 3. Action tendency 정규화 결과

기존 `action_tendency`는 긴 자연어 설명이라 exact-match 비교에서 거의 전부 고유값이었다. 따라서 canonical category로 변환했다.

| Action tendency class | Count |
|---|---:|
| `defend` | 26 |
| `confront` | 17 |
| `plan` | 11 |
| `seek_support` | 7 |
| `repair` | 7 |
| `approach` | 4 |
| `withdraw` | 4 |
| `other_action` | 3 |
| `inhibit` | 1 |

이 결과는 action tendency가 무작위 텍스트가 아니라 몇 개의 감정 행동 경향으로 압축될 수 있음을 보여준다.

## 4. Episode/appraisal family 정규화 결과

| Episode family | Count |
|---|---:|
| `other_blame_boundary` | 56 |
| `threat_anxiety` | 8 |
| `other_episode` | 7 |
| `self_blame_guilt` | 6 |
| `repair_gratitude` | 2 |
| `planning_control` | 1 |

| Appraisal family | Count |
|---|---:|
| `approach_or_repair` | 23 |
| `mixed_appraisal` | 23 |
| `low_control_distress` | 17 |
| `other_directed_defense` | 8 |
| `situation_focused_coping` | 5 |
| `self_directed_evaluation` | 3 |
| `withdrawal_or_protection` | 1 |

`episode_family`는 아직 `other_blame_boundary`에 많이 몰려 있다. 반면 `appraisal_family`는 더 잘 분산되며, trace-as-emotion proof에는 `episode_label`보다 유용한 축으로 보인다.

## 5. 정규화 후 nearest-neighbor consistency

정규화된 trace 공간에서 각 record의 nearest neighbor가 같은 axis 값을 공유하는지 측정했다.

| Axis | n | NN consistency | Majority baseline | Lift |
|---|---:|---:|---:|---:|
| `appraisal_family` | 80 | 0.9750 | 0.2875 | +0.6875 |
| `social_orientation` | 80 | 0.9375 | 0.3500 | +0.5875 |
| `control_state` | 80 | 0.9750 | 0.7000 | +0.2750 |
| `target` | 80 | 0.9500 | 0.6875 | +0.2625 |
| `action_tendency_class` | 80 | 0.5875 | 0.3250 | +0.2625 |
| `valence` | 80 | 0.9625 | 0.7750 | +0.1875 |
| `episode_family` | 80 | 0.7875 | 0.7000 | +0.0875 |
| `arousal` | 80 | 0.9250 | 0.9250 | +0.0000 |

가장 중요한 변화는 `action_tendency_class`다. 기존 free-text `action_tendency`는 lift가 음수였지만, canonical class로 정규화하자 lift가 `+0.2625`로 올라갔다.

## 6. 정규화 후 group separation

같은 axis 값을 가진 pair의 평균 거리와 다른 axis 값을 가진 pair의 평균 거리를 비교했다.

| Axis | Mean intra distance | Mean inter distance | Separation |
|---|---:|---:|---:|
| `appraisal_family` | 0.4333 | 0.6654 | +0.2321 |
| `social_orientation` | 0.4876 | 0.6661 | +0.1785 |
| `action_tendency_class` | 0.4742 | 0.6465 | +0.1723 |
| `valence` | 0.5520 | 0.7212 | +0.1691 |
| `episode_family` | 0.5339 | 0.6992 | +0.1653 |
| `arousal` | 0.5933 | 0.7476 | +0.1542 |
| `control_state` | 0.5458 | 0.6983 | +0.1526 |
| `target` | 0.5613 | 0.6705 | +0.1091 |

모든 정규화 축에서 intra-group distance가 inter-group distance보다 낮다. 즉 같은 감정 구조를 가진 record들이 trace 공간에서 더 가까운 경향이 있다.

## 7. 현재까지의 증명 상태

현재 단계에서 말할 수 있는 것:

> trace space는 `appraisal_family`, `social_orientation`, `target`, `control_state`, `action_tendency_class` 축에서 emotion-state-like structure를 보인다.

아직 말하면 안 되는 것:

> EmoNet이 완전히 감정을 학습했다.

> trace가 인간 감정의 완전한 신경 표현이다.

> 뉴런 수를 늘리면 자동으로 해결된다.

현재는 representation-level proof의 첫 단계가 성공한 것이다.

## 8. 다음 증명 단계

다음 단계는 causal proof다.

1. Trace ablation
   - `target`, `social_orientation`, `action_tendency_class`를 하나씩 제거한다.
   - appraisal fidelity와 action tendency fit이 떨어지는지 본다.

2. Trace perturbation
   - 같은 stimulus에서 `target=other`를 `target=self`로 바꾼다.
   - 응답의 blame 방향과 action tendency가 바뀌는지 본다.

3. Cluster discovery
   - label 없이 trace vector만으로 cluster를 만든다.
   - cluster가 appraisal/action family와 맞는지 확인한다.

4. Human validation
   - 사람이 응답만 보고 원래 trace state를 맞힐 수 있는지 평가한다.

## 9. 한 줄 결론

증명은 시작되었고, 첫 결과는 긍정적이다.

`action_tendency`를 canonical emotion-action class로 정규화하자 trace space에서 구조가 드러났다. 이는 trace를 단순 prompt metadata가 아니라 감정 상태 representation으로 볼 수 있다는 첫 representation-level 증거다.

