# Neural Trace Capacity Ablation Report

작성일: 2026-05-03

## 1. 목적

이번 실험은 사용자가 제안한 질문을 직접 확인하기 위한 것이다.

> 뉴런 수를 늘리면 neural activation trace가 더 감정답게 군집화되는가?

비교한 설정:

```text
n_neurons = 256
n_neurons = 512
n_neurons = 1024
```

공정 비교를 위해 세 설정 모두 같은 첫 40개 targeted record 기준으로 비교했다.

## 2. 실행 상태

| Neurons | Rows | Export status | Geometry status |
|---:|---:|---|---|
| 256 | 40 | 완료 | 완료 |
| 512 | 40 | 완료 | 완료 |
| 1024 | 40 | 완료 | 완료 |

추가로 512는 full80 export도 완료했다. 1024 full80은 시간 제한으로 중단되었고, trace file은 62개까지 저장되었지만 manifest/summary가 완성되지 않았다. 따라서 공식 비교는 first40 기준으로 한다.

## 3. Branch health

| Neurons | Mean dominant branch len | len1 count | len1 ratio | Mean activation density |
|---:|---:|---:|---:|---:|
| 256 | 10.275 | 20 | 0.500 | 0.4484 |
| 512 | 20.050 | 20 | 0.500 | 0.4635 |
| 1024 | 24.900 | 20 | 0.500 | 0.4630 |

해석:

- 뉴런 수를 늘리면 평균 branch length는 증가한다.
- 하지만 `dominant_branch_len <= 1`인 collapse 샘플 수는 줄지 않았다.
- 즉 뉴런 수 증가는 "살아남은 trace를 더 길게 만드는 효과"는 있지만, "죽는 trace를 살리는 효과"는 아직 없다.

## 4. Geometry: `branch_mean` nearest-neighbor lift

현재 가장 유망했던 neural representation인 `branch_mean` 기준으로 비교했다.

| Axis | 256 lift | 512 lift | 1024 lift |
|---|---:|---:|---:|
| `valence` | +0.075 | +0.050 | +0.050 |
| `arousal` | +0.000 | +0.000 | -0.025 |
| `target` | -0.100 | -0.100 | -0.075 |
| `control_state` | -0.175 | -0.200 | -0.125 |
| `social_orientation` | -0.075 | -0.075 | -0.150 |
| `action_tendency_class` | +0.025 | +0.050 | -0.075 |
| `episode_family` | -0.150 | -0.125 | -0.125 |
| `appraisal_family` | -0.075 | -0.075 | -0.200 |

해석:

- 512는 `action_tendency_class`에서 소폭 개선했다.
- 1024는 `target`, `control_state`에서 음수 폭이 조금 줄었지만, `social_orientation`, `action_tendency_class`, `appraisal_family`는 악화했다.
- 전체적으로 뉴런 수 증가가 emotion geometry를 일관되게 개선한다는 증거는 없다.

## 5. 현재 결론

이번 capacity ablation의 결론은 명확하다.

> 뉴런 수를 늘리는 것만으로는 trace-as-emotion 문제가 해결되지 않는다.

더 정확히 말하면:

```text
뉴런 수 증가
-> 평균 branch length 증가
-> 하지만 collapse ratio 유지
-> emotion geometry 개선은 불안정
```

따라서 성능 개선의 핵심은 단순 capacity가 아니라 다음 쪽일 가능성이 높다.

- activation propagation rule
- threshold/refractory 설정
- branch collapse mitigation
- stimulus encoder 품질
- neural trace feature extraction 방식
- neuron cluster regularization

## 6. 중요한 해석

이 결과는 실패라기보다 좋은 진단이다.

처음에는 "뉴런을 늘리면 알아서 군집화될 수 있나?"라는 질문이었다. 이번 결과는 그 답을 이렇게 준다.

> 현재 구조에서는 뉴런을 늘려도 자동 군집화는 충분히 일어나지 않는다.

즉 다음 증명 방향은:

```text
more neurons
```

가 아니라:

```text
better trace dynamics + cluster pressure + lower branch collapse
```

이다.

## 7. 다음 단계

다음 실험은 neuron count가 아니라 branch dynamics ablation이어야 한다.

우선순위:

1. len1 collapse를 줄이는 설정 sweep
   - threshold
   - refractory
   - convergence patience
   - input gain
   - lateral inhibition

2. same n=256에서 branch health가 좋아지는 설정 찾기
   - 목표: len1 ratio < 0.20
   - 평균 branch length > 20

3. 그 다음 다시 256/512/1024 비교
   - collapse가 낮아진 상태에서 capacity 효과를 재평가

4. neural trace feature 개선
   - branch_mean만으로는 부족하다.
   - route histogram, transition pattern, temporal pooling을 추가해야 한다.

## 8. 한 줄 결론

뉴런 수를 늘리면 trace 길이는 늘지만, 현재로서는 감정 군집화나 branch collapse 해결을 보장하지 않는다. 먼저 trace dynamics를 안정화해야 한다.

