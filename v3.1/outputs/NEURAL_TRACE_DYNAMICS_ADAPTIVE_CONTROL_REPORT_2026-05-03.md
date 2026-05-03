# Neural Trace Dynamics Adaptive Control Report

작성일: 2026-05-03

## 1. 목적

이 실험은 `NEURAL_TRACE_DYNAMICS_FINE_SWEEP_V2_REPORT_2026-05-03.md`에서 확인된 병목을 직접 겨냥한다.

기존 tradeoff:

```text
collapse를 제거하면 activation density가 0.94 이상으로 과활성되고,
density를 0.55~0.80으로 낮추면 len1_ratio가 0.375~0.400으로 다시 커졌다.
```

따라서 단순 threshold/fatigue/inhibition sweep이 아니라, early ignition과 late density control을 분리하는 opt-in dynamics를 추가했다.

## 2. 구현 변경

`v3/emonet/core.py`에 기본 비활성화 상태의 density-aware controller를 추가했다.

추가 config:

- `density_control_start_tick`
- `density_target_high`
- `density_soft_k_leak_gain`
- `density_hard_cap`
- `density_pruned_fatigue_gain`

동작:

1. 지정 tick 이전에는 개입하지 않는다.
2. 활성 후보 density가 `density_target_high`를 넘으면 K를 soft leak으로 낮춘다.
3. 그래도 `density_hard_cap`을 넘으면 K margin이 높은 후보만 남긴다.
4. cap에서 밀린 후보에는 fatigue를 조금 더한다.

기본값은 기존 동작을 보존한다.

## 3. v3.1 연결 변경

`v3.1/scripts/export_neural_activation_traces.py`가 새 파라미터를 전달하고 manifest에 기록하도록 수정했다.

`v3.1/scripts/tune_neural_trace_dynamics.py`에는 `--grid-mode adaptive`를 추가했다. 또한 `--resume` 상태에서 summary CSV를 덮어쓰지 않고 config별 row를 upsert하도록 고쳤다.

## 4. Smoke Result

명령:

```powershell
cd .\v3.1
python .\scripts\tune_neural_trace_dynamics.py --grid-mode adaptive --limit 8 --max-configs 1 --output-dir outputs\neural_trace_dynamics_adaptive_smoke8 --resume
```

결과:

| Config | n | len1_ratio | density | mean branch len |
|---|---:|---:|---:|---:|
| `adaptive_thr0.60_clip1.6_inh0.10_start8_cap0.78` | 8 | 0.000 | 0.696831 | 38.50 |

해석:

```text
작은 smoke에서 처음으로 collapse 제거와 target density가 동시에 만족되었다.
```

## 5. Adaptive Sweep Result

명령:

```powershell
cd .\v3.1
python .\scripts\tune_neural_trace_dynamics.py --grid-mode adaptive --limit 40 --max-configs 2 --output-dir outputs\neural_trace_dynamics_adaptive_sweep_v1 --resume
```

결과:

| Rank | Config | n | len1 | density | branch len | combined sep | balanced lift | objective |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `adaptive_thr0.63_clip1.6_inh0.10_start8_cap0.76` | 40 | 0.000 | 0.686406 | 48.95 | 0.204554 | 0.045305 | 0.987814 |
| 2 | `adaptive_thr0.63_clip1.8_inh0.12_start10_cap0.80` | 40 | 0.000 | 0.741523 | 44.15 | 0.190091 | 0.053214 | 0.949129 |
| 3 | `adaptive_thr0.66_clip1.6_inh0.12_start8_cap0.76` | 40 | 0.000 | 0.676906 | 46.625 | 0.173333 | 0.054263 | 0.892380 |
| 4 | `adaptive_thr0.60_clip1.6_inh0.10_start8_cap0.78` | 40 | 0.000 | 0.713828 | 45.55 | 0.167053 | -0.055580 | 0.781868 |

Best config tracked group-distance separation:

| Axis | separation |
|---|---:|
| valence | 0.509492 |
| social_orientation | 0.189205 |
| action_tendency_class | 0.167775 |
| appraisal_family | 0.139984 |
| control_state | 0.071716 |

Best config majority-baseline nearest-neighbor lift는 아직 음수다. 이 corpus는 `negative/high/mixed` 계열이 강하게 많은 불균형 subset이므로, class-balanced nearest-neighbor lift를 추가했다.

Best config balanced nearest-neighbor lift:

| Axis | balanced lift |
|---|---:|
| valence | 0.037330 |
| social_orientation | 0.165793 |
| action_tendency_class | -0.062729 |
| appraisal_family | 0.116277 |
| control_state | 0.043286 |

해석:

```text
majority baseline 기준 lift는 아직 약하지만,
class-balanced 기준에서는 action_tendency_class를 제외한 tracked axes가 양수다.
```

## 6. 결론

Balanced dynamics gate는 기존 상태인 `미완료`에서 다음 상태로 이동했다.

```text
adaptive late density control은 n=40 sweep에서
모든 후보가 len1_ratio=0.0과 density target 0.55~0.80을 동시에 만족했다.
```

## 6.1 n=80 Confirmatory Result

Best n=40 config를 n=80으로 재실행했다.

명령:

```powershell
cd .\v3.1
python .\scripts\tune_neural_trace_dynamics.py --grid-mode adaptive --limit 80 --start-index 2 --max-configs 1 --output-dir outputs\neural_trace_dynamics_adaptive_confirm80_v1 --resume
```

결과:

| Config | n | len1 | density | branch len | combined sep | balanced lift | objective |
|---|---:|---:|---:|---:|---:|---:|---:|
| `adaptive_thr0.63_clip1.6_inh0.10_start8_cap0.76` | 80 | 0.000 | 0.709412 | 50.475 | 0.238547 | 0.136426 | 1.184853 |

축별 class-balanced nearest-neighbor lift:

| Axis | balanced lift |
|---|---:|
| valence | 0.258480 |
| social_orientation | 0.119748 |
| action_tendency_class | 0.017921 |
| appraisal_family | 0.078728 |
| control_state | 0.175607 |

축별 group-distance separation:

| Axis | separation |
|---|---:|
| valence | 0.610754 |
| social_orientation | 0.113039 |
| action_tendency_class | 0.206921 |
| appraisal_family | 0.093028 |
| control_state | 0.206567 |

해석:

```text
n=80 confirm에서는 collapse 제거, target density, group-distance separation,
class-balanced nearest-neighbor lift가 동시에 통과했다.
특히 action_tendency_class도 n=40의 음수 balanced lift에서 n=80 양수로 회복했다.
```

다만 representation evidence는 아직 완성 판정이 아니다.

- group-distance separation은 best config에서 tracked axes 모두 양수다.
- nearest-neighbor lift는 majority baseline 기준으로 아직 보수적으로 해석해야 한다.
- class-balanced nearest-neighbor lift는 n=80 best config에서 tracked axes 모두 양수다.
- 따라서 dynamics 안정화는 n=80 기준 통과 후보이며, representation proof는 causal judge와 label-balance 해석을 붙이면 논문화 가능한 상태로 이동한다.

## 7. 다음 실행

우선순위:

1. API judge smoke를 실행해 causal pair success rate를 확인한다.
2. 논문용 figure/table을 위해 n=80 confirm 결과를 summary table로 고정한다.
3. 필요하면 best adaptive config로 full set trace export를 실행한다.
4. majority-baseline nearest-neighbor lift는 label imbalance limitation으로 명시하고 class-balanced metric을 주 지표로 둔다.
