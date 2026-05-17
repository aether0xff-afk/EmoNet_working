# Neural Trace Dynamics Fine Sweep v2 Report

작성일: 2026-05-03

## 1. 목적

이 실험은 v3.1의 핵심 병목을 좁히기 위한 dynamics fine sweep이다.

목표는 세 조건을 동시에 만족하는 설정을 찾는 것이었다.

```text
len1_ratio <= 0.10
mean_activation_density 0.55 ~ 0.80
branch_mean / branch_temporal geometry >= baseline
```

## 2. 스크립트 변경

`scripts/tune_neural_trace_dynamics.py`를 수정했다.

변경점:

- `branch_mean`과 `branch_temporal`을 함께 평가
- activation density 과활성 penalty 강화
- `--resume` 지원
- `--start-index`, `--max-configs` 지원
- `--grid-mode conservative` 추가
- config마다 summary를 incremental write

이제 긴 sweep이 중간에 끊겨도 완료된 config는 재사용할 수 있다.

## 3. Fine grid partial result

기존 fine grid 중 7개를 계산했다.

공통 패턴:

- `len1_ratio=0.0`
- `mean_activation_density=0.939~0.959`
- tracked lift는 대부분 음수

대표 상위 후보:

| Config | len1 | density | branch len | tracked sep | tracked lift |
|---|---:|---:|---:|---:|---:|
| `thr0.60_clip1.6_inh0.12_low_fatigue` | 0.000 | 0.955 | 40.45 | 0.2008 | -0.150 |
| `thr0.60_clip1.9_inh0.08_mid_fatigue` | 0.000 | 0.941 | 37.53 | 0.1791 | -0.135 |
| `thr0.60_clip1.6_inh0.08_low_fatigue` | 0.000 | 0.958 | 39.75 | 0.1761 | -0.110 |

해석:

```text
fine grid는 collapse 제거에는 성공하지만,
대부분 전역 과활성 상태로 간다.
```

## 4. High-threshold targeted result

`threshold=0.66` 후보 4개를 별도 실행했다.

결과:

| Config | len1 | density | branch len | tracked sep | tracked lift |
|---|---:|---:|---:|---:|---:|
| `thr0.66_clip1.6_inh0.12_mid_fatigue` | 0.000 | 0.940 | 33.68 | 0.2397 | -0.095 |
| `thr0.66_clip1.6_inh0.08_low_fatigue` | 0.000 | 0.957 | 39.33 | 0.2216 | -0.125 |
| `thr0.66_clip1.6_inh0.08_mid_fatigue` | 0.000 | 0.927 | 31.75 | 0.1757 | -0.120 |

해석:

```text
threshold를 0.66까지 올려도 density는 여전히 0.92 이상이다.
```

따라서 단순 threshold 상승만으로는 과활성이 해결되지 않는다.

## 5. Conservative grid result

grid 바깥의 보수적 후보를 4개 추가했다.

변경 방향:

- threshold `0.70~0.74`
- input clip `1.2~1.4`
- inhibition `0.16~0.20`
- fatigue 강화

결과:

| Config | len1 | density | branch len | tracked sep | tracked lift |
|---|---:|---:|---:|---:|---:|
| `thr0.74_clip1.2_inh0.20_high_fatigue` | 0.400 | 0.550 | 19.83 | 0.1553 | -0.085 |
| `thr0.70_clip1.2_inh0.16_high_fatigue` | 0.375 | 0.586 | 25.40 | 0.0723 | -0.160 |
| `thr0.74_clip1.4_inh0.20_high_fatigue` | 0.400 | 0.556 | 22.65 | 0.0626 | -0.095 |
| `thr0.70_clip1.4_inh0.16_high_fatigue` | 0.400 | 0.570 | 26.20 | 0.0408 | -0.095 |

해석:

```text
conservative grid는 density 제어에는 성공하지만,
branch collapse가 다시 커진다.
```

## 6. 핵심 결론

이번 sweep에서 가장 중요한 발견은 tradeoff다.

| Regime | 장점 | 단점 |
|---|---|---|
| persistent / low inhibition | collapse 제거 | density 0.94 이상 과활성 |
| conservative / high threshold | density 0.55~0.59 제어 | len1 ratio 0.375~0.400 |

따라서 현재 구조에서는 단순 파라미터 sweep만으로 세 조건을 동시에 만족하기 어렵다.

```text
collapse를 죽이면 density가 터지고,
density를 잡으면 collapse가 돌아온다.
```

## 7. 다음 설계 방향

다음 단계는 파라미터 크기 조절이 아니라 dynamics 구조 수정이다.

우선순위:

1. Early ignition과 late persistence를 분리한다.
   - 초반에는 threshold를 낮추고
   - 일정 tick 이후에는 adaptive fatigue/inhibition을 강화한다.

2. Density-aware inhibition을 추가한다.
   - active density가 목표 상단을 넘으면 inhibition을 동적으로 강화한다.

3. Branch selection을 activation density가 아니라 sparse route fidelity 쪽으로 정규화한다.

4. `z` embedding은 현 단계 주장이 아니다.
   - 주 representation은 계속 `branch_mean` / `branch_temporal`이다.

## 8. v3.1 Acceptance 상태

| Gate | 상태 |
|---|---|
| Neural trace export | 통과 |
| Feature probe | 통과 |
| Capacity ablation | 통과 |
| Collapse/density tradeoff 진단 | 통과 |
| Stable balanced dynamics | 미완료 |
| Causal pair judge | 준비 완료, API 실행 대기 |

## 9. 한 줄 결론

v3.1은 이제 "trace가 어디에 담기는가"와 "왜 아직 안정적 emotion geometry가 안 나오는가"를 꽤 좁혔다. 현재 병목은 capacity가 아니라 early ignition과 late density control을 분리하지 못하는 dynamics 구조다.
