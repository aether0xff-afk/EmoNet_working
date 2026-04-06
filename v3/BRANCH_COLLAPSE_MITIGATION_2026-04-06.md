# Branch Collapse Mitigation (2026-04-06)

## 문제 요약

current `out_z_training_extended40.csv` 기준 dominant branch 길이 분포는 아래와 같다.

- total rows: 51,628
- branch length = 1: 50,257
- mean branch length: 1.0539
- max length: 8

즉, 대부분의 샘플에서 branch가 사실상 한 tick 반응으로 끝난다. 이 상태에서는 branch history를 중간표현의 핵심 근거로 쓰기 어렵고, learned `z` encoder가 배울 수 있는 시계열 정보도 매우 제한된다.

## 현재 기본값

현재 `EmoNetConfig` 기본값은 초기판보다 branch persistence를 더 강하게 주도록 올려두었다.

- `max_ticks = 40`
- `min_ticks_before_converged = 6`
- `k_threshold_base = 0.72`
- `k_remem_base = 0.95`
- `k_decay = 0.99`
- `memory_decay = 0.985`
- `memory_stim_mix = 0.25`
- `memory_k_mix = 0.35`
- `mela_dropout_gain = 0.04`
- `sero_prune_gain = 0.04`
- `dopa_rewire_gain = 0.80`
- `branch_end_window = 6`
- `branch_length_bonus = 0.35`

즉, 현재 기준점은 "초기 기본값"이 아니라 이미 persistence를 강화한 상태다.

## 현재 코드에서 보이는 직접 원인

`emonet/core.py` 기준으로 branch가 짧아질 강한 요인은 아래와 같다.

### 1. dynamics가 너무 빨리 죽음

기본 설정:

- `k_threshold_base = 1.0`
- `k_decay = 0.95`
- `refractory_ticks = 3`
- `mela_dropout_gain = 0.30`
- `sero_prune_gain = 0.30`
- `memory_k_mix = 0.10`

이 조합은 한 번 활성된 노드가 다음 tick에 다시 이어서 기여하기 어렵게 만든다.

### 2. modulatory dropout이 공격적임

`_apply_modulatory_effects()`에서 melatonin 축이 올라가면 awake node 전체에 dropout이 걸린다. branch가 자라기 전에 활성 노드 풀이 급격히 줄 수 있다.

### 3. rewiring보다 pruning 압력이 강함

현재는 serotonin 기반 pruning과 dopamine 기반 add attempt가 동시에 있지만, 실제 branch를 유지하는 관점에서는 pruning이 더 즉각적으로 길이를 줄이는 쪽으로 작동할 가능성이 높다.

### 4. survivor-only pruning이 늦은 단발 spike를 선호함

`BranchExtractor.prune_to_survivors()`는 마지막 non-empty tick의 활성 노드만 survivor로 잡고 역추적한다. 이 구조에서는 earlier에 길게 이어진 경로가 있더라도, 마지막 tick의 고립된 강한 노드가 dominant branch의 끝점이 되면 길이 1 branch가 쉽게 나온다.

즉, 현재 branch 길이 1 문제는 단순히 dynamics만의 실패가 아니라, extraction criterion도 함께 만드는 현상이다.

## 해결 전략

## A. dynamics를 덜 빨리 죽게 만들기

가장 먼저 할 실험이다.

권장 조정:

- `refractory_ticks`: `3 -> 1` 또는 `2`
- `k_threshold_base`: `1.0 -> 0.8` 근처 탐색
- `k_decay`: `0.95 -> 0.98` 근처 탐색
- `memory_k_mix`: `0.10 -> 0.20` 또는 `0.25`
- `k_remem_base`: `1.2 -> 1.0` 근처 탐색

의도:

- 이미 켜진 경로가 다음 tick까지 더 쉽게 이어지게 만들기
- 약한 연쇄 활성도 history에 남게 만들기

## B. dropout / pruning을 완화하고 rewiring을 강화하기

권장 조정:

- `mela_dropout_gain`: `0.30 -> 0.05 ~ 0.10`
- `sero_prune_gain`: `0.30 -> 0.05 ~ 0.10`
- `dopa_rewire_gain`: `0.30 -> 0.50 ~ 0.80`
- `min_out_degree`: `1 -> 2`

의도:

- branch가 자라기 전에 그래프가 너무 빨리 희박해지는 현상 완화
- 활성 경로가 다음 tick에서 갈 곳이 없어서 끊기는 문제 완화

## C. 멈춤 조건을 늦추기

현재 `run_until_converged()`는 `delta_k < delta_k_eps`가 되면 바로 멈춘다. current setting에서는 매우 이른 tick에서 정지할 수 있다.

권장 수정:

- `min_ticks_before_converged` 같은 하한 추가
- 예: 최소 4 또는 6 tick 전에는 조기 종료 금지
- `max_ticks`도 32에서 48 또는 64까지 시험

의도:

- 초기 작은 흔들림으로 바로 수렴 판정을 내리지 않게 만들기

## D. dominant branch 점수식을 길이 친화적으로 바꾸기

사용자는 dominant branch를 single best path로 두길 원했고, 이 방향 자체는 유지 가능하다. 다만 현재 `score = sum(K)` 성격이라면 짧고 강한 마지막 spike가 길고 중간 강도의 경로를 쉽게 이긴다.

권장 수정:

- single best path는 유지
- path score를 `sum(K) + lambda * path_length` 형태로 보정
- 또는 `mean(K) + lambda * log(length)` 형태를 시험

의도:

- "짧고 강한 spike"와 "지속된 중간 강도 경로" 사이의 균형 회복

주의:

- 이건 branch 길이를 인위적으로 부풀리는 게 아니라, "지속성" 자체를 score에 반영하는 것이다.

## E. survivor-only pruning을 완화하기

현재는 마지막 tick survivor만 역추적한다. 이 방식은 길이가 긴 earlier path보다 마지막 isolated node를 우선시할 수 있다.

대안:

1. final tick만 보지 말고 recent window 안의 end node 후보를 모두 평가
2. end tick 자유형 best-path 검색으로 변경
3. "가장 늦게 끝난 경로"가 아니라 "가장 강하고 오래 유지된 경로"를 선택

이 변경은 dominant branch 정의를 깨지 않고도 collapse를 줄일 수 있다.

## F. branch에 대한 직접 목적함수 추가

장기적으로는 branch가 자라도록 구조적으로 유도하는 항이 필요하다.

예:

- alive transition reward
- sustained activation reward
- single-tick collapse penalty

즉, branch length를 결과 지표가 아니라 학습/탐색 목표에 포함시키는 방식이다.

## 바로 할 실험 우선순위

### 1순위: 파라미터만 바꾸는 빠른 sweep

아래 세트부터 본다.

- `refractory_ticks = 1`
- `k_threshold_base = 0.8`
- `k_decay = 0.98`
- `mela_dropout_gain = 0.08`
- `sero_prune_gain = 0.08`
- `dopa_rewire_gain = 0.60`
- `memory_k_mix = 0.20`

측정:

- mean branch length
- length=1 비율
- max branch length
- branch length와 generation 품질의 상관

### 2순위: 조기 종료 방지

- 최소 tick 4 또는 6 추가
- `max_ticks` 48 또는 64 실험

### 3순위: extraction 변경

- final survivor only 방식과 recent-window end-point 방식 비교
- single best path는 유지하되 length-aware score 추가

### 4순위: learned encoder 재학습

branch가 실제로 길어진 설정으로 `export-z -> label-local -> fit-z-encoder`를 다시 돌려 본다.

## 논문에 쓸 수 있는 해석

branch length 1이 많다는 사실은 연구가 무의미하다는 뜻이 아니다. 오히려 current pipeline은 문제를 정량적으로 드러냈다. 즉, EmoNet의 현재 병목은 "branch라는 개념이 쓸모없다"가 아니라, dynamics와 extraction rule이 지속 경로보다 단발 spike를 과도하게 선호한다는 데 있다. 따라서 다음 단계의 핵심은 branch 자체를 버리는 것이 아니라, branch가 실제로 유지될 수 있도록 동역학 파라미터와 path selection criterion을 다시 설계하는 것이다.

## 빠른 로컬 검증

로컬에서 training JSON 100개 샘플로 old-like 설정과 new default를 비교했을 때 아래 변화가 관찰되었다.

- old-like: mean length 1.07, `len=1` 99/100, max length 8
- new default: mean length 1.79, `len=1` 93/100, max length 20

즉, 이번 수정은 적어도 소규모 샘플 기준으로는 branch collapse를 완화하는 방향으로 작동한다. 다만 전체 51,628 rows에 대한 재export로 최종 분포를 다시 확인해야 한다.

추가로 `out_z_training_extended40_branchfix.csv` 앞 50개 샘플에서 기존 branchfix-like 설정과 더 강한 persistence 설정을 비교했을 때도 아래 변화가 확인되었다.

- branchfix-like: mean length 2.36, `len=1` 44/50, max length 21
- stronger persistence: mean length 5.08, `len=1` 36/50, max length 31

즉, branch 길이는 extraction 보정만으로 끝나는 문제가 아니라 dynamics 기본값을 더 오래 버티게 할 때 추가 개선 여지가 실제로 있다.
