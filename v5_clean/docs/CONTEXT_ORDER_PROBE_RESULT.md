# EmoNet v5 Clean Context-Order Probe Result

검증일: 2026-08-09 KST

## 실행

- branch: `feature/v5-clean-trace-rebuild`
- verified head commit: `42792ec844f34cda235c1c58931a180a61e61260`
- GitHub Actions workflow: `v5-clean-ci`
- workflow run id: `31264669649`
- job id: `93120771766`
- artifact: `v5-clean-context-order-probe`
- artifact id: `9023788804`
- Python: 3.11
- encoder: deterministic `HashingTextEncoder`
- recurrent seeds: `7, 13, 21, 42, 100`
- paired contexts: 80
- train pairs: 60
- held-out test pairs: 20
- samples per seed: 160
- probe: deterministic ridge binary linear probe, alpha `1.0`

## 연구 질문

이번 실험은 단순히 history가 trace를 바꾸는지를 넘어서 다음을 묻는다.

> 동일한 사건 집합을 서로 다른 순서로 경험했을 때, 현재 입력이 완전히 같아도 recurrent trace에서 과거 사건의 **순서**를 복원할 수 있는가?

각 pair는 아래 두 arm을 가진다.

```text
prefix -> ALPHA -> bridge -> BETA -> suffix -> SAME_CURRENT
prefix -> BETA  -> bridge -> ALPHA -> suffix -> SAME_CURRENT
```

두 arm은:

- 같은 event multiset
- 같은 prefix / bridge / suffix
- 같은 마지막 history event
- 같은 current text

를 사용한다. 따라서 current text, context bag, last-history-event는 label을 구분할 수 없도록 설계되어 있다.

label은 단순히 `ALPHA-before-BETA`와 `BETA-before-ALPHA`를 구분한다. 감정/valence/arousal/appraisal label은 사용하지 않는다.

## 평가 방식

main linear probe는 **real trace train set으로만 학습**한다.

동일한 probe를 held-out test set의 다음 condition에 그대로 적용한다.

- real trace
- temporally shuffled trace
- wrong-pair trace
- reset-history trace
- order-erased trace

즉 control condition마다 probe를 다시 학습하지 않는다.

별도 baseline probe:

- current text only
- context bag
- last history event
- real trace only

## 결과

### Mean accuracy across 5 recurrent seeds

| Condition | Accuracy |
| --- | ---: |
| current text only | 0.500 |
| context bag | 0.500 |
| last history event | 0.500 |
| real trace only | **0.835** |
| text + real trace | **0.835** |
| text + temporal-shuffled trace | 0.540 |
| text + wrong trace | 0.165 |
| text + reset trace | 0.500 |
| text + order-erased trace | 0.500 |

### Seed-wise real-trace accuracy

| Seed | Real | Temporal shuffle | Wrong | Reset | Order erased |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 1.000 | 0.450 | 0.000 | 0.500 | 0.500 |
| 13 | 0.750 | 0.575 | 0.250 | 0.500 | 0.500 |
| 21 | 0.750 | 0.575 | 0.250 | 0.500 | 0.500 |
| 42 | 0.750 | 0.525 | 0.250 | 0.500 | 0.500 |
| 100 | 0.925 | 0.575 | 0.075 | 0.500 | 0.500 |

real trace mean/std:

```text
0.835 ± 0.1068
```

모든 seed에서 real trace accuracy는 0.65를 넘었다.

### Gaps

```text
real - text only       = +0.335
real - temporal shuffle = +0.295
real - wrong trace      = +0.670
real - reset            = +0.335
real - order erased     = +0.335
```

### Acceptance

```text
real_above_0_80_mean = true
every_seed_real_above_0_65 = true
text_only_near_chance = true
context_bag_near_chance = true
last_event_near_chance = true
real_beats_wrong_by_0_20 = true
real_beats_reset_by_0_20 = true
real_beats_order_erased_by_0_20 = true
all_primary_gates = true
```

## 해석

이번 결과는 이전 smoke보다 한 단계 강하다.

이전 결과가 보여준 것은:

> history가 recurrent state에 잔존한다.

였다.

이번 결과가 추가로 보여주는 것은:

> 같은 사건들의 집합과 같은 현재 입력을 사용하더라도, 과거 사건의 순서에 관한 정보가 recurrent raw trace에 남아 있으며, 단순 linear probe가 held-out pair에서 그 정보를 복원할 수 있다.

특히:

- context bag이 chance인 상태에서 real trace가 0.835
- history를 reset하면 chance
- 사건 순서를 canonicalize하여 지우면 chance
- 다른 arm의 trace를 넣으면 예측이 반대로 무너짐
- trace tick 순서를 섞으면 거의 chance

라는 패턴은 결과가 현재 text, event multiset, 마지막 event 또는 단순 trace 존재 여부만으로 설명되지 않는다는 것을 지지한다.

## Claim boundary

이 결과는 아직 semantic 또는 affect representation 증거가 아니다.

현재 방어 가능한 주장은 다음이다.

> fixed random recurrent EmoNet v5 baseline은 현재 입력에 없는 과거 사건의 temporal-order information을 raw trace에 보존하며, 이 정보는 held-out controlled pairs에서 선형적으로 decode 가능하다.

아직 주장하지 않는다.

- 자연어 의미 맥락을 올바르게 이해한다.
- trace가 emotion representation이다.
- affective history를 일반화한다.
- learned recurrent dynamics가 random reservoir보다 우월하다.
- EmoNet이 GRU/LSTM/ESN보다 우월하다.

## 다음 단계

다음 실험은 이 controlled result를 의미 맥락으로 확장한다.

1. 의미 임베딩을 고정한 natural-language paired context fixture
2. same-current / different-history 구조 유지
3. text-only / trace-only / real / wrong / reset 비교
4. recurrent random reservoir baseline과 GRU baseline 추가
5. emotion label은 core training에 넣지 않고 마지막 probe에서만 사용

이 순서로 가야 `temporal memory`와 `semantic/affective structure`를 분리해서 검증할 수 있다.
