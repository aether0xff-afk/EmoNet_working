# Softness Bias Mitigation (2026-04-06)

## 문제 요약

current `extended40` keep set 2,832 rows를 보면 스타일 타깃은 아래처럼 과도하게 온건한 방향으로 몰려 있다.

- 높은 축: plainness 0.9538, softness 0.9537, cooperativeness 0.9522, calmness 0.9480, positivity 0.9431
- 낮은 축: hostility 0.0001, resentment 0.0001, shame 0.0008, volatility 0.0012, despair 0.0020, fearfulness 0.0046

즉, 현재 문제는 "모델이 감정을 전혀 못 배운다"기보다, 학습 대상 `s` 자체가 지나치게 안전하고 부드러운 방향으로 수렴한다는 데 있다.

## 원인 가설

### 1. 생성-자가평가 루프의 안전 편향

현재 라벨링 절차는 모델이 먼저 응답을 생성하고, 다시 그 응답을 읽고 `s`와 `s_hat`를 평가한다. 이 구조는 사회적으로 무난한 응답을 만들수록 자기일관성이 높게 나오기 쉽다.

### 2. keep 규칙이 평균 L1 위주로만 작동

현재 keep 규칙은 `mean(abs(s - s_hat))` 중심이라서, 강한 감정이 살아 있는 샘플보다 "무난하고 다시 재평가해도 비슷하게 나오는 샘플"을 통과시키기 쉽다.

### 3. prompt surface의 완화 압력

현재 prompt는 이전보다 간결해졌지만, 여전히 LLM은 기본적으로 공손하고 협조적인 응답을 선호한다. raw affect 축이 낮으면 최종 응답은 더 쉽게 위로/차분함 쪽으로 미끄러진다.

### 4. 데이터 샘플링의 감정 강도 불균형

입력 데이터가 balanced label subset이어도, raw affect intensity가 balanced하다는 뜻은 아니다. 특히 공격성, 원망, 절망, 불안정성과 같은 축은 실제로는 매우 희소하게 남는다.

## 해결 전략

### A. 라벨링 단계 수정

가장 먼저 손대야 할 부분이다.

1. 생성 응답 기반 자기평가만 쓰지 말고, 입력 텍스트 자체에 대한 목표 스타일 평가를 별도로 수집한다.
2. `s`와 `s_hat` 외에 `affect_intensity` 보조 점수를 추가해, 감정 강도가 낮은데 consistency만 높은 샘플을 구분한다.
3. pairwise/contrastive labeling을 넣는다.
   - 예: 같은 입력에 대해 `soft response`와 `sharp response`를 둘 다 만들고 어느 쪽이 더 상황에 맞는지 평가
4. raw affect 축에 대해서는 axis-specific question을 분리한다.
   - 예: "이 응답이 공격적인가?"가 아니라 "이 상황에서 억눌린 분노/원망/절망이 남아 있는가?"처럼 묻는다.

### B. keep 규칙 수정

현재 keep 규칙은 너무 순하다.

1. 전체 L1 하나만 보지 말고 raw affect 축 subset L1을 별도로 계산한다.
2. 아래 조건을 동시에 요구하는 다중 keep 규칙으로 바꾼다.
   - 전체 consistency L1
   - raw affect consistency L1
   - 최소 감정 강도
3. 지나치게 평균적인 샘플은 downweight한다.
4. rare raw affect 축이 일정 수준 이상인 샘플은 threshold를 조금 완화해 더 많이 살린다.

### C. 데이터 샘플링 수정

라벨링 전에 hard case를 더 많이 집어넣어야 한다.

1. input text에서 high arousal / negative affect 후보를 먼저 mining한다.
2. 감정 라벨 기반 balanced sampling 외에 아래 strata를 추가한다.
   - conflict
   - betrayal
   - shame
   - helplessness
   - panic
   - resentment
3. keep set 재학습 시 rare-axis 활성 샘플을 oversampling한다.

### D. 모델 및 손실 수정

학습 손실이 평균 축에만 끌려가지 않게 해야 한다.

1. `z -> s` 학습에서 rare axis weighted loss를 사용한다.
2. macro social tone 축과 raw affect 축을 분리한 multi-head decoder를 실험한다.
3. learned `z` encoder 학습 시 raw affect 예측 오차를 더 크게 penalize한다.
4. mean MAE만 보지 말고 axis-wise calibration을 함께 본다.

### E. prompt surface 수정

현재 응답이 부드러워지는 현상은 prompt 단계에서도 제어해야 한다.

1. raw affect 축이 높을 때는 positivity/calmness/softness를 자동 완화하는 규칙을 넣는다.
2. 아래와 같은 anti-softening rule을 조건부로 추가한다.
   - 필요 이상으로 위로하지 말 것
   - 지나치게 공손하거나 순화된 표현으로 감정을 희석하지 말 것
   - 불편함, 분노, 절망이 남아 있으면 그 톤을 보존할 것
3. style tags를 단순 top-k가 아니라 conflict-preserving tag set으로 바꾸는 실험을 한다.

## 바로 할 실험 우선순위

### 1순위

현재 라벨 CSV에서 rare-axis가 살아 있는 샘플만 따로 추려 소규모 재학습을 해본다.

목적:
- softness bias가 모델 문제인지 target 문제인지 빠르게 분리

### 2순위

label-local prompt를 두 갈래로 나눈다.

- target style labeling
- generated response self-rating

목적:
- 생성 안전 편향과 target annotation 편향 분리

### 3순위

generate-response에 anti-softening 조건부 규칙을 추가하고 A/B test를 한다.

비교:
- current prompt
- current prompt + anti-softening rule

### 4순위

predictor 평가를 overall MAE 하나가 아니라 axis-wise 표로 바꾼다.

목적:
- 실제로 어떤 축이 무너지는지 확인

## 논문에 쓸 수 있는 해석

현재 응답이 부드러움 쪽에 치우친 것은 연구 전체가 실패했다는 뜻이 아니다. 오히려 current cycle은 문제가 어디서 생기는지 더 명확히 드러냈다. 핵심은 EmoNet이 아직 "강한 감정을 안전하게 희석하는 경향"을 보이며, 이 현상은 branch collapse뿐 아니라 style target construction과 prompt surface 설계가 동시에 만드는 문제라는 점이다. 따라서 다음 단계의 성패는 축 수를 더 늘리는 데 있지 않고, raw affect를 실제로 살아남게 하는 라벨링 규칙과 손실, prompt 제약을 함께 설계하는 데 달려 있다.
