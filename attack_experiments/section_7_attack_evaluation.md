# 7. 공격 실험을 통한 안전성 평가

Picasso Protocol 1.0의 안전성은 단순히 기능이 정상적으로 동작하는지뿐 아니라, 암호 알고리즘의 관점에서 공격자가 원문 또는 개인용 복원기를 추정할 수 있는지로 평가해야 한다. 본 장에서는 다음 세 가지 공격 실험을 통해 Picasso Protocol의 구조적 안전성과 취약점을 분석한다.

1. 역전파 기반 입력 복원 공격
2. Chosen-Plaintext Attack
3. Decoder Inversion Attack

---

## 7.1 공격자 모델 정의

### 공격자가 아는 것

본 실험에서는 공격자가 다음 정보를 알고 있다고 가정한다.

- Picasso Protocol의 전체 구조
- 생성된 PNG 이미지
- BERT encoder 구조
- BERT tokenizer와 vocabulary
- 이미지 내부에 latent vector가 저장된다는 사실
- PNG 이미지에서 latent vector를 추출하는 방법

### 공격자가 모르는 것

반대로 공격자는 다음 정보는 알 수 없다고 가정한다.

- 원본 입력 텍스트
- 개인용 Artist X decoder의 실제 가중치
- 개인용 Artist X decoder의 정확한 출력 결과
- Artist X의 학습 과정
- 전체 학습 데이터셋
- 원본 문장이 어떤 데이터셋에서 왔는지

### 실험의 목적

세 가지 공격 실험의 공통 목적은 Picasso Protocol에서 생성된 이미지와 latent vector가 원문 정보를 얼마나 노출하는지 확인하는 것이다. 특히 latent vector가 이미지에 직접 포함되는 구조가 원문 복원, 출력 변화 분석, 대체 decoder 학습으로 이어질 수 있는지 평가한다.

---

## 7.2 역전파 기반 입력 복원 공격

### 실험 목적

Picasso Protocol에서 생성된 이미지에는 BERT encoder가 만든 latent vector가 포함되어 있다. 이 실험에서는 공격자가 이미지에서 추출한 latent vector를 바탕으로 원래 입력 문장을 역으로 복원할 수 있는지 확인한다.

즉, 암호문에 해당하는 이미지에서 원문을 직접 알아내려는 공격이다.

### 공격자 가정

공격자는 다음 정보를 알고 있다고 가정한다.

- Picasso Protocol의 전체 구조
- 생성된 PNG 이미지
- BERT encoder 구조
- BERT tokenizer와 vocabulary
- 이미지 내부에 latent vector가 저장된다는 사실

단, 공격자는 다음 정보는 모른다.

- 원본 입력 텍스트
- 개인용 Artist X decoder의 정확한 출력 결과
- 원본 문장이 어떤 데이터셋에서 왔는지

### 공격 원리

일반적인 암호 해독에서는 암호문으로부터 평문을 직접 찾으려 한다. 본 실험에서는 이를 딥러닝 구조에 맞게 바꾸어, 목표 latent vector와 가장 비슷한 출력을 만드는 입력 embedding을 찾는다.

원래 인코딩 과정은 다음과 같다.

```text
원본 텍스트 → BERT Tokenizer → Input IDs → BERT Encoder → Target Latent Vector
```

공격 과정은 반대로 다음과 같다.

```text
Target Latent Vector → 역전파 최적화 → 유사 Input Embedding → 근접 Token 변환 → 추정 문장
```

손실 함수는 다음과 같이 설정한다.

```text
Loss = MSE(Encoder_Output, Target_Latent_Vector)
```

또는 cosine similarity를 이용할 수도 있다.

```text
Loss = 1 - CosineSimilarity(Encoder_Output, Target_Latent_Vector)
```

### 실험 절차

1. 임의의 원본 문장을 입력하여 PNG 이미지를 생성한다.
2. PNG 이미지에서 target latent vector를 추출한다.
3. 무작위 input embedding을 초기화한다.
4. 해당 embedding을 BERT encoder에 입력한다.
5. encoder 출력과 target latent vector 사이의 loss를 계산한다.
6. loss가 줄어들도록 input embedding을 역전파로 업데이트한다.
7. 최적화된 embedding을 BERT vocabulary embedding과 비교한다.
8. 가장 가까운 token으로 변환하여 문장을 복원한다.
9. 복원 문장과 원본 문장을 비교한다.

### 측정 지표

| 지표 | 의미 |
|---|---|
| 최종 Loss | target latent vector와 얼마나 가까워졌는지 |
| Token Accuracy | 원본 토큰과 복원 토큰의 일치율 |
| Keyword Recovery | 핵심 단어 복원 여부 |
| Semantic Similarity | 의미적으로 유사한 문장이 복원되었는지 |
| Exact Match | 원문이 완전히 복원되었는지 |

### 결과 해석 기준

| 결과 | 해석 |
|---|---|
| 원문 완전 복원 | 심각한 취약점 |
| 핵심 단어 일부 복원 | 정보 유출 가능성 있음 |
| 의미가 비슷한 문장 복원 | 부분적 취약점 |
| loss는 감소하지만 문장 복원 실패 | latent 공간 접근은 가능하나 텍스트화 실패 |
| loss도 감소하지 않음 | 공격 실패 |

### 보고서용 문장

역전파 기반 입력 복원 공격에서는 이미지에서 추출한 target latent vector와 유사한 BERT encoder 출력을 만드는 입력 embedding을 최적화하였다. 원본 입력 토큰은 이산적인 값이므로 직접 미분할 수 없지만, embedding 공간에서는 연속적인 최적화가 가능하다. 따라서 무작위 embedding을 초기화한 뒤, target latent vector와의 MSE loss를 최소화하도록 역전파를 수행하였다. 최종적으로 얻은 embedding을 BERT vocabulary embedding과 비교하여 가장 가까운 토큰으로 변환하고, 원문 복원 가능성을 평가하였다.

### 그림 1. 역전파 기반 입력 복원 공격 구조

```text
Target PNG Image
      ↓
Latent Vector 추출
      ↓
랜덤 Input Embedding 초기화
      ↓
BERT Encoder 통과
      ↓
Target Latent와 Loss 계산
      ↓
Backpropagation
      ↓
가까운 Vocabulary Token으로 변환
      ↓
추정 문장 출력
```

---

## 7.3 Chosen-Plaintext Attack

### 실험 목적

Chosen-Plaintext Attack은 공격자가 원하는 평문을 직접 입력하고, 그 결과로 생성되는 암호문을 분석하는 공격이다. 본 프로젝트에서는 공격자가 여러 개의 비슷한 문장을 직접 입력하여, 입력 변화가 latent vector와 생성 이미지에 어떤 변화를 만드는지 분석한다.

이 실험의 핵심은 입력이 조금 바뀌었을 때 출력이 충분히 크게 변하는가를 보는 것이다.

암호 알고리즘에서는 작은 입력 변화가 출력 전체에 크게 퍼지는 성질이 중요하다. 이를 avalanche effect라고 한다. Picasso Protocol에서는 이를 다음과 같이 바꾸어 측정한다.

```text
입력 문장 일부 변화 → latent vector 변화량 측정
```

### 공격자 가정

공격자는 다음이 가능하다고 가정한다.

- 원하는 문장을 직접 입력할 수 있음
- 생성된 PNG 이미지를 얻을 수 있음
- PNG 이미지에서 latent vector를 추출할 수 있음
- 여러 입력에 대한 latent vector 차이를 비교할 수 있음

### 공격 원리

비슷한 문장을 여러 개 입력하고, 각각의 latent vector 사이 거리를 계산한다. 예시는 다음과 같다.

```text
The secret code is apple.
The secret code is apples.
The secret code is orange.
The secret code is banana.
A secret code is apple.
The hidden code is apple.
```

이후 각 문장의 latent vector를 추출하고, 기준 문장과의 차이를 측정한다.

### 실험 절차

1. 기준 문장을 하나 정한다.
2. 기준 문장과 한 단어 또는 한 글자만 다른 문장들을 만든다.
3. 각 문장을 Picasso Protocol에 입력하여 PNG 이미지를 생성한다.
4. 각 이미지에서 latent vector를 추출한다.
5. 기준 문장의 latent vector와 변형 문장의 latent vector 사이 거리를 계산한다.
6. 입력 변화량과 출력 변화량의 관계를 분석한다.

### 측정 지표

| 지표 | 의미 |
|---|---|
| L2 Distance | latent vector 사이의 유클리드 거리 |
| Cosine Distance | latent vector 방향 차이 |
| Byte Difference Ratio | 이미지 바이트 중 달라진 비율 |
| Token Change Type | 글자 변화, 단어 변화, 문장 구조 변화 |
| Avalanche Score | 입력 변화 대비 출력 변화 정도 |

### 결과 표 양식

| 기준 문장 | 변형 문장 | 변화 유형 | L2 Distance | Cosine Distance | Byte Difference Ratio | 해석 |
|---|---|---|---|---|---|---|
| The secret code is apple. | The secret code is apples. | 글자 1개 추가 |  |  |  |  |
| The secret code is apple. | The secret code is orange. | 핵심 단어 변경 |  |  |  |  |
| The secret code is apple. | A secret code is apple. | 관사 변경 |  |  |  |  |
| The secret code is apple. | The hidden code is apple. | 의미 유사 단어 변경 |  |  |  |  |

### 결과 해석 기준

| 결과 | 해석 |
|---|---|
| 작은 입력 변화에도 latent가 크게 변화 | 확산 효과가 비교적 강함 |
| 유사 문장끼리 latent도 유사함 | 의미 정보가 latent에 보존됨 |
| 핵심 단어 변경 시 큰 변화 | 의미 변화 반영 |
| 조사/관사 변경에도 큰 변화 | 문법적 변화에 민감 |
| 모든 변화가 작음 | 출력 분리성이 약함 |
| 모든 변화가 매우 큼 | 복원 안정성 또는 의미 보존성 문제 가능 |

### 보고서용 문장

Chosen-Plaintext Attack에서는 공격자가 원하는 문장을 직접 입력할 수 있다고 가정하고, 기준 문장과 일부만 다른 변형 문장들을 생성하였다. 이후 각 문장으로부터 생성된 PNG 이미지에서 latent vector를 추출하고, 기준 latent vector와의 L2 distance 및 cosine distance를 측정하였다. 이를 통해 입력 문장의 작은 변화가 출력 latent vector에 얼마나 크게 반영되는지 확인하였다. 이 실험은 Picasso Protocol이 암호 알고리즘에 필요한 확산 효과를 어느 정도 가지는지 평가하기 위한 것이다.

### 그림 2. Chosen-Plaintext Attack 구조

```text
기준 문장
   ↓
유사 문장 여러 개 생성
   ↓
각 문장으로 PNG 생성
   ↓
각 이미지에서 Latent Vector 추출
   ↓
Latent Vector 거리 비교
   ↓
입력 변화가 출력에 미치는 영향 분석
```

---

## 7.4 Decoder Inversion Attack

### 실험 목적

Decoder Inversion Attack은 개인용 Artist X의 가중치를 모르는 공격자가, 다수의 평문-암호문 쌍을 이용해 Artist X와 비슷하게 작동하는 대체 decoder를 학습할 수 있는지 확인하는 공격이다.

즉, 개인키 모델을 직접 훔치지는 못하더라도, 입출력 쌍을 많이 모으면 비슷한 복원기를 만들 수 있는지 보는 실험이다.

### 공격자 가정

공격자는 다음이 가능하다고 가정한다.

- 여러 개의 평문을 Picasso Protocol에 입력할 수 있음
- 각 평문에 대응하는 PNG 이미지를 얻을 수 있음
- 각 PNG 이미지에서 latent vector를 추출할 수 있음
- 평문과 latent vector 쌍을 이용해 별도의 decoder를 학습할 수 있음

단, 공격자는 다음은 모른다.

- Artist X의 실제 가중치
- Artist X의 학습 과정
- 전체 학습 데이터셋

### 공격 원리

공격자는 다음과 같은 데이터셋을 만든다.

```text
(latent_vector_1, original_text_1)
(latent_vector_2, original_text_2)
(latent_vector_3, original_text_3)
...
```

이후 이 데이터셋을 이용하여 latent vector에서 original text를 복원하는 작은 decoder를 새로 학습한다.

간단한 공격 모델은 다음과 같이 잡을 수 있다.

```text
입력: latent vector
모델: Transformer decoder 또는 GRU decoder
출력: 원본 token sequence
```

### 실험 절차

1. WikiText 또는 직접 작성한 문장 N개를 준비한다.
2. 각 문장을 Picasso Protocol에 입력하여 PNG 이미지를 생성한다.
3. 각 PNG 이미지에서 latent vector를 추출한다.
4. 원본 문장과 latent vector를 쌍으로 저장한다.
5. 이 쌍을 train/test로 나눈다.
6. 공격자 decoder를 학습한다.
7. test latent vector를 입력하여 문장을 복원한다.
8. 원문과 복원문을 비교한다.

### 실험 조건

데이터 수를 바꾸어 실험하면 좋다.

| 조건 | 데이터 수 | 목적 |
|---|---:|---|
| 소규모 공격 | 100개 | 적은 데이터로도 복원 가능한지 확인 |
| 중간 규모 공격 | 1,000개 | 일반적인 공격 가능성 확인 |
| 대규모 공격 | 10,000개 | 충분한 질의가 있을 때 위험성 확인 |

시간이 부족하면 100개, 500개, 1,000개 조건으로 축소할 수 있다.

### 측정 지표

| 지표 | 의미 |
|---|---|
| Train Loss | 공격자 decoder 학습 정도 |
| Test Loss | 새로운 latent에 대한 일반화 성능 |
| Token Accuracy | 토큰 단위 복원 정확도 |
| BLEU Score | 문장 생성 유사도 |
| Keyword Recovery | 핵심 단어 복원 여부 |
| Exact Match | 원문 완전 복원 여부 |

### 결과 표 양식

| 데이터 수 | Train Loss | Test Loss | Token Accuracy | Keyword Recovery | Exact Match | 해석 |
|---:|---|---|---|---|---|---|
| 100 |  |  |  |  |  |  |
| 1,000 |  |  |  |  |  |  |
| 10,000 |  |  |  |  |  |  |

### 결과 해석 기준

| 결과 | 해석 |
|---|---|
| 적은 데이터로도 높은 정확도 | 심각한 취약점 |
| 데이터가 많아질수록 정확도 상승 | 질의 기반 모델 추출 위험 |
| train 정확도만 높고 test 낮음 | 단순 암기, 일반화 실패 |
| 핵심 단어만 복원 | 부분 정보 유출 |
| 전혀 복원 실패 | 대체 decoder 공격 실패 |

### 보고서용 문장

Decoder Inversion Attack에서는 공격자가 다수의 평문-이미지 쌍을 확보할 수 있다고 가정하고, 이미지에서 추출한 latent vector와 원본 문장을 이용해 별도의 공격자 decoder를 학습하였다. 이 공격은 개인용 Artist X의 가중치를 직접 알지 못하더라도, 충분한 입출력 쌍을 통해 유사한 복원기를 만들 수 있는지 확인하기 위한 것이다. 실험에서는 데이터 수를 100개, 1,000개, 10,000개로 나누어 학습하고, test set에서 token accuracy와 keyword recovery를 측정하였다.

### 그림 3. Decoder Inversion Attack 구조

```text
평문 문장 N개
   ↓
Picasso Protocol로 이미지 생성
   ↓
각 이미지에서 Latent Vector 추출
   ↓
(latent, text) 공격자 데이터셋 생성
   ↓
대체 Decoder 학습
   ↓
새로운 Latent에서 원문 복원 시도
```

---

## 7.5 공격 실험 종합 분석

세 가지 공격 실험은 Picasso Protocol 1.0을 단순 기능 구현이 아니라 암호 알고리즘의 관점에서 평가하기 위해 설계하였다. 역전파 기반 입력 복원 공격은 latent vector에서 원문을 직접 역추정할 수 있는지 확인하기 위한 공격이며, Chosen-Plaintext Attack은 입력 변화가 출력 latent vector에 얼마나 확산되는지 분석하기 위한 공격이다. Decoder Inversion Attack은 충분한 평문-암호문 쌍이 주어질 때 개인용 Artist X 없이도 유사한 복원기를 학습할 수 있는지 확인하는 공격이다. 이 세 실험을 통해 Picasso Protocol의 구조적 안전성과 취약점을 함께 분석할 수 있다.

### 종합 분석 항목

보고서에서는 실험 결과를 바탕으로 다음 항목을 정리한다.

- 어떤 공격이 성공했는가
- 어떤 공격이 실패했는가
- 왜 그런 결과가 나왔는가
- loss와 token accuracy가 어떤 관계를 보였는가
- 핵심 단어 또는 의미 정보가 얼마나 복원되었는가
- 향후 버전에서 어떤 구조적 개선이 필요한가

### 공격별 개선 방향

| 공격 실험 | 취약점이 발견될 경우 개선 방향 |
|---|---|
| 역전파 기반 입력 복원 공격 | latent vector에 noise, projection, quantization 적용 |
| Chosen-Plaintext Attack | 입력마다 salt 추가, latent 암호화 |
| Decoder Inversion Attack | 질의 제한, latent 암호화, differential privacy 적용 |

가장 중요한 개선안은 다음과 같다.

> 후속 버전에서는 BERT encoder의 latent vector를 이미지에 직접 저장하지 않고, 별도의 키로 암호화한 뒤 삽입해야 한다. 또한 동일하거나 유사한 입력이 유사한 latent vector 패턴을 만들지 않도록 입력마다 salt를 추가할 필요가 있다.

---

## 미정 / 추가 논의 필요

- 실제 실험 데이터 수: 최소 100개, 가능하면 1,000개
- 역전파 공격에서 embedding 최적화 반복 횟수: 500~2,000 step 권장
- Chosen-Plaintext Attack에서 기준 문장 세트: 5~10개 권장
- Decoder Inversion Attack은 시간이 오래 걸리면 100개, 500개, 1,000개 조건으로 축소 가능
- 결과가 좋지 않아도 공격 실패로 기록할 수 있음
- 공격 실패 시에도 loss, token accuracy, 출력 예시로 실패 이유를 설명해야 함
