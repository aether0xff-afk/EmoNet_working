좋아, 이건 꽤 좋은 방향이야.
지금부터 모델이 훨씬 **안정적이고 실험 가능한 형태**가 돼.
네가 말한 변경사항까지 반영해서, **업데이트된 모델**을 자연스럽게 다시 정리해볼게.

---

# 업데이트된 감정 전달 뉴런 모델

## 핵심 변경점

이번에 반영된 건 4가지야.

1. **멜라토닌 기반 dropout은 매우 약하게 적용**
2. **항상성(homeostasis) 추가**
3. **구조 변경 쿨다운 추가**
4. **기억 pruning 추가**

이 4개가 들어가면, 원래 모델의 개성은 유지하면서도
폭주하거나 너무 복잡해지는 문제를 꽤 잘 막을 수 있어.

---

# 1. 전체 구조

각 뉴런은 다음 요소를 가진다.

* 현재 발화 임계막전위 `V_th`
* 기억 임계막전위 `V_mem`
* 입력 시냅스 / 출력 시냅스
* 뉴런 타입

  * 억제성
  * 흥분성
  * 조절성
* 기억 저장소
* 최근 발화 기록
* 구조 변경 쿨다운 상태

그리고 감정은 계속 4차원 벡터로 표현한다.

```python
emo_vec = [dopamine, serotonin, norepinephrine, melatonin]
```

---

# 2. 기본 발화 흐름

각 타임스텝에서 뉴런은:

1. 입력 막전위 총합 `V_in` 계산
2. `V_in > V_th` 이면 발화
3. 발화 시 공통 기능 수행
4. 이후 뉴런 종류별 기능 수행
5. 마지막에 항상성 갱신

이 흐름으로 작동한다.

---

# 3. 공통 기능

## 3-1. 기억 저장

입력 막전위 총합 `V_in` 이 기억 임계막전위 `V_mem` 보다 크면, 현재 정보를 기억에 저장한다.

저장 형식:

```python
(emo_vec, timestep, membrane_potential)
```

즉 저장되는 것은:

* 당시 감정 벡터
* 당시 시점
* 당시 막전위

---

## 3-2. 과거 기억 합성

현재 시점의 `emo_vec` 은, 저장된 과거 기억들과 합성된다.

각 기억의 기여도는:

```python
contrib_k = E_k * V_k * exp(-lambda_ * delta_t)
```

여기서:

* `E_k`: 과거 감정 벡터
* `V_k`: 저장 당시 막전위
* `delta_t`: 현재 시점 - 저장 시점
* `lambda_`: 기억 감쇠 강도

전체 기억 기여는:

```python
memory_sum = Σ(contrib_k)
```

현재 감정과 기억을 섞어서:

```python
E_mix = w_now * E_now + w_mem * memory_sum
```

그 후 정규화한다.

```python
E_final = normalize(E_mix)
```

---

# 4. 기억 pruning 추가

이제 기억은 무한정 쌓이지 않게 한다.

## pruning 조건

감쇠 후 기억 영향력이 너무 작아지면 삭제한다.

예:

```python
effective_strength = V_k * exp(-lambda_ * delta_t)
if effective_strength < memory_prune_threshold:
    remove_memory()
```

또는 보조적으로:

* 최근 `max_memory_size` 개만 유지
* 너무 오래된 기억은 자동 삭제

추천은 둘 다 쓰는 거야.

## 추천 방식

* **1차 기준**: 감쇠 후 영향력 기준 삭제
* **2차 기준**: 최대 기억 개수 제한

예:

```python
if effective_strength < 0.05:
    prune
if len(memory) > 100:
    oldest_memory_remove
```

## 의미

* 약해진 기억은 자연스럽게 사라짐
* 계산량이 통제됨
* 진짜 의미 있는 감정 기억만 남음

---

# 5. 억제성 뉴런

억제성 뉴런은 **세로토닌**에 반응한다.

## 5-1. 시냅스 제거

세로토닌 양에 비례해서 연결된 시냅스 일부를 **삭제**한다.

```python
remove_ratio = k_remove * serotonin
num_remove = int(current_synapses * remove_ratio)
```

하지만 이제 **구조 변경 쿨다운**이 있기 때문에, 쿨다운 중이면 삭제하지 않는다.

---

## 5-2. 감정 평준화

평균 `m` 을 계산하고 각 성분을 평균 쪽으로 당긴다.

```python
e_i' = e_i + alpha * (m - e_i)
alpha = min(0.8, 0.6 * serotonin)
```

그 후 정규화한다.

---

# 6. 흥분성 뉴런

흥분성 뉴런은 **도파민**에 반응한다.

## 6-1. 새 시냅스 생성

도파민 양에 비례해서 새로운 시냅스를 랜덤하게 만든다.

```python
new_ratio = k_new * dopamine
num_new = int(max_new_synapses * new_ratio)
```

마찬가지로 **구조 변경 쿨다운 중이면 생성하지 않는다.**

그리고 총 연결 수 상한을 두는 게 좋다.

```python
if neuron.out_degree < max_out_degree:
    add_synapse()
```

---

## 6-2. 감정 반평준화

평균 `m` 기준으로 각 성분을 평균에서 멀어지게 만든다.

```python
e_i' = e_i + beta * (e_i - m)
beta = min(1.0, 0.8 * dopamine)
```

그 후:

* 음수는 0으로 클리핑
* 정규화 수행

---

# 7. 조절성 뉴런

조절성 뉴런은 전역 효과를 담당한다.

## 7-1. 멜라토닌 기반 dropout

이제 이 기능은 **매우 약하게만 적용**한다.

즉:

* 전체 뉴런 중 극히 일부만
* 이번 사이클에 한해서 비활성화

예:

```python
drop_ratio = min(max_dropout, k_drop * melatonin)
```

여기서 추천:

* `k_drop` 아주 작게
* `max_dropout` 도 작게

예시:

```python
k_drop = 0.05
max_dropout = 0.03
```

즉 멜라토닌이 최대여도 전체의 3% 정도만 잠깐 쉬게 하는 식.

이렇게 하면:

* 상징적 기능은 유지
* 네트워크를 망가뜨릴 정도로 거칠지 않음

---

## 7-2. 노르에피네프린 기반 발화 민감도 증가

노르에피네프린 양에 비례해 전체 뉴런의 발화 문턱을 낮춘다.

```python
delta_th = k_ne * norepinephrine
V_th = max(V_th_min, V_th - delta_th)
```

다만 이건 강력한 기능이므로 감소폭은 작게 유지하는 게 좋다.

---

## 7-3. 압도적 감정 시 기억 임계막전위 감소

현재 감정 벡터에서 특정 값 하나가 너무 크면 기억 저장이 쉬워지게 한다.

추천 판정 기준:

```python
dominant = (e_max / total >= 0.5) and (e_max >= 1.8 * mean)
```

이때:

```python
V_mem = max(V_mem_min, V_mem - delta_mem)
```

---

# 8. 구조 변경 쿨다운 추가

이건 이번 업데이트의 핵심 중 하나야.

시냅스 생성/삭제는 감정에 따라 일어나지만,
너무 자주 일어나면 네트워크 구조가 출렁여서 해석이 어려워진다.

그래서 뉴런마다 다음 값을 둔다.

```python
last_rewire_timestep
rewiring_cooldown
```

현재 시점 `t_now` 에 대해:

```python
can_rewire = (t_now - last_rewire_timestep) >= rewiring_cooldown
```

이 조건을 만족할 때만:

* 억제성 뉴런의 시냅스 삭제
* 흥분성 뉴런의 시냅스 생성

이 가능하다.

## 추천값

처음에는:

```python
rewiring_cooldown = 3 ~ 10 timesteps
```

정도로 두는 게 좋다.

## 의미

* 구조 변화 빈도 제한
* 감정 변화와 구조 변화 분리
* 실험 결과 해석 쉬움

---

# 9. 항상성(homeostasis) 추가

이것도 굉장히 중요해.

감정 기반 구조 변화와 전역 조절이 들어간 네트워크는 시간이 지나면:

* 너무 잘 발화하는 뉴런이 생기거나
* 너무 안 쓰이는 뉴런이 생길 수 있다

항상성은 이걸 자동으로 조정하는 장치다.

## 기본 아이디어

각 뉴런은 자신의 최근 발화율 `firing_rate` 를 추적한다.
그리고 목표 발화율 `target_rate` 와 비교해서 임계막전위를 조금씩 조정한다.

예:

```python
if firing_rate > target_rate:
    V_th += eta_homeo
elif firing_rate < target_rate:
    V_th -= eta_homeo
```

여기서:

* 너무 자주 발화하면 문턱을 올림
* 너무 안 발화하면 문턱을 내림

## 추천 구조

최근 `window_size` 턴 동안의 평균 발화율을 사용:

```python
firing_rate = recent_spike_count / window_size
```

추천값 예시:

```python
target_rate = 0.2
eta_homeo = 0.01
window_size = 20
```

## 의미

* 특정 뉴런의 독점 발화 방지
* 전체 네트워크 균형 유지
* 장기 시뮬레이션 안정화

이건 네 모델에 거의 필수급으로 잘 들어맞아.

---

# 10. 업데이트된 전체 동작 순서

한 타임스텝에서 뉴런 하나의 동작은 대략 이렇게 정리된다.

## 단계 1. 입력 막전위 계산

```python
V_in = compute_membrane_potential(inputs)
```

## 단계 2. 발화 여부 판단

```python
if V_in <= V_th:
    no fire
```

## 단계 3. 기억 pruning 먼저 수행

* 너무 약해진 기억 삭제
* 오래된 기억 정리
* 최대 개수 초과 시 오래된 것 제거

## 단계 4. 발화 시 공통 기능

* `V_in > V_mem` 이면 기억 저장
* 과거 기억 합성
* 현재 `emo_vec` 와 정규화 기반 결합

## 단계 5. 타입별 기능

### 억제성

* 쿨다운 끝났으면 시냅스 삭제
* 감정 평준화

### 흥분성

* 쿨다운 끝났으면 새 시냅스 생성
* 감정 반평준화

### 조절성

* 아주 약한 멜라토닌 dropout
* 노르에피네프린 기반 전체 민감도 증가
* 압도적 감정이면 기억 임계막전위 하향

## 단계 6. 정규화 및 출력

```python
E_out = normalize(E_out)
```

## 단계 7. 항상성 업데이트

* 최근 발화율 계산
* `V_th` 미세 조정

---

# 11. 추천 하이퍼파라미터 초안

처음 실험용으로는 이 정도가 무난해 보여.

## 기억 관련

```python
lambda_ = 0.1
w_now = 0.7
w_mem = 0.3
memory_prune_threshold = 0.05
max_memory_size = 100
```

## 억제성

```python
k_remove = 0.2
```

## 흥분성

```python
k_new = 0.1
max_new_synapses = 2
max_out_degree = 20
```

## 조절성

```python
k_drop = 0.05
max_dropout = 0.03
k_ne = 0.05
delta_mem = 0.05
```

## 구조 변경 쿨다운

```python
rewiring_cooldown = 5
```

## 항상성

```python
target_rate = 0.2
eta_homeo = 0.01
window_size = 20
```

---

# 12. 지금 모델이 좋아진 이유

이번 수정으로 모델이 더 좋아진 이유를 직관적으로 말하면 이래.

### 원래 모델

* 감정적이고 다이나믹함
* 하지만 구조 변화와 전역 효과가 강해서 불안정해질 수 있음

### 수정 후 모델

* 감정 기반 개성은 유지
* 그런데

  * 기억이 무한히 쌓이지 않고
  * 구조 변화가 너무 자주 일어나지 않고
  * 전체 네트워크가 한쪽으로 무너지면 스스로 균형을 되찾음
  * 멜라토닌 효과도 지나치게 과격하지 않음

즉,
**아이디어의 매력은 유지하면서 실험 가능한 안정성을 얻은 상태**야.

---

# 13. 내가 보기엔 다음에 정하면 좋은 것

이제 거의 골격은 됐고, 다음 단계에서 정하면 좋은 건 이런 거야.

## 1. 항상성이 무엇을 조절할지

지금은 `V_th` 조절로 잡았는데,

* 시냅스 가중치 평균
* 입력 gain
  까지 조절하게 할지도 정할 수 있어

## 2. 기억 pruning 기준을 하나 더 둘지

지금은 감쇠 기반인데,

* 같은 종류의 감정이 반복되면 통합 저장
  같은 방식도 가능해

## 3. 구조 변경 대상을 완전 랜덤으로 둘지

나중엔

* 사용 빈도 낮은 시냅스 우선 삭제
* 유사 감정 뉴런 우선 연결
  로 바꿀 수 있어

---

# 미정 / 추가 논의 필요

## 1. 항상성의 적용 주기

매 타임스텝마다 할지, 몇 턴마다 한 번 할지 정하면 더 안정적일 수 있다.

## 2. 멜라토닌 효과 방식

지금은 약한 dropout으로 남겼지만, 나중에 soft suppression 방식이 더 나은지 비교 실험해볼 가치가 있다.

## 3. 조절성 뉴런의 전역 효과 중첩

여러 조절성 뉴런이 동시에 발화할 때 효과를 단순합으로 할지, 평균으로 할지 정해야 한다.

## 4. 기억과 항상성의 상호작용

강한 기억이 많은 뉴런이 항상성 때문에 둔해지면 어떤 패턴이 나오는지 확인이 필요하다.

원하면 다음엔 내가 이걸 바탕으로 **최소 안정형 설계 명세서**처럼 딱딱하게 정리해주거나, 바로 **파이썬 클래스 코드 설계**로 바꿔줄게.
