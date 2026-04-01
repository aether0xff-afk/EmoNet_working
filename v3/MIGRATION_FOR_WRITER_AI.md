# Migration File For Writer AI

이 문서는 현재 `EmoNet_working/v3` 프로젝트를 다른 글쓰기 인공지능에게 넘길 때 필요한 맥락을 최대한 빠짐없이 전달하기 위한 상세 이관 문서다. 목적은 "이 프로젝트가 정확히 무엇을 구현했고, 무엇이 이미 검증되었고, 무엇은 아직 미완료이며, 논문에서는 무엇을 주장해도 되고 무엇은 아직 주장하면 안 되는지"를 분명히 전달하는 것이다.

이 문서를 읽는 AI는 코드 작성자가 아니라 "논문/보고서/제안서/연구문서 작성자" 역할을 가정한다. 따라서 아래 내용은 구현 세부와 연구 서술을 동시에 포함한다.

## 1. 프로젝트 기본 정보

- 프로젝트명: `EmoNet`
- 현재 작업 경로: `v3`
- 성격: 한국어 감정 대화 입력을 받아 neuro-inspired 감정 동역학 중간표현을 거쳐 스타일 조건부 응답을 생성하는 실험적 파이프라인
- 구현 언어: Python
- 주요 파일:
  - [core.py](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/emonet/core.py)
  - [cli.py](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/emonet/cli.py)
  - [test_emonet_smoke.py](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/tests/test_emonet_smoke.py)
  - [response_generation_prompt.md](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/prompts/response_generation_prompt.md)
  - [style_generation_prompt.md](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/prompts/style_generation_prompt.md)
  - [style_rating_prompt.md](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/prompts/style_rating_prompt.md)

## 2. 이 프로젝트가 하려는 일

이 프로젝트는 "사용자의 감정 상태에 맞는 한국어 응답을 생성"하는 시스템을 만들려는 시도다. 다만 일반적인 end-to-end 챗봇처럼 입력 텍스트에서 바로 응답 텍스트를 생성하지 않고, 다음과 같이 여러 단계를 거친다.

1. 입력 텍스트를 4차원 정서 자극 벡터로 바꾼다.
2. 이 자극이 256개 뉴런으로 이루어진 그래프에서 퍼지고 수렴하는 과정을 시뮬레이션한다.
3. 시뮬레이션 결과에서 dominant branch를 뽑고, 그것을 64차원 잠재벡터 `z`로 압축한다.
4. `z`로부터 32차원 스타일 벡터 `s`를 예측한다.
5. 예측된 스타일을 프롬프트에 넣어 최종 한국어 응답을 생성한다.

즉, 이 프로젝트는 감정 응답 생성 문제를 아래의 구조로 분해한다.

`text -> stim_vec -> graph dynamics -> dominant branch -> z -> s -> response`

논문이나 보고서에서는 이 구조를 "해석 가능한 중간표현 기반 감정 응답 생성 파이프라인"으로 설명하는 것이 적절하다.

## 3. 코드 기준 실제 구현 내용

### 3.1 자극 인코더

구현 위치:
- [core.py](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/emonet/core.py)

설명:
- 입력 텍스트를 4차원 `stim_vec = [dopamine, serotonin, norepinephrine, melatonin]`로 바꾸는 모듈이 있다.
- TF-IDF + Ridge 회귀 기반이다.
- 학습 타깃은 외부의 실제 생리 신호가 아니라, 말뭉치의 `y` 값과 키워드 힌트 사전을 조합한 proxy target이다.
- 긍정성, 성취, 안전, 위협, 긴박, 피로, 휴식 등과 관련된 키워드 힌트가 들어간다.

중요한 해석:
- 이것은 실제 생체 신호 추정 모델이 아니다.
- "neurochemical analogy" 혹은 "neuro-inspired affective stimulus projection" 정도로 표현하는 것이 안전하다.
- 논문에서 "실제 도파민/세로토닌 수치를 예측한다"라고 쓰면 안 된다.

### 3.2 신경 그래프 동역학

구현 위치:
- [core.py](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/emonet/core.py)

설명:
- 256개 뉴런
- inhibitory 115
- excitatory 115
- modulatory 26
- 각 뉴런은 활성도 `K`, 기억, threshold, refractory 상태, dropout 여부, 입출력 edge를 가진다.
- tick 단위로 시뮬레이션한다.
- memory mixing, cosine similarity, modulatory dropout, threshold shift, rewiring, pruning이 구현되어 있다.
- 마지막 생존 경로들을 모아 dominant branch를 구성한다.

중요한 해석:
- 이 구조는 "neuroscience-validated model"이 아니라 "neuro-inspired graph dynamics"라고 부르는 것이 맞다.
- 연구 기여는 생물학적 정확성이 아니라, 감정 응답을 위한 중간 동역학 표현을 실험적으로 설계했다는 데 있다.

### 3.3 branch -> z 인코딩

설명:
- dominant branch의 각 step은 6차원 피처로 표현된다.
- 현재 기본 설정에서는 학습형 transformer보다 NumPy 기반 요약 인코더가 사용된다.
- 평균, 표준편차, 최소/최대, 처음/끝 값, 변화량, 기울기를 요약하고 무작위 선형사상 + `tanh`로 64차원 `z`를 만든다.

중요한 해석:
- 현재 `z`는 end-to-end 학습된 latent representation이 아니다.
- branch summary feature에 대한 고정 투영에 가깝다.
- 따라서 `z`의 의미를 너무 강하게 주장하면 안 된다.

### 3.4 z -> s 스타일 회귀

설명:
- 32차원 스타일 축을 정의해 두었다.
- `z`로부터 `s`를 예측하는 회귀기는 현재 선형 Ridge 회귀다.
- `LinearZtoSDecoder`가 구현되어 있다.

스타일 축 예시:
- verbosity
- directness
- warmth
- politeness
- cooperativeness
- calmness
- tension
- softness
- sharpness
- seriousness
- reflectiveness

중요한 해석:
- 현재 스타일 축은 "말투와 표현 특성" 중심이다.
- raw affect, hostility, resentment, self-loathing 같은 거친 감정축은 부족하다.
- 이것이 현재 시스템이 과하게 온화한 응답을 내는 큰 원인 중 하나다.

### 3.5 최종 응답 생성

설명:
- 입력 텍스트 + style tags + style summary + style vector를 프롬프트에 넣고 로컬 LLM을 호출한다.
- 한국어 평문 3~6문장 정도로 응답을 생성하게 되어 있다.
- 스타일 제어용 프롬프트 템플릿이 따로 있다.

중요한 해석:
- 현재 프롬프트는 온화함, 자연스러움, 공손함 쪽으로 수렴하기 쉬운 구조다.
- 실제 사용자 문제의 핵심은 여기서 "감정 날것"이 약화된다는 점이다.

## 4. 현재 확인된 데이터/산출물

### 4.1 실재하는 데이터 파일

확인된 파일:
- [out_z_training.csv](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/outputs/z/out_z_training.csv)
- [llm_subset.csv](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/outputs/llm/llm_subset.csv)
- [llm_subset_labeled_50_ollama.csv](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/outputs/llm/llm_subset_labeled_50_ollama.csv)
- [llm_subset_labeled_200_ollama.csv](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/outputs/llm/llm_subset_labeled_200_ollama.csv)
- [single_response.json](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/outputs/responses/single_response.json)
- [e2e_check_report.json](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/outputs/validation/e2e_check_report.json)
- [paper_metrics_snapshot.json](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/outputs/paper/paper_metrics_snapshot.json)

### 4.2 현재 로컬에서 확인된 핵심 통계

이 수치는 로컬에서 실제로 확인한 값이다.

- `out_z_training.csv`
  - rows: 51,628
  - cols: 74
  - unique labels: 60

- `llm_subset.csv`
  - rows: 500
  - cols: 75
  - unique labels: 60

- `llm_subset_labeled_50_ollama.csv`
  - rows: 50
  - kept_rows: 48
  - keep_rate: 0.96
  - consistency_mean: 0.044382
  - consistency_median: 0.042969

- `llm_subset_labeled_200_ollama.csv`
  - rows: 200
  - cols: 163
  - unique labels: 59
  - kept_rows: 190
  - keep_rate: 0.95
  - consistency_mean: 0.099796
  - consistency_median: 0.09375

이 값들은 [paper_metrics_snapshot.json](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/outputs/paper/paper_metrics_snapshot.json)에 정리되어 있다.

## 5. 현재 논문적으로 강하게 쓸 수 있는 사실

아래는 "이미 확인된 사실"이므로 논문/보고서에서 비교적 안전하게 쓸 수 있다.

1. 전체 파이프라인은 코드 수준에서 구현되어 있다.
2. `text -> z` 대규모 산출물이 존재한다.
3. 스타일 라벨링용 subset 생성과 LLM 기반 `(s, s_hat)` 구축 절차가 존재한다.
4. consistency filtering으로 keep sample을 고를 수 있다.
5. 적어도 200개 규모에서는 keep 비율이 95% 수준으로 나온다.
6. 스타일 분포가 특정 방향으로 강하게 편향돼 있다.

## 6. 현재 논문에서 조심해야 할 사실

아래는 아직 약하거나 미완료라서 강하게 주장하면 안 된다.

1. "우리 모델이 최종 응답 품질에서 확실히 우수하다"
   - 아직 완전한 baseline 비교표가 없다.

2. "z가 감정 의미를 잘 포착한다"
   - 해석은 가능하지만, 강한 증거는 아직 부족하다.

3. "신경과학적으로 타당하다"
   - 이건 부적절하다.
   - neuro-inspired라고만 쓰는 것이 맞다.

4. "z -> s 회귀가 매우 잘 된다"
   - 현재 간이 검증 결과는 오히려 baseline보다 약하다.

5. "최종 end-to-end 생성이 완전히 검증되었다"
   - 저장된 기본 e2e 로그는 LLM 서버 연결 실패였다.

## 7. 현재 확인된 중요한 부정적 결과

이건 숨기면 안 된다. 오히려 논문에서 중요한 분석 포인트다.

### 7.1 현재 z->s 회귀기는 baseline보다 아직 강하지 않다

`llm_subset_labeled_200_ollama.csv`의 keep 샘플 190개를 사용해 5개 seed hold-out split으로 평가한 결과:

- decoder_mae_mean: 0.142172
- baseline_mae_mean: 0.137052
- mean_gain: -0.00512

즉, 현재 선형 `z -> s` 회귀기는 단순 mean baseline보다 아직 낫지 않다.

이 사실의 해석:
- 현재 스타일 공간이 편향돼 평균값 근처만 잘 따라가도 일정 수준 성능이 나온다.
- 라벨링 consistency는 높지만, 스타일 축 설계가 지나치게 한 방향으로 몰려 있다.
- 병목은 모델 용량 부족일 수도 있지만, 더 근본적으로는 데이터 구성과 스타일 공간 설계 문제일 가능성이 크다.

### 7.2 스타일 편향이 매우 강하다

keep 샘플 평균 기준 주요 축:

- warmth: 0.7776
- politeness: 0.7789
- cooperativeness: 0.9197
- calmness: 0.9184
- softness: 0.9605
- positivity: 0.9092
- dominance: 0.1066
- tension: 0.0961
- sharpness: 0.0382

또한 가장 극단적으로 치우친 축:

- playfulness: 0.0105
- metaphoricity: 0.0105
- plainness: 0.9645
- sharpness: 0.0382
- softness: 0.9605
- cooperativeness: 0.9197

해석:
- 현재 시스템은 "부드럽고, 협조적이고, 차분하고, 공손한 응답" 쪽으로 강하게 몰린다.
- 이건 사용자가 처음 문제로 제기한 부분과 정확히 연결된다.

## 8. 왜 이런 편향이 생기는가

현재까지 파악한 원인은 세 층이다.

### 8.1 데이터 입력 구성

[cli.py](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/emonet/cli.py) 의 `flatten_dialogue_text`는 `HS01, SS01, HS02, SS02, HS03, SS03`를 `[SEP]`로 합친다.

즉:
- 사용자 감정 발화
- 시스템/상담자 발화

가 한 텍스트에 섞인다.

이 구조는 사용자의 raw emotion뿐 아니라 상담형/완충형 말투를 같이 인코딩하게 만든다.

### 8.2 스타일 축 설계

현재 축들은 아래 성향이 강하다.

- warmth
- politeness
- cooperativeness
- calmness
- softness
- positivity

반면 아래 축은 없거나 약하다.

- hostility
- resentment
- despair
- sarcasm
- self-loathing
- emotional volatility
- profanity tendency

따라서 감정 날것을 표현하기보다 "정중한 조절 톤"으로 해를 찾게 된다.

### 8.3 생성 프롬프트 구조

[response_generation_prompt.md](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/prompts/response_generation_prompt.md) 와 관련 생성 로직은 style tags와 macro summary를 함께 넣는다.

macro summary는 `따뜻함`, `구조화`, `감정개방성`, `형식성` 같은 요약값을 다시 강조한다.

이 구조는 모델을 더 평균적이고 무난한 톤으로 밀어주는 경향이 있다.

## 9. 현재 작성된 논문 초안과 보조 문서

현재 추가된 문서:

- [PAPER_DRAFT_ko.md](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/PAPER_DRAFT_ko.md)
- [PAPER_RESULTS_APPENDIX.md](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/PAPER_RESULTS_APPENDIX.md)
- [PAPER_WORKLIST.md](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/PAPER_WORKLIST.md)

역할:

- `PAPER_DRAFT_ko.md`
  - 현재 기준 한국어 논문 초안
  - 초록, 방법, 데이터셋, 결과, 논의, 한계, 향후 연구 포함

- `PAPER_RESULTS_APPENDIX.md`
  - 수치 정리
  - style bias와 `z -> s` 회귀의 부정적 결과 포함

- `PAPER_WORKLIST.md`
  - 로컬에서 끝난 것
  - 원격에서 돌릴 것
  - 제출 전 비어 있는 것

## 10. 현재 작성자(인간 사용자)의 우선순위

사용자의 요구는 단순 성능 자랑이 아니다. 아래 조건이 매우 중요하다.

1. 연구에서 "무엇을 했는지"가 분명해야 한다.
2. 실험에서 "무엇을 검증했는지"가 분명해야 한다.
3. 실생활에서 "어디에 쓸 수 있는지"가 분명해야 한다.
4. 지나치게 추상적이거나 뜬구름 잡는 설명보다, 실제 구현과 실제 활용 시나리오가 명확해야 한다.

즉 글쓰기 AI는 화려한 수사보다 명확한 설명을 우선해야 한다.

## 11. 논문에서 강조해야 할 연구 기여

현재 버전 기준으로 기여를 정리하면 다음이 적절하다.

1. 감정 응답 생성을 `text -> stim -> graph dynamics -> z -> s -> response`로 분해한 실험적 파이프라인을 제안했다.
2. 한국어 감정 대화로부터 4차원 자극벡터와 64차원 잠재표현을 생성하는 neuro-inspired graph dynamics를 구현했다.
3. 로컬 LLM을 이용해 `(z, s)` 스타일 약지도 데이터셋을 자동 구축하는 절차를 만들었다.
4. consistency filtering과 stage-wise e2e validation을 포함한 평가/운영 체계를 구현했다.
5. 현재 시스템의 온화함 편향을 데이터 구성, 스타일 공간, 프롬프트 구조의 상호작용으로 분석했다.

## 12. 실생활 활용 시나리오

사용자는 이 부분이 특히 명확해야 한다고 강조했다. 따라서 활용처는 추상적 표현이 아니라 구체적 시나리오로 써야 한다.

### 12.1 감정 케어 챗봇

사용 사례:
- 피로, 불안, 예민함, 억울함, 자책 등 감정 상태가 강한 사용자에게 기계적 답변이 아닌, 상태에 맞는 반응 톤을 조절한 응답을 생성

주의:
- 의료 진단이나 임상 판단 도구라고 쓰면 안 된다.
- "대화 톤 조절 보조 시스템" 정도로 쓰는 게 맞다.

### 12.2 상담/코칭 보조 시스템

사용 사례:
- 동일한 내용이라도 어떤 말투가 어떤 감정 상태에 더 적합한지 비교·훈련
- 상담자 교육용 시뮬레이션

### 12.3 고객 응대 자동화

사용 사례:
- 분노, 억울함, 피로, 불안이 섞인 고객 입력에 대해 지나치게 기계적이거나 공격적이지 않은 응답 톤 제어

### 12.4 게임/NPC/디지털 휴먼

사용 사례:
- 단순 scripted response가 아니라 내부 affect dynamics를 거친 반응 생성

### 12.5 감정 적응형 음성 에이전트

사용 사례:
- 내용은 같더라도 거리감, 직설성, 긴장도, 정서 개방성을 조정해 응답 스타일을 바꾸는 모듈

핵심 표현:
- "감정을 이해한다"보다 "응답 스타일을 제어 가능하게 조정한다"가 더 안전하고 정확하다.

## 13. 현재 원격 실행 상황

고성능이 필요한 작업은 사용자가 RDP로 별도 컴퓨터에서 돌리기로 했다.

원격 실행용으로 추가된 파일:

- [paper_metrics.py](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/scripts/paper_metrics.py)
- [paper_remote_runs.ps1](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/scripts/paper_remote_runs.ps1)
- [paper_remote_all.ps1](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/scripts/paper_remote_all.ps1)

원격에서 한 번에 돌리려던 작업:

1. 성공한 e2e 생성 로그 재생성
2. 500개 subset 전체 스타일 라벨링
3. 500-row 기반 `z -> s` 회귀 재학습
4. full training set에 `s_pred` 부착
5. paper metrics 재계산

현재 상태:
- 원격 PC에서 `python` alias 문제로 처음에는 실행이 실패했다.
- 이후 conda 환경의 Python 경로를 직접 사용하도록 유도했다.
- 하지만 이 migration 문서 시점에서는 "원격 산출물이 성공적으로 생성되었다"는 결과는 아직 확인되지 않았다.

즉, 아래 파일들은 "생성될 예정인 목표 파일"이지 아직 확인 완료 상태가 아닐 수 있다.

목표 파일:
- `outputs/validation/e2e_check_report_success.json`
- `outputs/validation/e2e_check_runs_success.csv`
- `outputs/validation/e2e_check_runs_success.jsonl`
- `outputs/llm/llm_subset_labeled_500_ollama.csv`
- `artifacts/z_to_s_decoder_500.npz`
- `outputs/z/out_z_training_with_s_pred_500.csv`
- `outputs/paper/paper_metrics_snapshot_remote.json`

## 14. 원격 실험이 성공했다고 가정할 때 논문이 주장해야 할 것

이건 "가정 하의 논문 방향"이다. 아직 실제 숫자가 들어오지 않았을 수 있다.

논문 핵심 메시지 후보:
- EmoNet은 감정 응답 생성을 직접 end-to-end generation으로 처리하지 않고, 감정 동역학 기반 중간표현과 스타일 회귀를 통해 더 해석 가능하고 더 제어 가능한 응답 생성 구조를 제시한다.

필요한 결과표:

1. 데이터셋 규모 표
2. `z -> s` 회귀 성능 표
3. direct prompting 대비 최종 응답 비교표
4. ablation 표
5. 인간평가 혹은 judge model 평가표

## 15. 현재 빠져 있는 것

아래 항목은 아직 논문 제출용으로 비어 있다.

1. 관련연구 섹션
2. 참고문헌
3. baseline 결과표
4. ablation 결과표
5. 최종 응답 품질의 정량 비교
6. figure/diagram
7. 하드웨어/실행 시간 정보
8. 인간평가 혹은 자동 judge 평가

## 16. 글쓰기 AI가 논문에서 절대 하면 안 되는 과장

다음 표현은 피해야 한다.

- "실제 인간의 감정을 이해한다"
- "도파민/세로토닌을 실제로 추정한다"
- "신경과학적으로 입증되었다"
- "임상적으로 유효하다"
- "현재 모델이 최종 성능에서 우수함이 입증되었다"

대신 다음 표현이 안전하다.

- neuro-inspired
- affective dynamics simulation
- interpretable intermediate representation
- style-controllable response generation
- experimental pipeline
- weakly supervised style labeling

## 17. 글쓰기 AI가 반드시 명확히 적어야 할 것

### 17.1 우리가 구현한 것

- 자극 인코더
- graph dynamics
- dominant branch extraction
- latent style regression
- local-LLM-based `(z, s)` data construction
- response generation
- consistency filtering
- e2e validation

### 17.2 우리가 실험한 것

- 대규모 `z` export
- subset sampling
- LLM style labeling
- consistency-based filtering
- 간이 `z -> s` hold-out 검증
- e2e stage validation

### 17.3 우리가 아직 못 끝낸 것

- 안정적인 대규모 final response evaluation
- baseline and ablation completion
- stronger style space redesign

## 18. 글쓰기 방향 권장

이 프로젝트는 "완성된 상용 시스템"보다 "연구용 파이프라인과 분석"으로 쓰는 게 더 강하다.

권장 톤:
- 솔직함
- 단계별 설명
- 실제 구현 중심
- 장점과 한계를 같이 제시

권장 논문 포지셔닝:
- "감정 응답 생성에서 해석 가능한 중간표현과 스타일 제어를 결합한 연구 프로토타입"

비권장 포지셔닝:
- "최고 성능의 감정 대화 생성 모델"

## 19. 이관 후 바로 해야 할 일

다른 글쓰기 AI가 이 문서를 받으면 우선 다음 순서로 작업하는 것이 좋다.

1. [PAPER_DRAFT_ko.md](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/PAPER_DRAFT_ko.md)를 읽고 기존 초안 구조를 파악한다.
2. [PAPER_RESULTS_APPENDIX.md](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/PAPER_RESULTS_APPENDIX.md)에서 현재 수치를 확인한다.
3. [paper_metrics_snapshot.json](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/outputs/paper/paper_metrics_snapshot.json)의 값과 초안의 수치가 일치하는지 확인한다.
4. 원격 실험 결과가 들어오면 `success/e2e/500-labeling` 관련 수치를 업데이트한다.
5. 관련연구와 practical applications 섹션을 강화한다.
6. 사용자가 특히 원하는 "무엇을 했고, 실생활에서 어떻게 쓰이는가"를 서론과 결론에서 더 선명하게 만든다.

## 20. 마지막 요약

이 프로젝트는 다음 상태에 있다.

- 구현은 상당 부분 되어 있다.
- 대규모 `z` 산출물과 소규모 `(z, s)` 라벨링 산출물이 존재한다.
- consistency filtering은 잘 작동한다.
- 하지만 현재 스타일 공간은 과하게 온화하고 협조적인 쪽으로 편향돼 있다.
- 현재 선형 `z -> s` 회귀기는 baseline보다 아직 강하지 않다.
- 따라서 이 연구의 현재 강점은 "해석 가능한 감정 응답 파이프라인의 구축과 분석"이지, "최종 성능 우위의 입증"은 아니다.

논문 작성 시 이 점을 정직하게 유지하는 것이 오히려 전체 설득력을 높인다.
