# Paper Draft Audit (2026-04-06)

## 1. 현재 기준 문서 상태

현재 top-level에서 유지하는 paper 관련 문서는 아래 세 개다.

- `PAPER_FRESH_DRAFT_ko.md`: 현재 메인 초안
- `PAPER_BLUEPRINT_ko.md`: 구조 설계용 청사진
- `PAPER_WORKLIST.md`: 현재 작업/실험 체크리스트

과거 초안은 삭제하지 않고 아래로 이동했다.

- `archive/2026-04-06_paper_history/PAPER_DRAFT_ko.md`
- `archive/2026-04-06_paper_history/PAPER_DRAFT_ko_v2.md`
- `archive/2026-04-06_paper_history/PAPER_RESULTS_APPENDIX.md`

## 2. 현재 active 산출물

현재 상위 경로에 남아 있는 active 결과물은 아래와 같다.

- `outputs/z/out_z_training_extended40.csv`: 51,628 rows
- `outputs/llm/llm_subset_4000_extended40.csv`: 4,000-row subset
- `outputs/llm/llm_subset_labeled_4000_extended40.csv`: 4,000 rows labeled
- `outputs/z/out_z_training_learned_extended40.csv`: 2,832 rows learned-encoder export
- `artifacts/dominant_branch_encoder_extended40.pt`: 학습형 `z` encoder
- `artifacts/z_to_s_decoder_extended40.npz`: `z -> s` decoder

`llm_subset_labeled_4000_extended40.csv` 기준 핵심 수치는 아래와 같다.

- total rows: 4,000
- `status=ok`: 3,971
- `status=error`: 29
- `keep_sample=True`: 2,832
- keep rate: 0.7080
- mean `consistency_l1` over ok rows: 0.102152
- mean `consistency_l1` over keep rows: 0.083183

## 3. 현재 draft에서 낡았거나 실패한 주장

### 3.1 방법론 서술이 현재 코드와 어긋남

`PAPER_FRESH_DRAFT_ko.md`는 아직 아래를 현재 시스템처럼 서술한다.

- `z -> s`가 32차원 style space를 사용한다고 서술함
- 500 subset 중 200 labeled, 190 keep 결과를 현재 본문 중심 결과처럼 서술함
- `z` encoder를 "학습형 transformer 대신 통계 요약 기반 인코더"라고 서술함
- 응답 생성 프롬프트를 `s_pred + style tags + style summary + expression cues` 조합으로 설명함

하지만 현재 active 산출물 기준으로는 아래가 맞다.

- active style profile은 `extended40`
- active labeled set은 4,000 rows / 3,971 ok / 2,832 keep
- active encoder artifact는 `dominant_branch_encoder_extended40.pt`
- 현재 기본 생성 프롬프트는 condensed `STYLE_TAGS + STYLE_SUMMARY` 중심이며 raw vector / expression cue를 기본 템플릿에서 제거했다

즉, 현재 draft는 "현재 시스템 설명"과 "legacy core32 실험 요약"이 섞여 있다.

### 3.2 결과표가 전부 legacy 기준임

`outputs/paper/requested_tables/*.json`과 `outputs/experiments/*`는 모두 2026-04-03 시점 산출물이며, 현재 active `extended40` 파이프라인을 반영하지 않는다.

확인된 예시는 아래와 같다.

- `baseline_predictor_table.json`: `rows_used=190`
- `baseline_generation_table.json`: `direct/stim_only/emonet_full` 비교가 legacy scored matrix 기준

따라서 현재 draft의 결과표를 "현 시스템의 최종 성능"처럼 제출하면 방법-결과 대응이 깨진다.

### 3.3 최신 end-to-end 근거가 비어 있음

현재 active top-level에는 아래가 없다.

- `outputs/responses/*`
- `outputs/validation/*`

즉, 현재 active `extended40 + learned z encoder` 경로에 대한 성공 smoke/e2e 로그가 논문용 증빙으로 남아 있지 않다.

### 3.4 `extended40`를 넣어도 raw affect bias가 해결됐다고 말할 수 없음

keep set 2,832 rows에서 `s_0..s_39`를 `extended40` 축 순서에 매핑해 평균을 확인하면, 새 raw affect 축은 여전히 거의 0에 붙어 있다.

낮은 축 예시:

- hostility: 0.0001
- resentment: 0.0001
- shame: 0.0008
- volatility: 0.0012
- despair: 0.0020
- fearfulness: 0.0046

높은 축 예시:

- directness: 0.7855
- explicitness: 0.8373
- initiative: 0.8818
- positivity: 0.9431
- calmness: 0.9480
- cooperativeness: 0.9522
- softness: 0.9537
- plainness: 0.9538

따라서 "style axis 확장으로 raw affect 문제가 해결됐다"는 식의 문장은 현재 데이터로는 방어되지 않는다.

## 4. 현재 부족한 부분

아래는 draft 제출 전에 메워야 하는 핵심 공백이다.

### 4.1 최신 predictor 평가 부재

현재 active learned encoder와 `extended40` decoder가 생겼지만, 아래 비교표가 없다.

- stat encoder vs learned transformer encoder
- old core32 vs current extended40
- `z -> s` vs mean baseline

즉, artifact는 있지만 성능 입증 표가 없다.

### 4.2 최신 generation 평가 부재

현재 prompt 구조가 바뀌었는데, 그 변경 후의 generation baseline이 없다.

필요한 최소 비교는 아래다.

- direct
- stim-only
- EmoNet current prompt

### 4.3 서술 포지셔닝 미정

지금 논문이 무엇을 주장하는지 아직 명확히 정리되지 않았다.

- 진단용 연구 프레임워크 논문인지
- 최신 active `extended40` 시스템 성능 업데이트 논문인지

이 선택이 안 되면, 본문은 계속 legacy 진단 결과와 current system artifact를 섞게 된다.

## 5. 권장 정리 방향

### 권장안 A: "진단 프레임워크" 중심으로 고정

가장 안전한 방향은 다음이다.

- legacy core32/200/190 결과는 "초기 진단 결과"로 명시
- current extended40/learned encoder는 "후속 확장 실험"으로 분리
- 논문의 1차 기여를 "최종 생성 성능"이 아니라 "병목을 분해해 드러내는 분석 프레임워크"에 둔다

이 방향의 장점:

- 현재 direct baseline이 더 좋았다는 결과를 숨기지 않아도 됨
- branch collapse, style bias, decoder weakness를 그대로 핵심 발견으로 쓸 수 있음
- 지금 당장 부족한 최신 generation 표가 없어도 구조적 일관성을 유지하기 쉬움

이 방향의 단점:

- 최신 active extended40 학습 결과는 주기여가 아니라 appendix/후속실험에 가까워짐

### 권장안 B: "최신 시스템 업데이트" 중심으로 재작성

이 방향을 택하려면 아래가 먼저 필요하다.

- current extended40 predictor 표 재생성
- current prompt 기준 generation 표 재생성
- current smoke/e2e 성공 로그 저장
- methods/results 전체를 current config로 재작성

이 방향의 장점:

- 현재 코드와 산출물을 본문 중심으로 바로 연결할 수 있음

이 방향의 단점:

- 아직 표와 로그가 모자라서 바로 제출 가능한 상태는 아님

## 6. 바로 실행할 해결 순서

### 1순위

현재 draft 상단에 "legacy 결과와 current artifact가 섞여 있음" 경고를 유지하고, 제출/공유 전에 반드시 이 audit 문서를 먼저 참조한다.

### 2순위

현재 active 파이프라인으로 아래를 재생성한다.

- smoke response JSON
- e2e validation report
- predictor comparison table
- generation comparison table

### 3순위

`PAPER_FRESH_DRAFT_ko.md`의 아래 문단을 전면 교체한다.

- 초록
- 방법론 중 `z` encoder 설명
- 데이터셋/실험 설정
- 결과 섹션의 표/수치
- 결론의 주장 강도

### 4순위

legacy 결과는 버리지 말고 appendix 혹은 "초기 진단 실험" 절로 옮긴다.

## 7. 결론

현재 draft의 가장 큰 문제는 "실패한 것" 자체가 아니라, 서로 다른 세대의 실험 결과를 하나의 현재 상태처럼 섞고 있다는 점이다. 지금 상태에서 가장 먼저 해야 할 일은 성능을 미화하는 것이 아니라, legacy 진단 결과와 current extended40 산출물을 분리해 논문의 주장 단위를 맞추는 것이다.
