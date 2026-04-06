# Paper Worklist

## 현재 canonical 문서

- `PAPER_FRESH_DRAFT_ko.md`: 현재 메인 초안
- `PAPER_DRAFT_AUDIT_2026-04-06.md`: 초안과 산출물 불일치 점검표
- `PAPER_BLUEPRINT_ko.md`: 구조/포지셔닝 청사진
- `SOFTNESS_BIAS_MITIGATION_2026-04-06.md`: 부드러움 편향 원인과 해결 실험 메모
- `BRANCH_COLLAPSE_MITIGATION_2026-04-06.md`: branch 길이 붕괴 원인과 해결 실험 메모

## 이미 끝난 것

- old draft를 `archive/2026-04-06_paper_history`로 이동
- legacy run 산출물을 `archive/2026-04-06_legacy_runs`로 이동
- `extended40` 4,000-row labeling 완료
- `learned z encoder` 및 `z -> s` decoder artifact 생성
- active top-level 산출물 정리

## 지금 논문에서 바로 쓸 수 있는 사실

- `out_z_training_extended40.csv`: 51,628 rows
- `llm_subset_labeled_4000_extended40.csv`: 4,000 rows
- `status=ok`: 3,971
- `keep_sample=True`: 2,832
- keep rate: 70.8%
- current active style profile: `extended40`
- current active encoder artifact: `dominant_branch_encoder_extended40.pt`

## 제출 전에 반드시 메워야 하는 것

### 1. current predictor 표

- stat encoder vs learned encoder 비교
- current `extended40` 기준 MAE
- mean baseline 대비 gain/loss

### 2. current generation 표

- direct
- stim-only
- current EmoNet prompt

### 3. current smoke / e2e 로그

- response JSON
- validation report
- run log

### 4. 본문 서술 정합화

- 32차원 style 서술 제거 또는 legacy 실험으로 강등
- 200/190 labeled 결과를 main result에서 분리
- statistical encoder를 current method처럼 서술한 부분 수정
- expression cues/raw vector 포함 prompt 설명 수정

## 추천 진행 순서

### A. 논문 포지션 먼저 확정

둘 중 하나를 먼저 고른다.

- 진단 프레임워크 중심
- 최신 system update 중심

현재 상태에서는 진단 프레임워크 중심이 더 안전하다.

### B. current evidence 재생성

- smoke response 1회
- e2e validation 1회 이상
- predictor evaluation 재실행
- generation baseline 재실행

### C. draft 본문 재작성

- 초록
- 방법론의 `z` encoder 문단
- 실험 설정
- 결과 섹션
- 결론

### D. legacy 표는 appendix로 이동

- `outputs/paper/requested_tables/*`의 기존 표는 current result가 아니라 legacy result로 명시
- 본문에 남길지 appendix로 보낼지 결정

## 현재 가장 위험한 부분

- current code와 current paper table이 서로 다른 세대의 결과를 가리킨다
- `extended40`를 넣었지만 raw affect 축은 아직 거의 0 근처다
- artifact는 생겼지만 current 성능표와 validation 로그가 없다
- dominant branch가 대부분 길이 1이라, branch history가 아직 충분한 시계열 정보를 제공하지 못한다
