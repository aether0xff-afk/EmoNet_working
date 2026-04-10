# Paper Worklist

## 현재 canonical 문서

- `PAPER_DRAFT_ko.md`: 현재 단일 초안
- `README.md`: active/archived 구조와 현재 파이프라인 설명
- `PAPER_METRICS_AND_PARAMETER_RATIONALE_2026-04-10.md`: 논문용 지표와 파라미터 변경 근거 기준

## 이미 정리된 것

- `v4` active line 생성
- old 문서와 구버전 artifact를 `archive/`로 분리
- calibrated reference config 기반 full export / fit 완료
- raw trajectory batch 분석 완료
- GPT-5.4 episode interpretation 경로 구축 완료
- `episode_trace`, `hybrid_episode` conditioning이 generation path에 연결됨
- `episode_trace v2`가 `mean_total=3.4673`으로 naive `emonet_full=3.2478`을 넘김

## 지금 논문에서 바로 쓸 수 있는 사실

- full export rows: `51,628`
- calibrated branch mean: `70.4684`
- calibrated `len1_ratio`: `0.0948`
- learned training rows: `1,800`
- encoder head val MAE: `0.10273`
- decoder val MAE: `0.117898`
- active learned artifacts:
  - `artifacts/dominant_branch_encoder_extended40_calref_v1.pt`
  - `artifacts/z_to_s_decoder_extended40_calref_v1.npz`
- active paper output:
  - `outputs/paper/refresh_2026-04-09_calref_v1`

## 지금 남은 핵심 과제

### 1. episode-conditioned generation 평가

- `episode_trace`
- `hybrid_episode`
- 기존 `direct / stim_only / emonet_full / raw_trace / appraisal_trace / hybrid_trace`와 비교

### 2. judge + refresh 재생성

- `GPT-5.4 judge`
- 새 generation matrix 기준 테이블/그림 refresh

### 3. 본문 업데이트

- heuristic emotion readout 대신 `raw trajectory -> GPT-5.4 episode interpretation` 경로를 main method로 서술
- `emotion`을 stimulus label이 아니라 `trajectory episode`로 정의
- 남은 병목을 `style target bias`, `surface softening`, `episode->response 연결`로 다시 정리

## 지금 가장 위험한 부분

- `legacy_cli.py`가 아직 크고 generation 관련 로직이 많이 남아 있다
- `episode_trace` end-to-end 품질 비교는 아직 judge score로 확정되지 않았다
- raw trajectory는 솔직하지만, 최종 응답 surface는 여전히 순화될 가능성이 있다
