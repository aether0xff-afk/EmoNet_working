# EmoNet v4

`v4`는 현재 active 작업선이다. `v3`에서 최신 calibrated branch line과 trajectory/episode 해석 경로만 옮겨왔고, 구버전 문서와 artifact는 `archive/`로 분리했다.

## Active 문서

- `PAPER_DRAFT_ko.md`
- `PAPER_WORKLIST.md`

## Active 코드

- `emonet/core.py`: branch dynamics core
- `emonet/cli.py`: public v4 CLI facade
- `emonet/legacy_cli.py`: 기존 대형 CLI 구현
- `emonet/llm_api.py`: OpenAI-compatible API 호출 분리 모듈
- `emonet/episode_conditioning.py`: episode JSON 기반 conditioning 모듈

## Active 연구 스크립트

- `scripts/inspect_emotion_trace.py`: 단일 샘플 raw trace 추출
- `scripts/analyze_emotion_trajectory_batch.py`: 다중 샘플 trajectory 분석
- `scripts/interpret_emotion_trajectory.py`: trajectory를 GPT-5.4 episode JSON으로 해석
- `scripts/experiment_matrix.py`: generation 비교 실험
- `scripts/score_experiment_matrix.py`: judge scoring
- `scripts/generate_paper_refresh_structfix.py`: paper refresh 산출물 생성

## Active 산출물

- `artifacts/dominant_branch_encoder_extended40_calref_v1.pt`
- `artifacts/z_to_s_decoder_extended40_calref_v1.npz`
- `outputs/z/out_z_training_extended40_calref_v1.csv`
- `outputs/z/out_z_training_learned_extended40_calref_v1.csv`
- `outputs/research/trajectory_batch_v1`
- `outputs/research/trajectory_batch_v1_gpt54`
- `outputs/paper/refresh_2026-04-09_calref_v1`

## Active 파이프라인

1. RDP에서 raw branch/trajectory 생성
2. 로컬에서 `interpret_emotion_trajectory.py`로 GPT-5.4 episode 해석
3. `episode_trace` 또는 `hybrid_episode` conditioning으로 generation 비교
4. judge scoring과 paper refresh

## Archive

- `archive/docs`: 이전 메모, blueprint, migration notes
- `archive/artifacts`: 구버전 encoder/decoder
- `archive/cache`: 임시/캐시용 예약 폴더

## 메모

- `emonet/legacy_cli.py`는 아직 크다. 현재는 facade + extracted module 방식으로 public surface만 정리한 상태다.
- 다음 정리 대상은 generation command group을 `legacy_cli.py`에서 별도 모듈로 완전히 떼어내는 것이다.
