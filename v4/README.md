# EmoNet v4

`v4`는 현재 active 작업선이다. `v3`에서 최신 calibrated branch line과 trajectory/episode 해석 경로만 옮겨왔고, 구버전 문서와 artifact는 `archive/`로 분리했다.

## Active 문서

- `RESEARCH_SUMMARY_2026-04-10.md`
- `PAPER_DRAFT_ko.md`
- `PAPER_WORKLIST.md`
- `paper/PAPER.md`
- `paper/README.md`

## Active 코드

- `emonet/core.py`: branch dynamics core
- `emonet/cli.py`: public v4 CLI facade
- `emonet/legacy_cli.py`: 기존 대형 CLI 구현
- `emonet/llm_api.py`: OpenAI-compatible API 호출 분리 모듈
- `emonet/episode_conditioning.py`: episode JSON 기반 conditioning 모듈
- `emonet/chat_service.py`: GUI용 runtime/generation service
- `local_gui.py`: 표준 라이브러리 기반 로컬 GUI 엔트리
- `streamlit_app.py`: 이전 Streamlit GUI 엔트리

## Active 연구 스크립트

- `scripts/inspect_emotion_trace.py`: 단일 샘플 raw trace 추출
- `scripts/analyze_emotion_trajectory_batch.py`: 다중 샘플 trajectory 분석
- `scripts/interpret_emotion_trajectory.py`: trajectory를 GPT-5.4 episode JSON으로 해석
- `scripts/experiment_matrix.py`: generation 비교 실험
- `scripts/score_experiment_matrix.py`: judge scoring
- `scripts/build_targeted_superiority_set.py`: episode-sensitive targeted set 구성
- `scripts/generate_episode_v3_targeted.py`: `episode_trace_v3` targeted 응답 생성
- `scripts/score_superiority_judge.py`: targeted episode-fidelity judge scoring
- `scripts/analyze_paired_superiority.py`: paired delta/bootstrap/win-rate 분석
- `scripts/prepare_human_eval.py`: blind human A/B CSV와 answer key 생성
- `scripts/generate_paper_refresh_structfix.py`: paper refresh 산출물 생성

## Active 산출물

- `artifacts/dominant_branch_encoder_extended40_calref_v1.pt`
- `artifacts/z_to_s_decoder_extended40_calref_v1.npz`
- `data/benchmark/benchmark_results_20260305_180830.csv`
- `outputs/z/out_z_training_extended40_calref_v1.csv`
- `outputs/z/out_z_training_learned_extended40_calref_v1.csv`
- `outputs/branch_calibration/reference_calibration_rdp_v1`
- `outputs/research/trajectory_batch_v1`
- `outputs/research/trajectory_batch_v1_gpt54`
- `outputs/research/trajectory_batch_matrix120_v1`
- `outputs/research/trajectory_batch_matrix120_v1_gpt54`
- `outputs/research/summary_2026-04-10`
- `outputs/paper/refresh_2026-04-09_calref_v1`
- `outputs/experiments/superiority_targeted_v1`
- `outputs/beta_judging/targeted_episode_v3_vs_stim_2026-05-03`
- `outputs/beta_judging/targeted_episode_v3_vs_episode_2026-05-03`

## Paper Workspace

- `paper/PAPER.md`: 현재 paper용 markdown 기준 문서
- `paper/sections/*.tex`: LaTeX section source
- `paper/tables/*.tex`: LaTeX table source
- `paper/build/main.pdf`: 현재 빌드된 PDF

## Active 파이프라인

1. RDP에서 raw branch/trajectory 생성
2. 로컬에서 `interpret_emotion_trajectory.py`로 GPT-5.4 episode 해석
3. `episode_trace` 또는 `hybrid_episode` conditioning으로 generation 비교
4. judge scoring과 paper refresh

## Current v4 status

2026-05-02 기준 targeted superiority 실험은 `outputs/experiments/superiority_targeted_v1/`에 정리되어 있다.

- Targeted set: 80개 episode-sensitive record
- 핵심 비교: `episode_trace_v3` vs `stim_only`
- Paired n: 78
- `mean_total` delta: +1.8308
- Bootstrap 95% CI: [+1.5667, +2.0897]
- Win / Tie / Loss: 70 / 3 / 5
- Win rate: 0.8974
- Naturalness: `episode_trace_v3` 4.4000, `stim_only` 4.1410

현재 입증 가능한 주장은 broad/general superiority가 아니라, episode 정보가 필요한 targeted 감정 입력에서 `episode_trace_v3`가 appraisal fidelity, raw affect preservation, anti-softening, action tendency fit, emotional specificity 기준으로 `stim_only`보다 우수하다는 것이다.

남은 confirmatory 단계는 human blind A/B다. 준비된 패키지는 `outputs/beta_judging/` 아래에 있다.

## Local GUI

```powershell
cd .\v4
pip install -r requirements.txt
python .\local_gui.py
```

- `Chat`: EmoNet branch dynamics 결과를 Claude API에 condition하여 답변하는 최소 안정 데모
- `Human A/B`: beta judging row-by-row 평가, 저장, CSV 다운로드
- 기본 LLM provider는 Claude API 고정이며, 기본 model은 `claude-sonnet-4-20250514`다.
- Claude API key는 왼쪽 password field에 입력하거나 `ANTHROPIC_API_KEY` 환경변수로 설정한다.
- API key는 repo 파일에 저장하지 않는다.
- Usage 패널은 Claude API 응답의 token usage를 바탕으로 session budget 대비 추정 사용액과 잔액을 표시한다.
- Human A/B 진행 상황은 `outputs/local_gui_progress/`에 로컬 JSON으로 저장한다.
- 불안정했던 Streamlit 상태 관리, research dashboard, result upload analyzer, provider switching, advanced runtime controls는 Local GUI에서 제거했다.
- 기본 runtime 경로는 `v4/artifacts`, `v4/data/benchmark`, `v4/outputs/z` 기준으로 잡힌다.

## Archive

- `archive/docs`: 이전 메모, blueprint, migration notes
- `archive/artifacts`: 구버전 encoder/decoder
- `archive/cache`: 임시/캐시용 예약 폴더

## 메모

- `emonet/legacy_cli.py`는 아직 크다. 현재는 facade + extracted module 방식으로 public surface만 정리한 상태다.
- stimulus encoder 기본 학습 데이터는 루트 공유 폴더가 아니라 현재 `v4`의 `outputs/z/*.csv`와 `data/benchmark/*.csv`를 기준으로 잡는다.
- 다음 정리 대상은 generation command group을 `legacy_cli.py`에서 별도 모듈로 완전히 떼어내는 것이다.
