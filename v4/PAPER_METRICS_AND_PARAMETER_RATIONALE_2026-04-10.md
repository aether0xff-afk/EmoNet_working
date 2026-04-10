# Paper Metrics And Parameter Rationale (2026-04-10)

## 목적

이 문서는 현재 `v4`에서 논문 본문/표/그림에 넣을 만한 지표를 정리하고, 파라미터를 수정할 때 요구할 근거 기준을 명시한다.

핵심 원칙은 하나다.

- 파라미터는 "좋아 보여서" 바꾸지 않는다.
- 어떤 파라미터든 반드시
  - 어떤 현상을 바꾸려는지,
  - 어떤 지표가 근거인지,
  - 어느 범위까지 개선되어야 채택하는지
  를 명시한다.

## 1. 지금 논문에 넣을 수 있는 핵심 지표

### 1.1 Branch dynamics recovery

출처:

- [paper_refresh_summary.json](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/paper/refresh_2026-04-09_calref_v1/tables/paper_refresh_summary.json)

핵심 수치:

- before mean branch len: `1.0539`
- after mean branch len: `70.4684`
- before `len1_ratio`: `0.9734`
- after `len1_ratio`: `0.0948`
- after `p90 / p95 / max`: `126 / 126 / 126`

논문에서 주장 가능한 것:

- branch collapse는 구조적으로 완화되었다.
- 다만 upper-tail saturation은 여전히 남아 있다.

추천 그림:

- [dominant_branch_before_after.svg](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/paper/refresh_2026-04-09_calref_v1/figures/dominant_branch_before_after.svg)
- [dominant_branch_length_distribution_structfix.svg](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/paper/refresh_2026-04-09_calref_v1/figures/dominant_branch_length_distribution_structfix.svg)

### 1.2 Calibration-backed reference config

출처:

- [combined_validation.json](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/branch_calibration/reference_calibration_rdp_v1/combined_validation.json)
- [calibrated_reference_config.json](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/branch_calibration/reference_calibration_rdp_v1/calibrated_reference_config.json)

핵심 수치:

- `is_feasible = true`
- `no_activity_ratio = 0.05`
- `len1_ratio = 0.05`
- `hit_max_ticks_ratio = 0.25`
- `mean_first_active_tick = 2.40`
- `late_ignition_ratio_ge_15 = 0.0`
- `mean_branch_len = 79.78`
- `mean_active_window_ticks = 80.02`
- `evidence_score = 88.8412`

논문에서 주장 가능한 것:

- current reference config는 임의 initial value가 아니라 calibration experiment로 선택되었다.
- selection target은 점화 실패, collapse, over-persistence를 동시에 줄이는 것이다.

### 1.3 Style target bias

출처:

- [paper_refresh_summary.json](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/paper/refresh_2026-04-09_calref_v1/tables/paper_refresh_summary.json)

핵심 수치:

- `softness = 0.9276`
- `calmness = 0.9132`
- `cooperativeness = 0.9202`
- `positivity = 0.9051`
- `hostility = 0.0003`
- `resentment = 0.0003`
- `despair = 0.0044`
- `volatility = 0.0022`

논문에서 주장 가능한 것:

- supervision target 자체가 너무 safe/cooperative하며, 이것이 최종 surface softening의 주요 원인 중 하나다.

추천 그림:

- [style_bias_axes_extended40.svg](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/paper/refresh_2026-04-09_calref_v1/figures/style_bias_axes_extended40.svg)
- [style_consistency_histogram_extended40.svg](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/paper/refresh_2026-04-09_calref_v1/figures/style_consistency_histogram_extended40.svg)

### 1.4 Predictor competitiveness

출처:

- [paper_refresh_summary.json](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/paper/refresh_2026-04-09_calref_v1/tables/paper_refresh_summary.json)

핵심 수치:

- `mean baseline = 0.117288`
- `stim_only = 0.116513`
- `text_tfidf = 0.114628`
- `legacy_z64 = 0.116846`
- `structfix_learned_z64 = 0.117328`

논문에서 주장 가능한 것:

- branch dynamics recovery는 predictor superiority를 자동으로 보장하지 않는다.
- 현재 `z -> s` path는 아직 text baseline을 넘지 못한다.

추천 그림:

- [predictor_mae_comparison_current.svg](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/paper/refresh_2026-04-09_calref_v1/figures/predictor_mae_comparison_current.svg)

### 1.5 Trajectory-to-episode interpretation quality

출처:

- [trajectory_batch_matrix120_v1_gpt54](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/research/trajectory_batch_matrix120_v1_gpt54)

핵심 수치:

- interpreted samples: `120`
- mean confidence: `0.9293`
- valence distribution:
  - negative `89`
  - mixed `19`
  - positive `12`
- arousal distribution:
  - high `109`
  - medium `9`
  - low `2`

논문에서 조심스럽게 주장 가능한 것:

- heuristic top-emotion readout 대신 `raw trajectory -> GPT-5.4 episode interpretation`은 더 풍부한 episode 설명을 제공한다.
- 다만 아직 이것만으로 "ground-truth emotion reading"을 증명했다고 쓰면 안 된다.
- 현재는 해석기의 reportability/semantic richness를 보여주는 증거로 쓰는 것이 안전하다.

추천 표:

- 대표 사례 4~6개 qualitative case table
  - `s_000555`
  - `s_003491`
  - `s_000527`
  - `s_001913`

### 1.6 End-to-end generation comparison

최신 baseline 비교:

- [paper_refresh_summary.json](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/paper/refresh_2026-04-09_calref_v1/tables/paper_refresh_summary.json)

최신 episode-lite 비교:

- [paper_matrix_current_episode_v2_scored_summary.json](/C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/outputs/experiments/paper_matrix_current_episode_v2_scored_summary.json)

현재 가장 중요한 최신 수치:

- `stim_only = 3.5404`
- `direct = 3.5115`
- `episode_trace = 3.4673`
- `raw_trace = 3.3717`
- `hybrid_trace = 3.3268`
- `emonet_full = 3.2478`
- `appraisal_trace = 3.2345`
- `hybrid_episode = 3.2018`

논문에서 주장 가능한 것:

- raw trajectory와 episode conditioning은 naive `emonet_full`보다 낫다.
- 특히 `episode_trace`는 `emonet_full`보다 높은 total score를 보인다.
- 그러나 현재 최고 성능은 여전히 `stim_only`다.

중요한 추가 해석:

- `episode_trace` v1: `2.6239`
- `episode_trace` v2 (episode-lite): `3.4673`

즉 episode interpretation 자체보다도, 그것을 어떻게 generation surface로 넘기느냐가 성능을 크게 좌우한다.

## 2. 지금 논문에 "넣지 말아야" 하는 주장

현재 단계에서 아직 과장인 주장:

- "EmoNet이 인간처럼 감정을 느낀다"
- "trajectory interpreter가 ground-truth emotion을 정확히 복원한다"
- "episode conditioning이 전체 generation에서 최고 성능이다"
- "current z representation이 text baseline보다 우월하다"

현재 안전한 표현:

- "trajectory-level affect processing can be extracted and interpreted"
- "raw trajectory is more informative than heuristic top-emotion readout"
- "episode-lite conditioning improves substantially over naive episode injection"

## 3. 파라미터 변경 시 반드시 요구할 근거

파라미터는 아래 네 묶음으로 다뤄야 한다.

### 3.1 Ignition / selectivity 파라미터

현재 대표값:

- `k_threshold_base = 0.70`
- `k_remem_base = 1.10`
- `input_signal_clip = 0.8`
- `intrinsic_alignment_gain = 0.28`
- `recent_activity_decay = 0.3`

이 파라미터를 바꿀 수 있는 근거 지표:

- `no_activity_ratio`
- `len1_ratio`
- `mean_first_active_tick`
- `late_ignition_ratio_ge_15`

채택 기준:

- `no_activity_ratio`가 줄어들어야 함
- `len1_ratio`가 유지되거나 줄어들어야 함
- `mean_first_active_tick`이 과도하게 커지면 안 됨
- `hit_max_ticks_ratio`가 동시에 악화되면 채택 금지

즉:

- 점화를 살리려고 threshold를 낮췄다면, 그 근거는 반드시 `dead/no-activity 감소`로 보여야 한다.

### 3.2 Persistence / saturation 파라미터

현재 대표값:

- `k_decay = 0.91`
- `memory_decay = 0.97`
- `memory_k_mix = 0.35`
- `fatigue_decay = 0.9`
- `fatigue_gain = 0.25`
- `fatigue_threshold_gain = 0.18`
- `fatigue_k_leak = 0.04`
- `inhibitory_suppression_gain = 0.24`

이 파라미터를 바꿀 수 있는 근거 지표:

- `hit_max_ticks_ratio`
- branch `p90`, `p95`, `max`
- `mean_active_window_ticks`
- `saturation_ratio`
- termination reason 분포

채택 기준:

- `hit_max_ticks_ratio`가 줄어들어야 함
- upper-tail saturation이 줄어들어야 함
- 동시에 `mean_branch_len`이 지나치게 무너지면 채택 금지

즉:

- persistence를 줄이는 파라미터는 반드시 "포화 완화" 증거로 정당화해야 한다.

### 3.3 Convergence / stopping 파라미터

현재 대표값:

- `convergence_patience = 3`
- `activity_count_delta_eps = 3.0`
- `edge_count_delta_eps = 12.0`
- `activity_churn_eps = 0.01`

이 파라미터를 바꿀 수 있는 근거 지표:

- `termination_reason`
- `mean_active_window_ticks`
- `hit_max_ticks_ratio`
- `silent_tail_ticks`

채택 기준:

- `stable_convergence` 비율이 늘거나
- `max_ticks` 종료 비율이 줄어야 함
- branch mean이 collapse하면 채택 금지

즉:

- stopping rule은 "빨리 멈췄다"가 아니라 "합리적으로 수렴했다"는 증거가 필요하다.

### 3.4 Structural / extraction 파라미터

현재 대표값:

- `max_ticks = 128`
- `topk_branches = 4`
- `branch_end_window = 6`
- `branch_length_bonus = 0.35`

이 파라미터를 바꿀 수 있는 근거 지표:

- branch 분포 before/after
- `mean_path_coverage`
- `len1_ratio`
- 상위 quantile branch length

채택 기준:

- branch extraction 품질이 개선되어야 함
- 단순히 길이만 길어지고 해석성이 떨어지면 채택 금지

즉:

- structural parameter는 "보기 좋아서"가 아니라 extraction fidelity로 정당화해야 한다.

### 3.5 Generation / surface conditioning 파라미터

현재 observation:

- `episode_trace v1`은 실패
- `episode_trace v2`는 크게 회복

이 영역을 바꿀 수 있는 근거 지표:

- `mean_total`
- `naturalness`
- `overall_quality`
- `emotional_appropriateness`
- retry/error rate

채택 기준:

- `mean_total`이 오르거나
- 적어도 `naturalness`와 `overall_quality`가 개선되어야 함
- instruction echo, repetition error가 늘면 채택 금지

즉:

- generation prompt는 qualitative intuition이 아니라 judge metric으로만 정당화한다.

## 4. 파라미터 변경 프로토콜

앞으로 모든 파라미터 변경은 아래 템플릿을 따라야 한다.

1. 변경 대상 파라미터:
   - 예: `k_threshold_base 0.70 -> 0.68`

2. 가설:
   - 예: 점화 실패 샘플을 줄일 수 있다.

3. primary metric:
   - 예: `no_activity_ratio`

4. guardrail metric:
   - 예: `hit_max_ticks_ratio`, `mean_branch_len`

5. 실험 규모:
   - smoke `20`
   - selection `60`
   - adoption `200` 또는 full

6. 채택 기준:
   - pre/post 수치로 명시

이 템플릿을 못 채우면, 그 파라미터 변경은 논문 기준으로 채택하지 않는다.

## 5. 앞으로 추가로 뽑아야 할 논문용 지표

지금 있으면 좋은데 아직 계산이 부족한 것:

- bootstrap CI for `mean_total`
- paired win-rate of `episode_trace` vs `stim_only`
- per-sample delta distribution of `episode_trace - stim_only`
- episode interpretation consistency on paraphrase pairs
- causal ablation metric
  - 특정 노드/경로 약화 시 episode label 또는 action tendency가 예측 가능하게 변하는지

이 다섯 개가 들어가면 논문은 훨씬 더 단단해진다.

## 6. 현재 가장 강한 논문 메시지

현재 가장 방어 가능한 메시지는 이 세 줄이다.

1. branch collapse는 해결되었고, reference config는 calibration 실험으로 정당화되었다.
2. raw trajectory는 heuristic emotion label보다 더 풍부한 affect episode를 담고 있으며, GPT-5.4 episode interpretation이 그 정보를 더 잘 복원한다.
3. 그러나 style bias와 surface softening 때문에, 내부 affect quality가 아직 최종 generation superiority로 완전히 이어지지는 않는다.

이 메시지는 현재 데이터와 가장 잘 맞는다.
