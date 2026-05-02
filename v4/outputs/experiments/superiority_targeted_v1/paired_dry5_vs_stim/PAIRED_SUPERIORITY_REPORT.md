# Paired Superiority Analysis

Baseline condition: `stim_only`

## Mean Total Comparisons

| condition | paired_n | delta_mean | delta_median | wins | ties | losses | win_rate | bootstrap_ci_low | bootstrap_ci_high | sign_test_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_trace | 5 | 1.16 | 1.0 | 4 | 0 | 1 | 0.8 | 0.199 | 2.2 | 0.375 |
| episode_trace_v3 | 5 | 2.32 | 2.0 | 5 | 0 | 0 | 1.0 | 1.56 | 3.16 | 0.0625 |

## Metric-Level Comparisons

| condition | metric | paired_n | delta_mean | wins | ties | losses | win_rate | bootstrap_ci_low | bootstrap_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_trace | mean_total | 5 | 1.16 | 4 | 0 | 1 | 0.8 | 0.199 | 2.2 |
| episode_trace | appraisal_fidelity | 5 | 1.4 | 5 | 0 | 0 | 1.0 | 1.0 | 2.2 |
| episode_trace | raw_affect_preservation | 5 | 0.8 | 4 | 1 | 0 | 0.8 | 0.4 | 1.0 |
| episode_trace | anti_softening | 5 | 1.0 | 3 | 1 | 1 | 0.6 | -1.6 | 3.205 |
| episode_trace | action_tendency_fit | 5 | 1.8 | 4 | 1 | 0 | 0.8 | 0.6 | 3.2 |
| episode_trace | emotional_specificity | 5 | 0.8 | 3 | 2 | 0 | 0.6 | 0.2 | 1.4 |
| episode_trace_v3 | mean_total | 5 | 2.32 | 5 | 0 | 0 | 1.0 | 1.56 | 3.16 |
| episode_trace_v3 | appraisal_fidelity | 5 | 2.4 | 5 | 0 | 0 | 1.0 | 1.4 | 3.4 |
| episode_trace_v3 | raw_affect_preservation | 5 | 1.2 | 3 | 2 | 0 | 0.6 | 0.2 | 2.2 |
| episode_trace_v3 | anti_softening | 5 | 2.6 | 4 | 1 | 0 | 0.8 | 1.2 | 3.8 |
| episode_trace_v3 | action_tendency_fit | 5 | 3.6 | 5 | 0 | 0 | 1.0 | 2.8 | 4.0 |
| episode_trace_v3 | emotional_specificity | 5 | 1.8 | 4 | 1 | 0 | 0.8 | 0.8 | 3.0 |

## Episode Subsets

| condition | subset_axis | subset_value | paired_n | delta_mean | wins | ties | losses | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_trace | arousal | high | 5 | 1.16 | 4 | 0 | 1 | 0.8 |
| episode_trace | control_state | mixed | 5 | 1.16 | 4 | 0 | 1 | 0.8 |
| episode_trace | social_orientation | defend | 3 | 1.533333 | 3 | 0 | 0 | 1.0 |
| episode_trace | social_orientation | approach | 2 | 0.6 | 1 | 0 | 1 | 0.5 |
| episode_trace | target | other | 5 | 1.16 | 4 | 0 | 1 | 0.8 |
| episode_trace | valence | negative | 5 | 1.16 | 4 | 0 | 1 | 0.8 |
| episode_trace_v3 | arousal | high | 5 | 2.32 | 5 | 0 | 0 | 1.0 |
| episode_trace_v3 | control_state | mixed | 5 | 2.32 | 5 | 0 | 0 | 1.0 |
| episode_trace_v3 | social_orientation | defend | 3 | 2.933333 | 3 | 0 | 0 | 1.0 |
| episode_trace_v3 | social_orientation | approach | 2 | 1.4 | 2 | 0 | 0 | 1.0 |
| episode_trace_v3 | target | other | 5 | 2.32 | 5 | 0 | 0 | 1.0 |
| episode_trace_v3 | valence | negative | 5 | 2.32 | 5 | 0 | 0 | 1.0 |

## Largest Wins

| condition | record_id | delta_mean_total | episode_label | valence | arousal |
| --- | --- | --- | --- | --- | --- |
| episode_trace_v3 | s_001255 | 3.5999999999999996 | 침해 망상성 경계-보복 고착 | negative | high |
| episode_trace_v3 | s_003836 | 3.2 | 단념으로 굳어진 공세적 경계 | negative | high |
| episode_trace | s_003836 | 2.8 | 단념으로 굳어진 공세적 경계 | negative | high |
| episode_trace_v3 | s_000555 | 1.9999999999999996 | 공개적 배제에 대한 공세적 당혹-분노 | negative | high |
| episode_trace_v3 | s_003887 | 1.7999999999999998 | 불공정 지각에 따른 공세적 항의 모드 | negative | high |
| episode_trace | s_003887 | 1.7999999999999998 | 불공정 지각에 따른 공세적 항의 모드 | negative | high |
| episode_trace | s_001255 | 1.0 | 침해 망상성 경계-보복 고착 | negative | high |
| episode_trace_v3 | s_002383 | 0.9999999999999998 | 부당전가에 대한 공세적 항의 준비 | negative | high |

## Largest Losses

| condition | record_id | delta_mean_total | episode_label | valence | arousal |
| --- | --- | --- | --- | --- | --- |
| episode_trace | s_002383 | -0.6000000000000001 | 부당전가에 대한 공세적 항의 준비 | negative | high |
| episode_trace | s_000555 | 0.7999999999999998 | 공개적 배제에 대한 공세적 당혹-분노 | negative | high |
| episode_trace_v3 | s_002383 | 0.9999999999999998 | 부당전가에 대한 공세적 항의 준비 | negative | high |
| episode_trace | s_001255 | 1.0 | 침해 망상성 경계-보복 고착 | negative | high |
| episode_trace_v3 | s_003887 | 1.7999999999999998 | 불공정 지각에 따른 공세적 항의 모드 | negative | high |
| episode_trace | s_003887 | 1.7999999999999998 | 불공정 지각에 따른 공세적 항의 모드 | negative | high |
| episode_trace_v3 | s_000555 | 1.9999999999999996 | 공개적 배제에 대한 공세적 당혹-분노 | negative | high |
| episode_trace | s_003836 | 2.8 | 단념으로 굳어진 공세적 경계 | negative | high |

## Artifacts

- overall CSV: `outputs\experiments\superiority_targeted_v1\paired_dry5_vs_stim\paired_overall.csv`
- subset CSV: `outputs\experiments\superiority_targeted_v1\paired_dry5_vs_stim\paired_subsets.csv`
- examples CSV: `outputs\experiments\superiority_targeted_v1\paired_dry5_vs_stim\paired_examples.csv`
- summary JSON: `outputs\experiments\superiority_targeted_v1\paired_dry5_vs_stim\paired_summary.json`
