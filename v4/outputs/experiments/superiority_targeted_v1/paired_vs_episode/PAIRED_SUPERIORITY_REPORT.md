# Paired Superiority Analysis

Baseline condition: `episode_trace`

## Mean Total Comparisons

| condition | paired_n | delta_mean | delta_median | wins | ties | losses | win_rate | bootstrap_ci_low | bootstrap_ci_high | sign_test_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_trace_v3 | 77 | 0.451948 | 0.2 | 41 | 6 | 30 | 0.532468 | 0.14026 | 0.755844 | 0.23509756 |

## Metric-Level Comparisons

| condition | metric | paired_n | delta_mean | wins | ties | losses | win_rate | bootstrap_ci_low | bootstrap_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_trace_v3 | mean_total | 77 | 0.451948 | 41 | 6 | 30 | 0.532468 | 0.14026 | 0.755844 |
| episode_trace_v3 | appraisal_fidelity | 77 | 0.662338 | 40 | 22 | 15 | 0.519481 | 0.324675 | 1.0 |
| episode_trace_v3 | raw_affect_preservation | 77 | 0.272727 | 34 | 19 | 24 | 0.441558 | -0.168831 | 0.701299 |
| episode_trace_v3 | anti_softening | 77 | 0.428571 | 37 | 20 | 20 | 0.480519 | 0.077922 | 0.779221 |
| episode_trace_v3 | action_tendency_fit | 77 | 0.441558 | 30 | 30 | 17 | 0.38961 | 0.090909 | 0.792208 |
| episode_trace_v3 | emotional_specificity | 77 | 0.454545 | 36 | 18 | 23 | 0.467532 | 0.090909 | 0.818182 |

## Episode Subsets

| condition | subset_axis | subset_value | paired_n | delta_mean | wins | ties | losses | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_trace_v3 | arousal | low | 1 | 1.8 | 1 | 0 | 0 | 1.0 |
| episode_trace_v3 | arousal | high | 71 | 0.473239 | 38 | 6 | 27 | 0.535211 |
| episode_trace_v3 | arousal | medium | 5 | -0.12 | 2 | 0 | 3 | 0.4 |
| episode_trace_v3 | control_state | low | 19 | 0.568421 | 11 | 1 | 7 | 0.578947 |
| episode_trace_v3 | control_state | mixed | 53 | 0.467925 | 28 | 4 | 21 | 0.528302 |
| episode_trace_v3 | control_state | high | 5 | -0.16 | 2 | 1 | 2 | 0.4 |
| episode_trace_v3 | social_orientation | mixed | 21 | 0.809524 | 13 | 3 | 5 | 0.619048 |
| episode_trace_v3 | social_orientation | defend | 27 | 0.518519 | 15 | 0 | 12 | 0.555556 |
| episode_trace_v3 | social_orientation | withdraw | 6 | 0.266667 | 3 | 0 | 3 | 0.5 |
| episode_trace_v3 | social_orientation | approach | 23 | 0.095652 | 10 | 3 | 10 | 0.434783 |
| episode_trace_v3 | target | self | 3 | 1.133333 | 2 | 0 | 1 | 0.666667 |
| episode_trace_v3 | target | mixed | 53 | 0.464151 | 28 | 4 | 21 | 0.528302 |
| episode_trace_v3 | target | other | 11 | 0.381818 | 6 | 0 | 5 | 0.545455 |
| episode_trace_v3 | target | situation | 10 | 0.26 | 5 | 2 | 3 | 0.5 |
| episode_trace_v3 | valence | negative | 60 | 0.62 | 35 | 3 | 22 | 0.583333 |
| episode_trace_v3 | valence | mixed | 12 | 0.1 | 5 | 2 | 5 | 0.416667 |
| episode_trace_v3 | valence | positive | 5 | -0.72 | 1 | 1 | 3 | 0.2 |

## Largest Wins

| condition | record_id | delta_mean_total | episode_label | valence | arousal |
| --- | --- | --- | --- | --- | --- |
| episode_trace_v3 | s_001582 | 3.5999999999999996 | 후회 기반 관계위협 경계 | negative | high |
| episode_trace_v3 | s_001255 | 3.0 | 침해 망상성 경계-보복 고착 | negative | high |
| episode_trace_v3 | s_001417 | 3.0 | 처지 위축 속 해결 압박 | negative | high |
| episode_trace_v3 | s_001498 | 2.6000000000000005 | 자기비난성 부담-속죄 에피소드 | negative | high |
| episode_trace_v3 | s_000414 | 2.6 | 압박성 자기몰이 불안 | negative | high |
| episode_trace_v3 | s_000314 | 2.4 | 강요에 대한 혐오성 방어 긴장 | negative | high |
| episode_trace_v3 | s_000033 | 2.4 | 존재 공허의 경계성 고착 | negative | high |
| episode_trace_v3 | s_000929 | 2.2 | 실패회피형 경계 수렴 | negative | high |

## Largest Losses

| condition | record_id | delta_mean_total | episode_label | valence | arousal |
| --- | --- | --- | --- | --- | --- |
| episode_trace_v3 | s_003414 | -3.0 | 경계가 섞인 감격 | mixed | high |
| episode_trace_v3 | s_003918 | -2.5999999999999996 | 안도에 기반한 자부심적 만족 | positive | high |
| episode_trace_v3 | s_003539 | -1.7999999999999998 | 상실예고에 대한 경계성 먹먹함 | negative | high |
| episode_trace_v3 | s_002464 | -1.7999999999999998 | 결핍 비교로 점화된 방어적 자책-경계 | negative | high |
| episode_trace_v3 | s_002299 | -1.4 | 상실 위기 속 만회 압박 | negative | high |
| episode_trace_v3 | s_002862 | -1.2000000000000002 | 배신 해석에 점화된 공세적 경계 | negative | high |
| episode_trace_v3 | s_003321 | -1.2000000000000002 | 죄책감 기반 관계복구 추동 | mixed | high |
| episode_trace_v3 | s_000149 | -1.2000000000000002 | 신뢰 기반 기대 고양 | positive | medium |

## Artifacts

- overall CSV: `outputs\experiments\superiority_targeted_v1\paired_vs_episode\paired_overall.csv`
- subset CSV: `outputs\experiments\superiority_targeted_v1\paired_vs_episode\paired_subsets.csv`
- examples CSV: `outputs\experiments\superiority_targeted_v1\paired_vs_episode\paired_examples.csv`
- summary JSON: `outputs\experiments\superiority_targeted_v1\paired_vs_episode\paired_summary.json`
