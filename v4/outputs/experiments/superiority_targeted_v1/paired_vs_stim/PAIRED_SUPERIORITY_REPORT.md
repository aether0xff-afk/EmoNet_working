# Paired Superiority Analysis

Baseline condition: `stim_only`

## Mean Total Comparisons

| condition | paired_n | delta_mean | delta_median | wins | ties | losses | win_rate | bootstrap_ci_low | bootstrap_ci_high | sign_test_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_trace | 77 | 1.415584 | 1.6 | 69 | 6 | 2 | 0.896104 | 1.174026 | 1.651948 | 0.0 |
| episode_trace_v3 | 78 | 1.830769 | 2.0 | 70 | 3 | 5 | 0.897436 | 1.566667 | 2.089744 | 0.0 |

## Metric-Level Comparisons

| condition | metric | paired_n | delta_mean | wins | ties | losses | win_rate | bootstrap_ci_low | bootstrap_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_trace | mean_total | 77 | 1.415584 | 69 | 6 | 2 | 0.896104 | 1.174026 | 1.651948 |
| episode_trace | appraisal_fidelity | 77 | 1.324675 | 60 | 14 | 3 | 0.779221 | 1.051948 | 1.597403 |
| episode_trace | raw_affect_preservation | 77 | 1.246753 | 50 | 25 | 2 | 0.649351 | 0.961039 | 1.532468 |
| episode_trace | anti_softening | 77 | 1.792208 | 62 | 13 | 2 | 0.805195 | 1.454545 | 2.103896 |
| episode_trace | action_tendency_fit | 77 | 1.311688 | 51 | 21 | 5 | 0.662338 | 0.974026 | 1.636364 |
| episode_trace | emotional_specificity | 77 | 1.402597 | 59 | 16 | 2 | 0.766234 | 1.12987 | 1.662338 |
| episode_trace_v3 | mean_total | 78 | 1.830769 | 70 | 3 | 5 | 0.897436 | 1.566667 | 2.089744 |
| episode_trace_v3 | appraisal_fidelity | 78 | 1.948718 | 65 | 9 | 4 | 0.833333 | 1.628205 | 2.25641 |
| episode_trace_v3 | raw_affect_preservation | 78 | 1.5 | 58 | 15 | 5 | 0.74359 | 1.179487 | 1.820513 |
| episode_trace_v3 | anti_softening | 78 | 2.153846 | 67 | 5 | 6 | 0.858974 | 1.794872 | 2.487179 |
| episode_trace_v3 | action_tendency_fit | 78 | 1.717949 | 57 | 18 | 3 | 0.730769 | 1.358974 | 2.076923 |
| episode_trace_v3 | emotional_specificity | 78 | 1.833333 | 65 | 11 | 2 | 0.833333 | 1.538462 | 2.128205 |

## Episode Subsets

| condition | subset_axis | subset_value | paired_n | delta_mean | wins | ties | losses | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_trace | arousal | high | 71 | 1.473239 | 65 | 4 | 2 | 0.915493 |
| episode_trace | arousal | medium | 5 | 0.8 | 3 | 2 | 0 | 0.6 |
| episode_trace | arousal | low | 1 | 0.4 | 1 | 0 | 0 | 1.0 |
| episode_trace | control_state | low | 19 | 1.547368 | 18 | 1 | 0 | 0.947368 |
| episode_trace | control_state | mixed | 53 | 1.388679 | 47 | 4 | 2 | 0.886792 |
| episode_trace | control_state | high | 5 | 1.2 | 4 | 1 | 0 | 0.8 |
| episode_trace | social_orientation | defend | 27 | 1.585185 | 24 | 3 | 0 | 0.888889 |
| episode_trace | social_orientation | approach | 23 | 1.426087 | 20 | 2 | 1 | 0.869565 |
| episode_trace | social_orientation | withdraw | 6 | 1.4 | 6 | 0 | 0 | 1.0 |
| episode_trace | social_orientation | mixed | 21 | 1.190476 | 19 | 1 | 1 | 0.904762 |
| episode_trace | target | situation | 10 | 1.54 | 10 | 0 | 0 | 1.0 |
| episode_trace | target | other | 11 | 1.472727 | 10 | 0 | 1 | 0.909091 |
| episode_trace | target | mixed | 53 | 1.4 | 47 | 5 | 1 | 0.886792 |
| episode_trace | target | self | 3 | 1.066667 | 2 | 1 | 0 | 0.666667 |
| episode_trace | valence | mixed | 12 | 1.583333 | 11 | 0 | 1 | 0.916667 |
| episode_trace | valence | negative | 60 | 1.413333 | 55 | 4 | 1 | 0.916667 |
| episode_trace | valence | positive | 5 | 1.04 | 3 | 2 | 0 | 0.6 |
| episode_trace_v3 | arousal | low | 1 | 2.2 | 1 | 0 | 0 | 1.0 |
| episode_trace_v3 | arousal | high | 72 | 1.905556 | 66 | 2 | 4 | 0.916667 |
| episode_trace_v3 | arousal | medium | 5 | 0.68 | 3 | 1 | 1 | 0.6 |
| episode_trace_v3 | control_state | low | 19 | 2.115789 | 17 | 2 | 0 | 0.894737 |
| episode_trace_v3 | control_state | mixed | 54 | 1.803704 | 49 | 1 | 4 | 0.907407 |
| episode_trace_v3 | control_state | high | 5 | 1.04 | 4 | 0 | 1 | 0.8 |
| episode_trace_v3 | social_orientation | mixed | 21 | 2.0 | 20 | 0 | 1 | 0.952381 |
| episode_trace_v3 | social_orientation | defend | 28 | 1.992857 | 27 | 0 | 1 | 0.964286 |
| episode_trace_v3 | social_orientation | withdraw | 6 | 1.666667 | 4 | 2 | 0 | 0.666667 |
| episode_trace_v3 | social_orientation | approach | 23 | 1.521739 | 19 | 1 | 3 | 0.826087 |
| episode_trace_v3 | target | self | 3 | 2.2 | 3 | 0 | 0 | 1.0 |
| episode_trace_v3 | target | mixed | 53 | 1.864151 | 48 | 2 | 3 | 0.90566 |
| episode_trace_v3 | target | situation | 10 | 1.8 | 9 | 1 | 0 | 0.9 |
| episode_trace_v3 | target | other | 12 | 1.616667 | 10 | 0 | 2 | 0.833333 |
| episode_trace_v3 | valence | negative | 61 | 1.983607 | 57 | 2 | 2 | 0.934426 |
| episode_trace_v3 | valence | mixed | 12 | 1.683333 | 10 | 1 | 1 | 0.833333 |
| episode_trace_v3 | valence | positive | 5 | 0.32 | 3 | 0 | 2 | 0.6 |

## Largest Wins

| condition | record_id | delta_mean_total | episode_label | valence | arousal |
| --- | --- | --- | --- | --- | --- |
| episode_trace | s_001987 | 4.0 | 불안이 섞인 성취기대 | mixed | high |
| episode_trace_v3 | s_000314 | 4.0 | 강요에 대한 혐오성 방어 긴장 | negative | high |
| episode_trace_v3 | s_001255 | 4.0 | 침해 망상성 경계-보복 고착 | negative | high |
| episode_trace_v3 | s_002131 | 4.0 | 피해 고립에 대한 공세적 경계 고착 | negative | high |
| episode_trace | s_003414 | 3.8 | 경계가 섞인 감격 | mixed | high |
| episode_trace_v3 | s_001417 | 3.8 | 처지 위축 속 해결 압박 | negative | high |
| episode_trace_v3 | s_001582 | 3.8 | 후회 기반 관계위협 경계 | negative | high |
| episode_trace | s_002299 | 3.8 | 상실 위기 속 만회 압박 | negative | high |

## Largest Losses

| condition | record_id | delta_mean_total | episode_label | valence | arousal |
| --- | --- | --- | --- | --- | --- |
| episode_trace | s_000211 | -1.6 | 기대와 경계가 엉킨 가족재편 불안 | mixed | high |
| episode_trace_v3 | s_000149 | -1.2000000000000002 | 신뢰 기반 기대 고양 | positive | medium |
| episode_trace_v3 | s_000211 | -1.0 | 기대와 경계가 엉킨 가족재편 불안 | mixed | high |
| episode_trace_v3 | s_002456 | -1.0 | 명예퍄손 위협에 대한 경계적 반격 | negative | high |
| episode_trace | s_002383 | -0.6000000000000001 | 부당전가에 대한 공세적 항의 준비 | negative | high |
| episode_trace_v3 | s_003918 | -0.40000000000000013 | 안도에 기반한 자부심적 만족 | positive | high |
| episode_trace_v3 | s_002383 | -0.40000000000000013 | 부당전가에 대한 공세적 항의 준비 | negative | high |
| episode_trace | s_000149 | 0.0 | 신뢰 기반 기대 고양 | positive | medium |

## Artifacts

- overall CSV: `outputs\experiments\superiority_targeted_v1\paired_vs_stim\paired_overall.csv`
- subset CSV: `outputs\experiments\superiority_targeted_v1\paired_vs_stim\paired_subsets.csv`
- examples CSV: `outputs\experiments\superiority_targeted_v1\paired_vs_stim\paired_examples.csv`
- summary JSON: `outputs\experiments\superiority_targeted_v1\paired_vs_stim\paired_summary.json`
