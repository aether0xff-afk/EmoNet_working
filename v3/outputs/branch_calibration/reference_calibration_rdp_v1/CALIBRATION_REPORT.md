# Reference Config Calibration Report

## Goal

Calibrate a reference configuration with experimentally justified parameter values rather than arbitrary defaults.

## Target Constraints

- max_no_activity_ratio: `0.1`
- max_len1_ratio: `0.15`
- max_hit_max_ticks_ratio: `0.35`
- max_first_active_tick: `10.0`
- max_late_ignition_ratio: `0.1`
- min_mean_branch_len: `40.0`

## Target Operating Point

- target_first_active_tick: `4.0`
- target_branch_ratio: `0.6`
- target_active_window_ratio: `0.6`

## Center Config

```json
{
  "max_ticks": 128,
  "min_ticks_before_converged": 6,
  "convergence_patience": 4,
  "activity_count_delta_eps": 2.0,
  "edge_count_delta_eps": 12.0,
  "activity_churn_eps": 0.01,
  "k_threshold_base": 0.72,
  "k_remem_base": 1.05,
  "k_decay": 0.93,
  "refractory_ticks": 1,
  "input_topk": 2,
  "input_signal_clip": 0.8,
  "memory_decay": 0.97,
  "memory_stim_mix": 0.25,
  "memory_k_mix": 0.35,
  "state_self_stim_mix": 0.55,
  "state_parent_stim_mix": 0.25,
  "state_base_stim_mix": 0.1,
  "state_bias_stim_mix": 0.05,
  "recent_activity_decay": 0.3,
  "hysteresis_threshold_gain": 0.03,
  "hysteresis_remem_gain": 0.02,
  "hysteresis_k_bonus": 0.08,
  "intrinsic_alignment_gain": 0.24,
  "fatigue_decay": 0.9,
  "fatigue_gain": 0.2,
  "fatigue_threshold_gain": 0.18,
  "fatigue_k_leak": 0.04,
  "fire_output_log_gain": 0.5,
  "inhibitory_suppression_gain": 0.18,
  "max_out_degree": 12,
  "min_out_degree": 1,
  "dopa_rewire_gain": 0.8,
  "sero_prune_gain": 0.04,
  "mela_dropout_gain": 0.04,
  "ne_thresh_reduce_gain": 0.25,
  "ne_remem_reduce_gain": 0.25,
  "global_recovery_rate": 0.1,
  "topk_branches": 4,
  "branch_end_window": 6,
  "branch_length_bonus": 0.35
}
```

## Parameter Recommendations

### activity_count_delta_eps

- recommended_value: `3.0`
- feasible: `False`
- constraint_penalty: `0.047619`
- no_activity_ratio: `0.0667`
- len1_ratio: `0.0667`
- hit_max_ticks_ratio: `0.3667`
- mean_first_active_tick: `3.3036`
- mean_branch_len: `87.0333`
- evidence_score: `84.6968`

### convergence_patience

- recommended_value: `3`
- feasible: `True`
- constraint_penalty: `0.0`
- no_activity_ratio: `0.0167`
- len1_ratio: `0.0167`
- hit_max_ticks_ratio: `0.2333`
- mean_first_active_tick: `3.2881`
- mean_branch_len: `76.1500`
- evidence_score: `93.0780`

### fatigue_gain

- recommended_value: `0.25`
- feasible: `False`
- constraint_penalty: `0.142857`
- no_activity_ratio: `0.1000`
- len1_ratio: `0.1000`
- hit_max_ticks_ratio: `0.4000`
- mean_first_active_tick: `3.4630`
- mean_branch_len: `86.8167`
- evidence_score: `83.0803`

### inhibitory_suppression_gain

- recommended_value: `0.24`
- feasible: `False`
- constraint_penalty: `0.52381`
- no_activity_ratio: `0.0000`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `0.5333`
- mean_first_active_tick: `3.0167`
- mean_branch_len: `101.2333`
- evidence_score: `77.9453`

### intrinsic_alignment_gain

- recommended_value: `0.28`
- feasible: `False`
- constraint_penalty: `0.047619`
- no_activity_ratio: `0.0667`
- len1_ratio: `0.0667`
- hit_max_ticks_ratio: `0.3667`
- mean_first_active_tick: `2.3929`
- mean_branch_len: `86.8167`
- evidence_score: `83.0738`

### k_decay

- recommended_value: `0.91`
- feasible: `True`
- constraint_penalty: `0.0`
- no_activity_ratio: `0.0833`
- len1_ratio: `0.0833`
- hit_max_ticks_ratio: `0.3500`
- mean_first_active_tick: `3.6727`
- mean_branch_len: `79.6000`
- evidence_score: `87.9593`

### k_remem_base

- recommended_value: `1.1`
- feasible: `False`
- constraint_penalty: `0.333333`
- no_activity_ratio: `0.1333`
- len1_ratio: `0.1333`
- hit_max_ticks_ratio: `0.3500`
- mean_first_active_tick: `3.4038`
- mean_branch_len: `82.7833`
- evidence_score: `84.2116`

### k_threshold_base

- recommended_value: `0.7`
- feasible: `False`
- constraint_penalty: `0.142857`
- no_activity_ratio: `0.0833`
- len1_ratio: `0.0833`
- hit_max_ticks_ratio: `0.4000`
- mean_first_active_tick: `3.1091`
- mean_branch_len: `90.7000`
- evidence_score: `81.5665`

## Calibrated Reference Config

```json
{
  "max_ticks": 128,
  "min_ticks_before_converged": 6,
  "convergence_patience": 3,
  "activity_count_delta_eps": 3.0,
  "edge_count_delta_eps": 12.0,
  "activity_churn_eps": 0.01,
  "k_threshold_base": 0.7,
  "k_remem_base": 1.1,
  "k_decay": 0.91,
  "refractory_ticks": 1,
  "input_topk": 2,
  "input_signal_clip": 0.8,
  "memory_decay": 0.97,
  "memory_stim_mix": 0.25,
  "memory_k_mix": 0.35,
  "state_self_stim_mix": 0.55,
  "state_parent_stim_mix": 0.25,
  "state_base_stim_mix": 0.1,
  "state_bias_stim_mix": 0.05,
  "recent_activity_decay": 0.3,
  "hysteresis_threshold_gain": 0.03,
  "hysteresis_remem_gain": 0.02,
  "hysteresis_k_bonus": 0.08,
  "intrinsic_alignment_gain": 0.28,
  "fatigue_decay": 0.9,
  "fatigue_gain": 0.25,
  "fatigue_threshold_gain": 0.18,
  "fatigue_k_leak": 0.04,
  "fire_output_log_gain": 0.5,
  "inhibitory_suppression_gain": 0.24,
  "max_out_degree": 12,
  "min_out_degree": 1,
  "dopa_rewire_gain": 0.8,
  "sero_prune_gain": 0.04,
  "mela_dropout_gain": 0.04,
  "ne_thresh_reduce_gain": 0.25,
  "ne_remem_reduce_gain": 0.25,
  "global_recovery_rate": 0.1,
  "topk_branches": 4,
  "branch_end_window": 6,
  "branch_length_bonus": 0.35
}
```

## Combined Validation

```json
{
  "config_name": "combined_validation",
  "is_feasible": true,
  "constraint_penalty": 0.0,
  "constraint_failures": "",
  "no_activity_ratio": 0.05,
  "len1_ratio": 0.05,
  "hit_max_ticks_ratio": 0.25,
  "mean_first_active_tick": 2.4035087719298245,
  "late_ignition_ratio_ge_15": 0.0,
  "mean_branch_len": 79.78333333333333,
  "mean_active_window_ticks": 80.01666666666667,
  "evidence_score": 88.8412,
  "params_json": "{\"activity_churn_eps\": 0.01, \"activity_count_delta_eps\": 3.0, \"branch_end_window\": 6, \"branch_length_bonus\": 0.35, \"convergence_patience\": 3, \"dopa_rewire_gain\": 0.8, \"edge_count_delta_eps\": 12.0, \"fatigue_decay\": 0.9, \"fatigue_gain\": 0.25, \"fatigue_k_leak\": 0.04, \"fatigue_threshold_gain\": 0.18, \"fire_output_log_gain\": 0.5, \"global_recovery_rate\": 0.1, \"hysteresis_k_bonus\": 0.08, \"hysteresis_remem_gain\": 0.02, \"hysteresis_threshold_gain\": 0.03, \"inhibitory_suppression_gain\": 0.24, \"input_signal_clip\": 0.8, \"input_topk\": 2, \"intrinsic_alignment_gain\": 0.28, \"k_decay\": 0.91, \"k_remem_base\": 1.1, \"k_threshold_base\": 0.7, \"max_out_degree\": 12, \"max_ticks\": 128, \"mela_dropout_gain\": 0.04, \"memory_decay\": 0.97, \"memory_k_mix\": 0.35, \"memory_stim_mix\": 0.25, \"min_out_degree\": 1, \"min_ticks_before_converged\": 6, \"ne_remem_reduce_gain\": 0.25, \"ne_thresh_reduce_gain\": 0.25, \"recent_activity_decay\": 0.3, \"refractory_ticks\": 1, \"sero_prune_gain\": 0.04, \"state_base_stim_mix\": 0.1, \"state_bias_stim_mix\": 0.05, \"state_parent_stim_mix\": 0.25, \"state_self_stim_mix\": 0.55, \"topk_branches\": 4}"
}
```

## Evidence Table Preview

```csv
parameter_name,candidate_value,is_center,is_recommended,is_feasible,no_activity_ratio,len1_ratio,hit_max_ticks_ratio,mean_first_active_tick,mean_branch_len,evidence_score
activity_count_delta_eps,3.0,False,True,False,0.06666666666666667,0.06666666666666667,0.36666666666666664,3.3035714285714284,87.03333333333333,84.6968
activity_count_delta_eps,1.0,False,False,False,0.13333333333333333,0.13333333333333333,0.3,3.4615384615384617,83.91666666666667,84.8771
convergence_patience,3,False,True,True,0.016666666666666666,0.016666666666666666,0.23333333333333334,3.288135593220339,76.15,93.078
convergence_patience,6,False,False,False,0.05,0.05,0.7666666666666667,3.6666666666666665,108.53333333333333,69.6458
fatigue_gain,0.25,False,True,False,0.1,0.1,0.4,3.462962962962963,86.81666666666666,83.0803
fatigue_gain,0.15,False,False,False,0.016666666666666666,0.016666666666666666,0.4666666666666667,3.2542372881355934,94.01666666666667,81.8764
inhibitory_suppression_gain,0.24,False,True,False,0.0,0.0,0.5333333333333333,3.0166666666666666,101.23333333333333,77.9453
inhibitory_suppression_gain,0.12,False,False,False,0.2833333333333333,0.2833333333333333,0.26666666666666666,4.232558139534884,63.06666666666667,77.5327
intrinsic_alignment_gain,0.28,False,True,False,0.06666666666666667,0.06666666666666667,0.36666666666666664,2.392857142857143,86.81666666666666,83.0738
intrinsic_alignment_gain,0.2,False,False,False,0.05,0.05,0.4,3.754385964912281,87.98333333333333,85.171
k_decay,0.91,False,True,True,0.08333333333333333,0.08333333333333333,0.35,3.672727272727273,79.6,87.9593
k_decay,0.95,False,False,False,0.16666666666666666,0.16666666666666666,0.18333333333333332,3.52,68.41666666666667,85.4919
k_remem_base,1.1,False,True,False,0.13333333333333333,0.13333333333333333,0.35,3.4038461538461537,82.78333333333333,84.2116
k_remem_base,1.0,False,False,False,0.15,0.15,0.31666666666666665,3.627450980392157,74.98333333333333,86.2585
k_threshold_base,0.7,False,True,False,0.08333333333333333,0.08333333333333333,0.4,3.109090909090909,90.7,81.5665
k_threshold_base,0.68,False,False,False,0.11666666666666667,0.11666666666666667,0.35,3.2264150943396226,83.25,84.3633
k_threshold_base,0.74,False,False,False,0.13333333333333333,0.13333333333333333,0.35,3.480769230769231,76.01666666666667,86.3871
```

## Figures

- `activity_count_delta_eps_calibration.svg`
- `convergence_patience_calibration.svg`
- `fatigue_gain_calibration.svg`
- `inhibitory_suppression_gain_calibration.svg`
- `intrinsic_alignment_gain_calibration.svg`
- `k_decay_calibration.svg`
- `k_remem_base_calibration.svg`
- `k_threshold_base_calibration.svg`
