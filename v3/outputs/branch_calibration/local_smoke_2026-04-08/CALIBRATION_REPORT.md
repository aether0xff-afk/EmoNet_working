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
  "max_ticks": 32,
  "min_ticks_before_converged": 6,
  "convergence_patience": 6,
  "activity_count_delta_eps": 2.0,
  "edge_count_delta_eps": 12.0,
  "activity_churn_eps": 0.02,
  "k_threshold_base": 0.72,
  "k_remem_base": 0.95,
  "k_decay": 0.99,
  "refractory_ticks": 1,
  "input_topk": 2,
  "input_signal_clip": 1.5,
  "memory_decay": 0.985,
  "memory_stim_mix": 0.25,
  "memory_k_mix": 0.35,
  "state_self_stim_mix": 0.55,
  "state_parent_stim_mix": 0.25,
  "state_base_stim_mix": 0.15,
  "state_bias_stim_mix": 0.05,
  "recent_activity_decay": 0.8,
  "hysteresis_threshold_gain": 0.12,
  "hysteresis_remem_gain": 0.08,
  "hysteresis_k_bonus": 0.08,
  "intrinsic_alignment_gain": 0.24,
  "fatigue_decay": 0.9,
  "fatigue_gain": 0.3,
  "fatigue_threshold_gain": 0.18,
  "fatigue_k_leak": 0.08,
  "fire_output_log_gain": 0.75,
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

### intrinsic_alignment_gain

- recommended_value: `0.2`
- feasible: `False`
- constraint_penalty: `5.419524`
- no_activity_ratio: `0.4000`
- len1_ratio: `0.4000`
- hit_max_ticks_ratio: `0.4000`
- mean_first_active_tick: `6.0000`
- mean_branch_len: `15.6000`
- evidence_score: `66.6250`

### k_threshold_base

- recommended_value: `0.7`
- feasible: `False`
- constraint_penalty: `5.399524`
- no_activity_ratio: `0.4000`
- len1_ratio: `0.4000`
- hit_max_ticks_ratio: `0.4000`
- mean_first_active_tick: `4.3333`
- mean_branch_len: `16.4000`
- evidence_score: `71.0000`

## Calibrated Reference Config

```json
{
  "max_ticks": 32,
  "min_ticks_before_converged": 6,
  "convergence_patience": 6,
  "activity_count_delta_eps": 2.0,
  "edge_count_delta_eps": 12.0,
  "activity_churn_eps": 0.02,
  "k_threshold_base": 0.7,
  "k_remem_base": 0.95,
  "k_decay": 0.99,
  "refractory_ticks": 1,
  "input_topk": 2,
  "input_signal_clip": 1.5,
  "memory_decay": 0.985,
  "memory_stim_mix": 0.25,
  "memory_k_mix": 0.35,
  "state_self_stim_mix": 0.55,
  "state_parent_stim_mix": 0.25,
  "state_base_stim_mix": 0.15,
  "state_bias_stim_mix": 0.05,
  "recent_activity_decay": 0.8,
  "hysteresis_threshold_gain": 0.12,
  "hysteresis_remem_gain": 0.08,
  "hysteresis_k_bonus": 0.08,
  "intrinsic_alignment_gain": 0.2,
  "fatigue_decay": 0.9,
  "fatigue_gain": 0.3,
  "fatigue_threshold_gain": 0.18,
  "fatigue_k_leak": 0.08,
  "fire_output_log_gain": 0.75,
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

## Combined Validation

```json
{
  "config_name": "combined_validation",
  "is_feasible": false,
  "constraint_penalty": 5.980952,
  "constraint_failures": "no_activity_ratio>0.1;len1_ratio>0.15;hit_max_ticks_ratio>0.35;mean_branch_len<40.0",
  "no_activity_ratio": 0.4,
  "len1_ratio": 0.4,
  "hit_max_ticks_ratio": 0.6,
  "mean_first_active_tick": 5.666666666666667,
  "late_ignition_ratio_ge_15": 0.0,
  "mean_branch_len": 16.0,
  "mean_active_window_ticks": 15.8,
  "evidence_score": 63.875,
  "params_json": "{\"activity_churn_eps\": 0.02, \"activity_count_delta_eps\": 2.0, \"branch_end_window\": 6, \"branch_length_bonus\": 0.35, \"convergence_patience\": 6, \"dopa_rewire_gain\": 0.8, \"edge_count_delta_eps\": 12.0, \"fatigue_decay\": 0.9, \"fatigue_gain\": 0.3, \"fatigue_k_leak\": 0.08, \"fatigue_threshold_gain\": 0.18, \"fire_output_log_gain\": 0.75, \"global_recovery_rate\": 0.1, \"hysteresis_k_bonus\": 0.08, \"hysteresis_remem_gain\": 0.08, \"hysteresis_threshold_gain\": 0.12, \"inhibitory_suppression_gain\": 0.18, \"input_signal_clip\": 1.5, \"input_topk\": 2, \"intrinsic_alignment_gain\": 0.2, \"k_decay\": 0.99, \"k_remem_base\": 0.95, \"k_threshold_base\": 0.7, \"max_out_degree\": 12, \"max_ticks\": 32, \"mela_dropout_gain\": 0.04, \"memory_decay\": 0.985, \"memory_k_mix\": 0.35, \"memory_stim_mix\": 0.25, \"min_out_degree\": 1, \"min_ticks_before_converged\": 6, \"ne_remem_reduce_gain\": 0.25, \"ne_thresh_reduce_gain\": 0.25, \"recent_activity_decay\": 0.8, \"refractory_ticks\": 1, \"sero_prune_gain\": 0.04, \"state_base_stim_mix\": 0.15, \"state_bias_stim_mix\": 0.05, \"state_parent_stim_mix\": 0.25, \"state_self_stim_mix\": 0.55, \"topk_branches\": 4}"
}
```

## Evidence Table Preview

```csv
parameter_name,candidate_value,is_center,is_recommended,is_feasible,no_activity_ratio,len1_ratio,hit_max_ticks_ratio,mean_first_active_tick,mean_branch_len,evidence_score
intrinsic_alignment_gain,0.2,False,True,False,0.4,0.4,0.4,6.0,15.6,66.625
k_threshold_base,0.7,False,True,False,0.4,0.4,0.4,4.333333333333333,16.4,71.0
```

## Figures

- `intrinsic_alignment_gain_calibration.svg`
- `k_threshold_base_calibration.svg`
