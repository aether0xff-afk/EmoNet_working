# Branch Dynamics Optimization Report

- search_mode: `grid`
- preset: `sticky_reduction`
- sample_size: `8`
- sample_mode: `random`
- sample_seed: `42`
- model_seed: `42`
- num_workers: `1`

## Objective

Balanced score rewards low `len1_ratio`, low `hit_max_ticks_ratio`, low late ignition, and closeness to the configured branch/activation targets.

- target_branch_ratio: `0.45`
- target_first_active_tick: `4.0`
- target_active_window_ratio: `0.45`

## Constraints

- max_len1_ratio: `0.2`
- max_hit_max_ticks_ratio: `0.7`
- max_first_active_tick: `15.0`
- max_late_ignition_ratio: `0.3`
- min_mean_branch_len: `20.0`

## Search Space

```json
{
  "k_threshold_base": [
    0.68,
    0.7,
    0.72
  ],
  "intrinsic_alignment_gain": [
    0.2,
    0.24,
    0.28,
    0.32
  ]
}
```

## Fixed Center

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

## Baseline

- balanced_score: `69.1667`
- mean_branch_len: `107.1250`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `0.5000`
- mean_first_active_tick: `3.0000`
- active_window_ratio: `0.8369`

## Top Candidates

### grid:k_threshold_base=0.68;intrinsic_alignment_gain=0.2

- balanced_score: `71.5495`
- pareto_front: `True`
- feasible: `True`
- constraint_penalty: `0.000000`
- constraint_failures: ``
- mean_branch_len: `97.0000`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `0.5000`
- mean_first_active_tick: `3.0000`
- active_window_ratio: `0.7588`
- params_json: `{"activity_churn_eps": 0.01, "activity_count_delta_eps": 2.0, "branch_end_window": 6, "branch_length_bonus": 0.35, "convergence_patience": 4, "dopa_rewire_gain": 0.8, "edge_count_delta_eps": 12.0, "fatigue_decay": 0.9, "fatigue_gain": 0.2, "fatigue_k_leak": 0.04, "fatigue_threshold_gain": 0.18, "fire_output_log_gain": 0.5, "global_recovery_rate": 0.1, "hysteresis_k_bonus": 0.08, "hysteresis_remem_gain": 0.02, "hysteresis_threshold_gain": 0.03, "inhibitory_suppression_gain": 0.18, "input_signal_clip": 0.8, "input_topk": 2, "intrinsic_alignment_gain": 0.2, "k_decay": 0.93, "k_remem_base": 1.05, "k_threshold_base": 0.68, "max_out_degree": 12, "max_ticks": 128, "mela_dropout_gain": 0.04, "memory_decay": 0.97, "memory_k_mix": 0.35, "memory_stim_mix": 0.25, "min_out_degree": 1, "min_ticks_before_converged": 6, "ne_remem_reduce_gain": 0.25, "ne_thresh_reduce_gain": 0.25, "recent_activity_decay": 0.3, "refractory_ticks": 1, "sero_prune_gain": 0.04, "state_base_stim_mix": 0.1, "state_bias_stim_mix": 0.05, "state_parent_stim_mix": 0.25, "state_self_stim_mix": 0.55, "topk_branches": 4}`

### grid:k_threshold_base=0.68;intrinsic_alignment_gain=0.24

- balanced_score: `69.9516`
- pareto_front: `True`
- feasible: `True`
- constraint_penalty: `0.000000`
- constraint_failures: ``
- mean_branch_len: `100.8750`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `0.5000`
- mean_first_active_tick: `3.1250`
- active_window_ratio: `0.7881`
- params_json: `{"activity_churn_eps": 0.01, "activity_count_delta_eps": 2.0, "branch_end_window": 6, "branch_length_bonus": 0.35, "convergence_patience": 4, "dopa_rewire_gain": 0.8, "edge_count_delta_eps": 12.0, "fatigue_decay": 0.9, "fatigue_gain": 0.2, "fatigue_k_leak": 0.04, "fatigue_threshold_gain": 0.18, "fire_output_log_gain": 0.5, "global_recovery_rate": 0.1, "hysteresis_k_bonus": 0.08, "hysteresis_remem_gain": 0.02, "hysteresis_threshold_gain": 0.03, "inhibitory_suppression_gain": 0.18, "input_signal_clip": 0.8, "input_topk": 2, "intrinsic_alignment_gain": 0.24, "k_decay": 0.93, "k_remem_base": 1.05, "k_threshold_base": 0.68, "max_out_degree": 12, "max_ticks": 128, "mela_dropout_gain": 0.04, "memory_decay": 0.97, "memory_k_mix": 0.35, "memory_stim_mix": 0.25, "min_out_degree": 1, "min_ticks_before_converged": 6, "ne_remem_reduce_gain": 0.25, "ne_thresh_reduce_gain": 0.25, "recent_activity_decay": 0.3, "refractory_ticks": 1, "sero_prune_gain": 0.04, "state_base_stim_mix": 0.1, "state_bias_stim_mix": 0.05, "state_parent_stim_mix": 0.25, "state_self_stim_mix": 0.55, "topk_branches": 4}`

### baseline

- balanced_score: `69.1667`
- pareto_front: `False`
- feasible: `True`
- constraint_penalty: `0.000000`
- constraint_failures: ``
- mean_branch_len: `107.1250`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `0.5000`
- mean_first_active_tick: `3.0000`
- active_window_ratio: `0.8369`
- params_json: `{"activity_churn_eps": 0.01, "activity_count_delta_eps": 2.0, "branch_end_window": 6, "branch_length_bonus": 0.35, "convergence_patience": 4, "dopa_rewire_gain": 0.8, "edge_count_delta_eps": 12.0, "fatigue_decay": 0.9, "fatigue_gain": 0.2, "fatigue_k_leak": 0.04, "fatigue_threshold_gain": 0.18, "fire_output_log_gain": 0.5, "global_recovery_rate": 0.1, "hysteresis_k_bonus": 0.08, "hysteresis_remem_gain": 0.02, "hysteresis_threshold_gain": 0.03, "inhibitory_suppression_gain": 0.18, "input_signal_clip": 0.8, "input_topk": 2, "intrinsic_alignment_gain": 0.24, "k_decay": 0.93, "k_remem_base": 1.05, "k_threshold_base": 0.72, "max_out_degree": 12, "max_ticks": 128, "mela_dropout_gain": 0.04, "memory_decay": 0.97, "memory_k_mix": 0.35, "memory_stim_mix": 0.25, "min_out_degree": 1, "min_ticks_before_converged": 6, "ne_remem_reduce_gain": 0.25, "ne_thresh_reduce_gain": 0.25, "recent_activity_decay": 0.3, "refractory_ticks": 1, "sero_prune_gain": 0.04, "state_base_stim_mix": 0.1, "state_bias_stim_mix": 0.05, "state_parent_stim_mix": 0.25, "state_self_stim_mix": 0.55, "topk_branches": 4}`

### grid:k_threshold_base=0.68;intrinsic_alignment_gain=0.28

- balanced_score: `57.0833`
- pareto_front: `False`
- feasible: `False`
- constraint_penalty: `0.250000`
- constraint_failures: `hit_max_ticks_ratio>0.7`
- mean_branch_len: `124.2500`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `0.8750`
- mean_first_active_tick: `2.0000`
- active_window_ratio: `0.9707`
- params_json: `{"activity_churn_eps": 0.01, "activity_count_delta_eps": 2.0, "branch_end_window": 6, "branch_length_bonus": 0.35, "convergence_patience": 4, "dopa_rewire_gain": 0.8, "edge_count_delta_eps": 12.0, "fatigue_decay": 0.9, "fatigue_gain": 0.2, "fatigue_k_leak": 0.04, "fatigue_threshold_gain": 0.18, "fire_output_log_gain": 0.5, "global_recovery_rate": 0.1, "hysteresis_k_bonus": 0.08, "hysteresis_remem_gain": 0.02, "hysteresis_threshold_gain": 0.03, "inhibitory_suppression_gain": 0.18, "input_signal_clip": 0.8, "input_topk": 2, "intrinsic_alignment_gain": 0.28, "k_decay": 0.93, "k_remem_base": 1.05, "k_threshold_base": 0.68, "max_out_degree": 12, "max_ticks": 128, "mela_dropout_gain": 0.04, "memory_decay": 0.97, "memory_k_mix": 0.35, "memory_stim_mix": 0.25, "min_out_degree": 1, "min_ticks_before_converged": 6, "ne_remem_reduce_gain": 0.25, "ne_thresh_reduce_gain": 0.25, "recent_activity_decay": 0.3, "refractory_ticks": 1, "sero_prune_gain": 0.04, "state_base_stim_mix": 0.1, "state_bias_stim_mix": 0.05, "state_parent_stim_mix": 0.25, "state_self_stim_mix": 0.55, "topk_branches": 4}`

## Figures

- `optimizer_balanced_score.svg`
- `optimizer_len1_vs_hitmax.svg`
- `optimizer_activation_tradeoff.svg`
- `optimizer_top_metrics.svg`
