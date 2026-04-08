# Branch Dynamics Optimization Report

- search_mode: `random`
- preset: `sticky_reduction`
- sample_size: `40`
- sample_mode: `random`
- sample_seed: `42`
- model_seed: `42`
- num_workers: `8`

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
  "convergence_patience": [
    4,
    6,
    8
  ],
  "activity_count_delta_eps": [
    1.0,
    2.0,
    3.0
  ],
  "edge_count_delta_eps": [
    8.0,
    12.0,
    20.0
  ],
  "activity_churn_eps": [
    0.01,
    0.02,
    0.05
  ],
  "k_threshold_base": [
    0.72,
    0.8,
    0.9,
    1.0
  ],
  "k_remem_base": [
    0.95,
    1.05,
    1.15
  ],
  "k_decay": [
    0.93,
    0.95,
    0.97,
    0.99
  ],
  "refractory_ticks": [
    1,
    2,
    3
  ],
  "input_signal_clip": [
    0.8,
    1.0,
    1.2,
    1.5
  ],
  "recent_activity_decay": [
    0.3,
    0.5,
    0.7,
    0.8
  ],
  "hysteresis_threshold_gain": [
    0.0,
    0.03,
    0.06,
    0.12
  ],
  "hysteresis_remem_gain": [
    0.0,
    0.02,
    0.04,
    0.08
  ],
  "hysteresis_k_bonus": [
    0.0,
    0.02,
    0.04,
    0.08
  ],
  "intrinsic_alignment_gain": [
    0.16,
    0.24,
    0.32
  ],
  "fatigue_decay": [
    0.85,
    0.9,
    0.95
  ],
  "fatigue_gain": [
    0.2,
    0.3,
    0.4
  ],
  "fatigue_threshold_gain": [
    0.12,
    0.18,
    0.28
  ],
  "fatigue_k_leak": [
    0.04,
    0.08,
    0.12
  ],
  "fire_output_log_gain": [
    0.5,
    0.75,
    1.25
  ],
  "inhibitory_suppression_gain": [
    0.1,
    0.18,
    0.3
  ],
  "memory_decay": [
    0.97,
    0.98,
    0.985
  ],
  "memory_k_mix": [
    0.0,
    0.1,
    0.2,
    0.35
  ],
  "state_base_stim_mix": [
    0.05,
    0.1,
    0.15
  ]
}
```

## Fixed Center

```json
{
  "max_ticks": 128,
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

## Baseline

- balanced_score: `82.5818`
- mean_branch_len: `54.1750`
- len1_ratio: `0.3750`
- hit_max_ticks_ratio: `0.3000`
- mean_first_active_tick: `4.6800`
- active_window_ratio: `0.4207`

## Top Candidates

### random:convergence_patience=4;activity_count_delta_eps=2.0;edge_count_delta_eps=12.0;activity_churn_eps=0.01;k_threshold_base=0.72;k_remem_base=1.05;k_decay=0.93;refractory_ticks=1;input_signal_clip=0.8;recent_activity_decay=0.3;hysteresis_threshold_gain=0.03;hysteresis_remem_gain=0.02;hysteresis_k_bonus=0.08;intrinsic_alignment_gain=0.24;fatigue_decay=0.9;fatigue_gain=0.2;fatigue_threshold_gain=0.18;fatigue_k_leak=0.04;fire_output_log_gain=0.5;inhibitory_suppression_gain=0.18;memory_decay=0.97;memory_k_mix=0.35;state_base_stim_mix=0.1

- balanced_score: `79.5242`
- pareto_front: `True`
- feasible: `True`
- constraint_penalty: `0.000000`
- constraint_failures: ``
- mean_branch_len: `80.2000`
- len1_ratio: `0.1250`
- hit_max_ticks_ratio: `0.3750`
- mean_first_active_tick: `3.5429`
- active_window_ratio: `0.6268`
- params_json: `{"activity_churn_eps": 0.01, "activity_count_delta_eps": 2.0, "branch_end_window": 6, "branch_length_bonus": 0.35, "convergence_patience": 4, "dopa_rewire_gain": 0.8, "edge_count_delta_eps": 12.0, "fatigue_decay": 0.9, "fatigue_gain": 0.2, "fatigue_k_leak": 0.04, "fatigue_threshold_gain": 0.18, "fire_output_log_gain": 0.5, "global_recovery_rate": 0.1, "hysteresis_k_bonus": 0.08, "hysteresis_remem_gain": 0.02, "hysteresis_threshold_gain": 0.03, "inhibitory_suppression_gain": 0.18, "input_signal_clip": 0.8, "input_topk": 2, "intrinsic_alignment_gain": 0.24, "k_decay": 0.93, "k_remem_base": 1.05, "k_threshold_base": 0.72, "max_out_degree": 12, "max_ticks": 128, "mela_dropout_gain": 0.04, "memory_decay": 0.97, "memory_k_mix": 0.35, "memory_stim_mix": 0.25, "min_out_degree": 1, "min_ticks_before_converged": 6, "ne_remem_reduce_gain": 0.25, "ne_thresh_reduce_gain": 0.25, "recent_activity_decay": 0.3, "refractory_ticks": 1, "sero_prune_gain": 0.04, "state_base_stim_mix": 0.1, "state_bias_stim_mix": 0.05, "state_parent_stim_mix": 0.25, "state_self_stim_mix": 0.55, "topk_branches": 4}`

### random:convergence_patience=4;activity_count_delta_eps=2.0;edge_count_delta_eps=20.0;activity_churn_eps=0.05;k_threshold_base=0.8;k_remem_base=1.05;k_decay=0.95;refractory_ticks=1;input_signal_clip=1.2;recent_activity_decay=0.7;hysteresis_threshold_gain=0.0;hysteresis_remem_gain=0.04;hysteresis_k_bonus=0.02;intrinsic_alignment_gain=0.32;fatigue_decay=0.85;fatigue_gain=0.4;fatigue_threshold_gain=0.12;fatigue_k_leak=0.08;fire_output_log_gain=1.25;inhibitory_suppression_gain=0.18;memory_decay=0.985;memory_k_mix=0.1;state_base_stim_mix=0.05

- balanced_score: `80.3263`
- pareto_front: `True`
- feasible: `False`
- constraint_penalty: `0.140000`
- constraint_failures: `mean_branch_len<20.0`
- mean_branch_len: `17.2000`
- len1_ratio: `0.2000`
- hit_max_ticks_ratio: `0.0000`
- mean_first_active_tick: `4.7188`
- active_window_ratio: `0.1330`
- params_json: `{"activity_churn_eps": 0.05, "activity_count_delta_eps": 2.0, "branch_end_window": 6, "branch_length_bonus": 0.35, "convergence_patience": 4, "dopa_rewire_gain": 0.8, "edge_count_delta_eps": 20.0, "fatigue_decay": 0.85, "fatigue_gain": 0.4, "fatigue_k_leak": 0.08, "fatigue_threshold_gain": 0.12, "fire_output_log_gain": 1.25, "global_recovery_rate": 0.1, "hysteresis_k_bonus": 0.02, "hysteresis_remem_gain": 0.04, "hysteresis_threshold_gain": 0.0, "inhibitory_suppression_gain": 0.18, "input_signal_clip": 1.2, "input_topk": 2, "intrinsic_alignment_gain": 0.32, "k_decay": 0.95, "k_remem_base": 1.05, "k_threshold_base": 0.8, "max_out_degree": 12, "max_ticks": 128, "mela_dropout_gain": 0.04, "memory_decay": 0.985, "memory_k_mix": 0.1, "memory_stim_mix": 0.25, "min_out_degree": 1, "min_ticks_before_converged": 6, "ne_remem_reduce_gain": 0.25, "ne_thresh_reduce_gain": 0.25, "recent_activity_decay": 0.7, "refractory_ticks": 1, "sero_prune_gain": 0.04, "state_base_stim_mix": 0.05, "state_bias_stim_mix": 0.05, "state_parent_stim_mix": 0.25, "state_self_stim_mix": 0.55, "topk_branches": 4}`

### random:convergence_patience=6;activity_count_delta_eps=3.0;edge_count_delta_eps=12.0;activity_churn_eps=0.01;k_threshold_base=0.8;k_remem_base=1.15;k_decay=0.99;refractory_ticks=2;input_signal_clip=1.2;recent_activity_decay=0.5;hysteresis_threshold_gain=0.03;hysteresis_remem_gain=0.04;hysteresis_k_bonus=0.0;intrinsic_alignment_gain=0.16;fatigue_decay=0.9;fatigue_gain=0.2;fatigue_threshold_gain=0.18;fatigue_k_leak=0.08;fire_output_log_gain=1.25;inhibitory_suppression_gain=0.18;memory_decay=0.97;memory_k_mix=0.35;state_base_stim_mix=0.15

- balanced_score: `56.3132`
- pareto_front: `False`
- feasible: `False`
- constraint_penalty: `0.142857`
- constraint_failures: `hit_max_ticks_ratio>0.7`
- mean_branch_len: `90.2750`
- len1_ratio: `0.2000`
- hit_max_ticks_ratio: `0.8000`
- mean_first_active_tick: `7.2188`
- active_window_ratio: `0.7549`
- params_json: `{"activity_churn_eps": 0.01, "activity_count_delta_eps": 3.0, "branch_end_window": 6, "branch_length_bonus": 0.35, "convergence_patience": 6, "dopa_rewire_gain": 0.8, "edge_count_delta_eps": 12.0, "fatigue_decay": 0.9, "fatigue_gain": 0.2, "fatigue_k_leak": 0.08, "fatigue_threshold_gain": 0.18, "fire_output_log_gain": 1.25, "global_recovery_rate": 0.1, "hysteresis_k_bonus": 0.0, "hysteresis_remem_gain": 0.04, "hysteresis_threshold_gain": 0.03, "inhibitory_suppression_gain": 0.18, "input_signal_clip": 1.2, "input_topk": 2, "intrinsic_alignment_gain": 0.16, "k_decay": 0.99, "k_remem_base": 1.15, "k_threshold_base": 0.8, "max_out_degree": 12, "max_ticks": 128, "mela_dropout_gain": 0.04, "memory_decay": 0.97, "memory_k_mix": 0.35, "memory_stim_mix": 0.25, "min_out_degree": 1, "min_ticks_before_converged": 6, "ne_remem_reduce_gain": 0.25, "ne_thresh_reduce_gain": 0.25, "recent_activity_decay": 0.5, "refractory_ticks": 2, "sero_prune_gain": 0.04, "state_base_stim_mix": 0.15, "state_bias_stim_mix": 0.05, "state_parent_stim_mix": 0.25, "state_self_stim_mix": 0.55, "topk_branches": 4}`

### random:convergence_patience=4;activity_count_delta_eps=2.0;edge_count_delta_eps=20.0;activity_churn_eps=0.05;k_threshold_base=1.0;k_remem_base=0.95;k_decay=0.95;refractory_ticks=3;input_signal_clip=1.2;recent_activity_decay=0.8;hysteresis_threshold_gain=0.06;hysteresis_remem_gain=0.08;hysteresis_k_bonus=0.08;intrinsic_alignment_gain=0.16;fatigue_decay=0.85;fatigue_gain=0.2;fatigue_threshold_gain=0.12;fatigue_k_leak=0.08;fire_output_log_gain=0.5;inhibitory_suppression_gain=0.3;memory_decay=0.985;memory_k_mix=0.1;state_base_stim_mix=0.15

- balanced_score: `61.2520`
- pareto_front: `True`
- feasible: `False`
- constraint_penalty: `0.232143`
- constraint_failures: `len1_ratio>0.2;hit_max_ticks_ratio>0.7`
- mean_branch_len: `71.6250`
- len1_ratio: `0.2250`
- hit_max_ticks_ratio: `0.7750`
- mean_first_active_tick: `7.0645`
- active_window_ratio: `0.7322`
- params_json: `{"activity_churn_eps": 0.05, "activity_count_delta_eps": 2.0, "branch_end_window": 6, "branch_length_bonus": 0.35, "convergence_patience": 4, "dopa_rewire_gain": 0.8, "edge_count_delta_eps": 20.0, "fatigue_decay": 0.85, "fatigue_gain": 0.2, "fatigue_k_leak": 0.08, "fatigue_threshold_gain": 0.12, "fire_output_log_gain": 0.5, "global_recovery_rate": 0.1, "hysteresis_k_bonus": 0.08, "hysteresis_remem_gain": 0.08, "hysteresis_threshold_gain": 0.06, "inhibitory_suppression_gain": 0.3, "input_signal_clip": 1.2, "input_topk": 2, "intrinsic_alignment_gain": 0.16, "k_decay": 0.95, "k_remem_base": 0.95, "k_threshold_base": 1.0, "max_out_degree": 12, "max_ticks": 128, "mela_dropout_gain": 0.04, "memory_decay": 0.985, "memory_k_mix": 0.1, "memory_stim_mix": 0.25, "min_out_degree": 1, "min_ticks_before_converged": 6, "ne_remem_reduce_gain": 0.25, "ne_thresh_reduce_gain": 0.25, "recent_activity_decay": 0.8, "refractory_ticks": 3, "sero_prune_gain": 0.04, "state_base_stim_mix": 0.15, "state_bias_stim_mix": 0.05, "state_parent_stim_mix": 0.25, "state_self_stim_mix": 0.55, "topk_branches": 4}`

### random:convergence_patience=8;activity_count_delta_eps=3.0;edge_count_delta_eps=8.0;activity_churn_eps=0.05;k_threshold_base=0.9;k_remem_base=1.15;k_decay=0.93;refractory_ticks=1;input_signal_clip=1.2;recent_activity_decay=0.3;hysteresis_threshold_gain=0.0;hysteresis_remem_gain=0.02;hysteresis_k_bonus=0.04;intrinsic_alignment_gain=0.24;fatigue_decay=0.95;fatigue_gain=0.2;fatigue_threshold_gain=0.28;fatigue_k_leak=0.08;fire_output_log_gain=0.5;inhibitory_suppression_gain=0.3;memory_decay=0.985;memory_k_mix=0.2;state_base_stim_mix=0.15

- balanced_score: `73.5972`
- pareto_front: `True`
- feasible: `False`
- constraint_penalty: `0.375000`
- constraint_failures: `len1_ratio>0.2`
- mean_branch_len: `74.9000`
- len1_ratio: `0.2750`
- hit_max_ticks_ratio: `0.4750`
- mean_first_active_tick: `5.2414`
- active_window_ratio: `0.5902`
- params_json: `{"activity_churn_eps": 0.05, "activity_count_delta_eps": 3.0, "branch_end_window": 6, "branch_length_bonus": 0.35, "convergence_patience": 8, "dopa_rewire_gain": 0.8, "edge_count_delta_eps": 8.0, "fatigue_decay": 0.95, "fatigue_gain": 0.2, "fatigue_k_leak": 0.08, "fatigue_threshold_gain": 0.28, "fire_output_log_gain": 0.5, "global_recovery_rate": 0.1, "hysteresis_k_bonus": 0.04, "hysteresis_remem_gain": 0.02, "hysteresis_threshold_gain": 0.0, "inhibitory_suppression_gain": 0.3, "input_signal_clip": 1.2, "input_topk": 2, "intrinsic_alignment_gain": 0.24, "k_decay": 0.93, "k_remem_base": 1.15, "k_threshold_base": 0.9, "max_out_degree": 12, "max_ticks": 128, "mela_dropout_gain": 0.04, "memory_decay": 0.985, "memory_k_mix": 0.2, "memory_stim_mix": 0.25, "min_out_degree": 1, "min_ticks_before_converged": 6, "ne_remem_reduce_gain": 0.25, "ne_thresh_reduce_gain": 0.25, "recent_activity_decay": 0.3, "refractory_ticks": 1, "sero_prune_gain": 0.04, "state_base_stim_mix": 0.15, "state_bias_stim_mix": 0.05, "state_parent_stim_mix": 0.25, "state_self_stim_mix": 0.55, "topk_branches": 4}`

## Figures

- `optimizer_balanced_score.svg`
- `optimizer_len1_vs_hitmax.svg`
- `optimizer_activation_tradeoff.svg`
- `optimizer_top_metrics.svg`
