# Branch Dynamics Optimization Report

- search_mode: `random`
- preset: `sticky_reduction`
- sample_size: `2`
- sample_mode: `random`
- sample_seed: `42`
- model_seed: `42`
- num_workers: `2`

## Objective

Balanced score rewards low `len1_ratio`, low `hit_max_ticks_ratio`, low late ignition, and closeness to the configured branch/activation targets.

- target_branch_ratio: `0.45`
- target_first_active_tick: `4.0`
- target_active_window_ratio: `0.45`

## Constraints

- max_len1_ratio: `0.3`
- max_hit_max_ticks_ratio: `0.8`
- max_first_active_tick: `15.0`
- max_late_ignition_ratio: `0.5`
- min_mean_branch_len: `8.0`

## Search Space

```json
{
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
  "max_ticks": 32,
  "min_ticks_before_converged": 6,
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

- balanced_score: `53.2143`
- mean_branch_len: `12.0000`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `1.0000`
- mean_first_active_tick: `20.0000`
- active_window_ratio: `0.3750`

## Top Candidates

### baseline

- balanced_score: `53.2143`
- pareto_front: `True`
- feasible: `False`
- constraint_penalty: `0.583333`
- constraint_failures: `hit_max_ticks_ratio>0.8;mean_first_active_tick>15.0`
- mean_branch_len: `12.0000`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `1.0000`
- mean_first_active_tick: `20.0000`
- active_window_ratio: `0.3750`
- params_json: `{"branch_end_window": 6, "branch_length_bonus": 0.35, "dopa_rewire_gain": 0.8, "global_recovery_rate": 0.1, "hysteresis_k_bonus": 0.08, "hysteresis_remem_gain": 0.08, "hysteresis_threshold_gain": 0.12, "input_signal_clip": 1.5, "input_topk": 2, "k_decay": 0.99, "k_remem_base": 0.95, "k_threshold_base": 0.72, "max_out_degree": 12, "max_ticks": 32, "mela_dropout_gain": 0.04, "memory_decay": 0.985, "memory_k_mix": 0.35, "memory_stim_mix": 0.25, "min_out_degree": 1, "min_ticks_before_converged": 6, "ne_remem_reduce_gain": 0.25, "ne_thresh_reduce_gain": 0.25, "recent_activity_decay": 0.8, "refractory_ticks": 1, "sero_prune_gain": 0.04, "state_base_stim_mix": 0.15, "state_bias_stim_mix": 0.05, "state_parent_stim_mix": 0.25, "state_self_stim_mix": 0.55, "topk_branches": 4}`

### random:k_threshold_base=0.72;k_remem_base=0.95;k_decay=0.97;refractory_ticks=1;input_signal_clip=1.0;recent_activity_decay=0.5;hysteresis_threshold_gain=0.0;hysteresis_remem_gain=0.0;hysteresis_k_bonus=0.08;memory_decay=0.97;memory_k_mix=0.0;state_base_stim_mix=0.05

- balanced_score: `46.6071`
- pareto_front: `True`
- feasible: `False`
- constraint_penalty: `1.550000`
- constraint_failures: `hit_max_ticks_ratio>0.8;mean_first_active_tick>15.0;late_ignition_ratio_ge_15>0.5`
- mean_branch_len: `12.5000`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `1.0000`
- mean_first_active_tick: `19.5000`
- active_window_ratio: `0.3906`
- params_json: `{"branch_end_window": 6, "branch_length_bonus": 0.35, "dopa_rewire_gain": 0.8, "global_recovery_rate": 0.1, "hysteresis_k_bonus": 0.08, "hysteresis_remem_gain": 0.0, "hysteresis_threshold_gain": 0.0, "input_signal_clip": 1.0, "input_topk": 2, "k_decay": 0.97, "k_remem_base": 0.95, "k_threshold_base": 0.72, "max_out_degree": 12, "max_ticks": 32, "mela_dropout_gain": 0.04, "memory_decay": 0.97, "memory_k_mix": 0.0, "memory_stim_mix": 0.25, "min_out_degree": 1, "min_ticks_before_converged": 6, "ne_remem_reduce_gain": 0.25, "ne_thresh_reduce_gain": 0.25, "recent_activity_decay": 0.5, "refractory_ticks": 1, "sero_prune_gain": 0.04, "state_base_stim_mix": 0.05, "state_bias_stim_mix": 0.05, "state_parent_stim_mix": 0.25, "state_self_stim_mix": 0.55, "topk_branches": 4}`

## Figures

- `optimizer_balanced_score.svg`
- `optimizer_len1_vs_hitmax.svg`
- `optimizer_activation_tradeoff.svg`
- `optimizer_top_metrics.svg`
