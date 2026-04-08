# Branch Dynamics Research

## Configuration

- max_ticks: 128
- min_ticks_before_converged: 6
- k_threshold_base: 0.72
- k_decay: 0.93
- input_topk: 2
- input_signal_clip: 0.8

## Key Findings

- Current hard cap is `max_ticks=128` with `min_ticks_before_converged=6`.
- Full export improved mean dominant branch length to `18.96` and reduced `L1` ratio to `0.0154`, but the upper tail is still bounded (`p95=25.0`, `max=30`).
- `30.0%` of sampled runs terminated by hitting `max_ticks`, so the hard cap is already a material bottleneck.
- Mean ticks run is `78.91`, well below the cap, which indicates the current `delta_k` convergence test is still aggressive.
- Silent tail after the last active tick averages `1.44` ticks, which means the model often keeps stepping after meaningful branch activity has already decayed.

## Version Summary

 rows      mean  median  len1_count  len1_ratio  p90  p95  max                                dataset                                                                                              source_csv
51628  1.053944     1.0       50257    0.973445  1.0  1.0  8.0              out_z_training_extended40              C:\Users\remote\Documents\GitHub\EmoNet_working\v3\outputs\z\out_z_training_extended40.csv
51628  2.730824     1.0       42509    0.823371  9.0 14.0 26.0    out_z_training_extended40_branchfix    C:\Users\remote\Documents\GitHub\EmoNet_working\v3\outputs\z\out_z_training_extended40_branchfix.csv
51628  6.156117     1.0       35595    0.689451 23.0 26.0 35.0 out_z_training_extended40_branchfix_v2 C:\Users\remote\Documents\GitHub\EmoNet_working\v3\outputs\z\out_z_training_extended40_branchfix_v2.csv
51628 18.964922    20.0         796    0.015418 24.0 25.0 30.0    out_z_training_extended40_structfix    C:\Users\remote\Documents\GitHub\EmoNet_working\v3\outputs\z\out_z_training_extended40_structfix.csv

## Sample Probe Summary

{
  "rows": 200,
  "mean_branch_len": 74.45,
  "p95_branch_len": 125.0,
  "mean_ticks_run": 78.91,
  "p95_ticks_run": 128.0,
  "hit_max_ticks_ratio": 0.3,
  "mean_path_coverage": 0.8380837265363826,
  "mean_silent_tail_ticks": 1.44,
  "termination_counts": [
    {
      "termination_reason": "delta_k",
      "count": 140
    },
    {
      "termination_reason": "max_ticks",
      "count": 60
    }
  ]
}
