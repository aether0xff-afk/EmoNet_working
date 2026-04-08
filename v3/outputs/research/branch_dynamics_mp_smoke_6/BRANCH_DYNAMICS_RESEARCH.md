# Branch Dynamics Research

## Configuration

- max_ticks: 40
- min_ticks_before_converged: 6
- k_threshold_base: 0.72
- k_decay: 0.99
- input_topk: 2
- input_signal_clip: 1.5

## Key Findings

- Current hard cap is `max_ticks=40` with `min_ticks_before_converged=6`.
- Full export improved mean dominant branch length to `18.96` and reduced `L1` ratio to `0.0154`, but the upper tail is still bounded (`p95=25.0`, `max=30`).
- `100.0%` of sampled runs terminated by hitting `max_ticks`, so the hard cap is already a material bottleneck.

## Version Summary

 rows      mean  median  len1_count  len1_ratio  p90  p95  max                                dataset                                                                                               source_csv
51628  1.053944     1.0       50257    0.973445  1.0  1.0  8.0              out_z_training_extended40              C:\Users\esl01\OneDrive\문서\GitHub\EmoNet_working\v3\outputs\z\out_z_training_extended40.csv
51628  2.730824     1.0       42509    0.823371  9.0 14.0 26.0    out_z_training_extended40_branchfix    C:\Users\esl01\OneDrive\문서\GitHub\EmoNet_working\v3\outputs\z\out_z_training_extended40_branchfix.csv
51628  6.156117     1.0       35595    0.689451 23.0 26.0 35.0 out_z_training_extended40_branchfix_v2 C:\Users\esl01\OneDrive\문서\GitHub\EmoNet_working\v3\outputs\z\out_z_training_extended40_branchfix_v2.csv
51628 18.964922    20.0         796    0.015418 24.0 25.0 30.0    out_z_training_extended40_structfix    C:\Users\esl01\OneDrive\문서\GitHub\EmoNet_working\v3\outputs\z\out_z_training_extended40_structfix.csv

## Sample Probe Summary

{
  "rows": 6,
  "mean_branch_len": 40.0,
  "p95_branch_len": 40.0,
  "mean_ticks_run": 40.0,
  "p95_ticks_run": 40.0,
  "hit_max_ticks_ratio": 1.0,
  "mean_path_coverage": 1.0,
  "mean_silent_tail_ticks": 0.0,
  "termination_counts": [
    {
      "termination_reason": "max_ticks",
      "count": 6
    }
  ]
}
