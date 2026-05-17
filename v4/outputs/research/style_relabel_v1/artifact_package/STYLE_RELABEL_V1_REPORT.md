# Style Relabel v1 Report

## Purpose

Current `extended40` style supervision is heavily biased toward soft, calm, cooperative response surfaces. This run mined hard cases whose Korean dialogue contains anger, resentment, despair, fear, shame, or conflict cues while the existing raw negative style axes remain near zero.

## Inputs

- Base CSV: `outputs/z/out_z_training_learned_extended40_calref_v1.csv`
- Candidate CSV: `outputs/research/style_relabel_v1/style_relabel_candidates.csv`
- Claude relabel CSV: `outputs/research/style_relabel_v1/style_relabel_claude.csv`
- Applied training CSV: `outputs/research/style_relabel_v1/out_z_training_learned_extended40_calref_v1_style_relabel_v1.csv`
- Applied rows: `120`

## Candidate Set

- anger_resentment: `25`
- despair_helplessness: `25`
- fear_panic: `25`
- shame_guilt: `25`
- relationship_conflict: `20`

Before relabeling, this subset had:

- mean current negative raw max: `0.00625`
- mean current soft bias mean: `0.847222`

## Relabel Effect

On the 120 relabeled rows:

- negative raw mean: `0.001042 -> 0.324653`
- soft bias mean: `0.847222 -> 0.414931`

On all kept rows:

- negative raw mean: `0.003155 -> 0.025772`
- soft bias mean: `0.814818 -> 0.784605`
- edge mean: `0.103873 -> 0.124054`

## Decoder Check

Same seed, 40 style axes, `keep_sample` rows.

| target set | all MAE | soft MAE | negative raw MAE | edge MAE |
|---|---:|---:|---:|---:|
| original | 0.120546 | 0.118468 | 0.007217 | 0.132776 |
| style_relabel_v1 | 0.136223 | 0.150443 | 0.058784 | 0.155046 |

The relabeled target is harder, so all-axis MAE increases. This is expected: the original negative raw axes were nearly constant zeros, so low MAE there did not indicate real affect learning. After relabeling, negative raw axes become a non-trivial prediction target.

## Interpretation

This supports the current diagnosis: style bias is primarily a target-construction problem, not only a final prompt problem. Rebalancing the existing keep set did not fix the issue because the raw negative labels were already collapsed. Relabeling hard cases with a `felt_state` / `response_style` split restores raw affect signal while reducing over-softened targets.
