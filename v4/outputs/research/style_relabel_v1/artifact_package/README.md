# Style Relabel v1 Artifact Package

This package summarizes the current style-bias mitigation proof of concept.

## Key Files

- `summary_metrics.csv`: before/after bias metrics.
- `focus_axes_before_after.csv`: axis-level means for key style and raw-affect axes.
- `decoder_group_mae.csv`: decoder MAE by axis group.
- `style_relabel_before_after.svg`: compact figure for slides or paper draft.
- `paper_table_style_relabel_v1.tex`: LaTeX table.
- `STYLE_RELABEL_V1_REPORT.md`: short narrative report.

## Source Run

- Candidate rows: `120`
- Applied relabel rows: `120`
- Candidate buckets: `{"anger_resentment": 25, "despair_helplessness": 25, "fear_panic": 25, "shame_guilt": 25, "relationship_conflict": 20}`

## Main Result

Relabeled subset negative raw mean changed from `0.001042` to `0.324653`.
Relabeled subset soft bias mean changed from `0.847222` to `0.414931`.
