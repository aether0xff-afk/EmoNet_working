# Archive History

## 2026-04-06 legacy runs

Archive root:
`archive/2026-04-06_legacy_runs`

Purpose:
- keep the current `extended40` training/generation assets at the top level
- move superseded run outputs into a dated archive instead of deleting them
- keep paper/experiment deliverables in place

Active canonical files kept in place:
- `artifacts/ridge_stim_encoder.joblib`
- `artifacts/dominant_branch_encoder_extended40.pt`
- `artifacts/z_to_s_decoder_extended40.npz`
- `outputs/llm/llm_subset_4000_extended40.csv`
- `outputs/llm/llm_subset_4000_extended40_prompts.jsonl`
- `outputs/llm/llm_subset_labeled_4000_extended40.csv`
- `outputs/z/out_z_training_extended40.csv`
- `outputs/z/out_z_training_learned_extended40.csv`

Left untouched:
- `outputs/experiments/**`
- `outputs/paper/**`

Archived categories:
- legacy `z_to_s_decoder` artifacts
- legacy `llm` subsets and labeled CSVs
- legacy `z` exports
- old one-off response outputs
- old validation reports and logs

## 2026-04-06 paper history

Archive root:
`archive/2026-04-06_paper_history`

Purpose:
- keep earlier paper drafts without deleting them
- treat `PAPER_FRESH_DRAFT_ko.md` as the current working draft
- keep blueprint/worklist at the top level and move superseded drafts into dated history

Current top-level paper files:
- `PAPER_FRESH_DRAFT_ko.md`
- `PAPER_BLUEPRINT_ko.md`
- `PAPER_WORKLIST.md`

Archived paper files:
- `PAPER_DRAFT_ko.md`
- `PAPER_DRAFT_ko_v2.md`
- `PAPER_RESULTS_APPENDIX.md`

## 2026-04-09 paper unification

Archive root:
`archive/2026-04-09_paper_unification`

Purpose:
- unify the working draft into a single canonical file
- keep only the latest refresh bundle active at the top level
- move superseded paper drafts and legacy paper figures/tables into a dated archive instead of deleting them

Current active paper files kept in place:
- `PAPER_DRAFT_ko.md`
- `PAPER_BLUEPRINT_ko.md`
- `PAPER_WORKLIST.md`
- `outputs/paper/refresh_2026-04-09_calref_v1`

Archived top-level draft files:
- `PAPER_FRESH_DRAFT_ko.md`
- `PAPER_FRESH_DRAFT_ko_structfix_v1.md`
- `PAPER_DRAFT_AUDIT_2026-04-06.md`

Archived paper output sets:
- `outputs/paper/refresh_2026-04-07_structfix_v1`
- `outputs/paper/refresh_2026-04-07_structfix_v2`
- `outputs/paper/refresh_2026-04-08_structfix_v4`
- `outputs/paper/figures`
- `outputs/paper/requested_tables`
- `outputs/paper/paper_metrics_snapshot.json`
