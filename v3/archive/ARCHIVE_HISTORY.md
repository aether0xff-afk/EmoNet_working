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
