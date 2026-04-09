# v4 Scripts

## Active now

- `inspect_emotion_trace.py`
  - single-sample raw node/tick trace extraction
- `analyze_emotion_trajectory_batch.py`
  - multi-sample trajectory phase analysis
- `interpret_emotion_trajectory.py`
  - local GPT-5.4 trajectory-to-episode interpretation
- `experiment_matrix.py`
  - response generation matrix including `episode_trace` and `hybrid_episode`
- `score_experiment_matrix.py`
  - judge scoring
- `generate_paper_refresh_structfix.py`
  - refresh current paper figures/tables

## Still useful but secondary

- `analyze_branch_dynamics.py`
- `optimize_branch_dynamics.py`
- `calibrate_reference_config.py`
- `analyze_branch_traces.py`
- `debug_judge_chat_response.py`
- `prepare_human_eval.py`

## Historical helpers

Moved to `../archive/scripts/`:

- `paper_experiments.ps1`
- `paper_remote_all.ps1`
- `paper_remote_runs.ps1`
- `paper_requested_tables.ps1`
- `branch_sweep_phase2_grid.json`

## Rule of thumb

- RDP-heavy: branch export, batch trajectory extraction, large generation matrices
- Local-only: GPT-5.4 episode interpretation, GPT-5.4 judging, draft updates
