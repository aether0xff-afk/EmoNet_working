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
- `build_targeted_superiority_set.py`
  - build the 80-row episode-sensitive targeted set
- `generate_episode_v3_targeted.py`
  - generate targeted responses with `episode_trace_v3`
- `score_superiority_judge.py`
  - score targeted episode-fidelity dimensions
- `analyze_paired_superiority.py`
  - paired bootstrap, win/tie/loss, and sign-test analysis
- `prepare_human_eval.py`
  - prepare blinded human A/B CSVs and answer keys
- `analyze_human_eval_results.py`
  - unblind filled human A/B results and compute win/tie/loss statistics
- `generate_paper_refresh_structfix.py`
  - refresh current paper figures/tables

## Still useful but secondary

- `analyze_branch_dynamics.py`
- `optimize_branch_dynamics.py`
- `calibrate_reference_config.py`
- `analyze_branch_traces.py`
- `debug_judge_chat_response.py`

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
