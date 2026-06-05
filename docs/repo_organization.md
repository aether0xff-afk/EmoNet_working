# Repo Organization and Heuristic Algorithm Map

This repository is a research workspace with multiple preserved implementation
generations. The safest organization model is to keep version directories in
place and use documentation as the routing layer, because many scripts assume
version-local imports, outputs, and relative paths.

## Active Directory Roles

- `src/`: Minecraft RL Agent MVP in JavaScript. This is the active root
  Node.js project.
- `docs/`: Cross-version architecture and repo-level documentation.
- `v6/`: Current Ruca/Rookie autonomous character runtime.
- `v5/`: Character chat MVP that v6 extends.
- `v4/`: Research, evaluation, and GUI generation before the character runtime
  split.
- `v3.1/`: Trace-as-emotion experiment line.
- `v3/`: Earlier standalone research CLI and branch dynamics experiments.
- `v2/`: Early modular PyTorch MVP.
- `v1/`: Initial emotion-z pipeline and GUI experiments.
- `blueprints/`: Design notes and older architecture drafts.
- `Dataset/`: Shared raw dataset.
- `encoder-LLM-testing/`: LLM labeling benchmarks.
- `encoder-ML testing/`: ML encoder benchmark materials.
- `output/`, `outputs/`: Generated figures and experiment results.
- `tmp/`: Temporary document, poster, and conversion artifacts.

## Heuristic Algorithm Locations

The exact word `heuristic` is not used in the active code, but several modules
implement heuristic or rule-based decision logic under names such as policy,
reward, score, gate, and optimize.

### Minecraft RL MVP

- `src/policy.js`
  - `ABCPolicy` maintains probability tables for WHAT/HOW/WHERE action
    choices.
  - `epsilon`, `learningRate`, probability normalization, sampling, and reward
    updates make this the root policy heuristic.
- `src/imagination.js`
  - `ImaginationCycle.choose()` scores candidate actions using short rollouts.
  - Score components include predicted reward, novelty bonus, uncertainty
    bonus, error penalty, discounting, and candidate ranking.
- `src/reward.js`
  - `RewardModule.compute()` defines reward shaping.
  - Components include repeat reward decay, error penalty, knowledge delta
    reward, goal reward, and prediction accuracy reward.
- `src/actions.js`
  - Action execution contains practical selection rules such as nearest known
    tree or stone block, random exploration yaw, and crafting fallback steps.

### Branch Dynamics Search

- `v6/scripts/optimize_branch_dynamics.py`
  - Main hyperparameter optimizer for branch dynamics.
  - Supports preset search spaces, grid search, random search, baseline
    inclusion, resume, score figures, Pareto tagging, and best config export.
  - `summarize_candidate()` computes the balanced score from branch length,
    max-tick behavior, late ignition, activation targets, and constraints.
  - `mark_pareto_front()` identifies non-dominated candidates across multiple
    objectives.
- Same script lineage also exists in `v5/scripts/` and `v4/scripts/`.

### Ruca/Rookie Runtime Gates

- `v6/ruca_engine/event_scheduler.py`
  - Converts user text, no-reply time, and silence into normalized runtime
    events.
  - Uses thresholds such as `elapsed >= 45.0` for long-silence handling.
- `v6/ruca_engine/spontaneous.py`
  - Rule-based spontaneous reaction decision logic.
  - Uses thresholds for elapsed silence, alarm, warmth, protective tension,
    action pressure, and repeated alarm memories.
- `v6/ruca_engine/response_gate.py`
  - Final response action gate.
  - Chooses among `send_message`, `update_internal_only`, and `stay_silent`
    based on event type, spontaneous decision, elapsed time, and arousal.

## Suggested Organization Rules

- Keep version roots (`v1` through `v6`) stable unless doing a deliberate
  migration with import-path fixes and tests.
- Put new repo-level explanations in `docs/`.
- Put new v6 runtime code under `v6/ruca_engine/`.
- Put v6 experiment scripts under `v6/scripts/`.
- Put generated v6 runtime state under `v6/outputs/`.
- Put throwaway conversion/build material under `tmp/` and do not import from
  it.
- Avoid adding new large archives at the repo root. Prefer a dated archive
  directory or external storage for generated bundles.

## Cleanup Candidates

These are candidates for a separate cleanup branch, not automatic moves:

- Root zip files such as `emonet_core_results.zip`,
  `emonet_paper_figures.zip`, `emonet_paper_figures_ko.zip`, and
  `emonet_python_files.zip`.
- Root document artifacts such as `.docx` research logs.
- Duplicate rendered document folders: `docx_render_v4/` and
  `docx_render_v4_final/`.
- Historical generated outputs under old version directories if they are no
  longer used by papers or reports.

Before moving any of these, check whether they are tracked, referenced by
reports, or needed to reproduce existing paper figures.
