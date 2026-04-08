# Branch Depth Research (2026-04-08)

## What was checked

- Full-export branch length distributions across four stages:
  - `extended40`
  - `branchfix`
  - `branchfix_v2`
  - `structfix`
- Current-code sample probe on `200` randomly sampled texts from `out_z_training_extended40_structfix.csv`
- Tick-level activity profile, termination reason, path coverage, and delayed-ignition metrics

## Evidence files

- Version comparison figure: `v3/outputs/research/branch_dynamics_2026-04-08_v1/figures/branch_version_summary.svg`
- Tick-vs-branch figure: `v3/outputs/research/branch_dynamics_2026-04-08_v1/figures/ticks_vs_branch_length.svg`
- Tick activity figure: `v3/outputs/research/branch_dynamics_2026-04-08_v1/figures/tick_activity_profile.svg`
- Termination reason figure: `v3/outputs/research/branch_dynamics_2026-04-08_v1/figures/termination_reasons.svg`
- Path coverage figure: `v3/outputs/research/branch_dynamics_2026-04-08_v1/figures/path_coverage_histogram.svg`
- First-active-tick figure: `v3/outputs/research/branch_dynamics_2026-04-08_v1/figures/first_active_tick_histogram.svg`
- Active-window figure: `v3/outputs/research/branch_dynamics_2026-04-08_v1/figures/active_window_histogram.svg`
- Version summary table: `v3/outputs/research/branch_dynamics_2026-04-08_v1/tables/branch_version_summary.csv`
- Sample probe summary: `v3/outputs/research/branch_dynamics_2026-04-08_v1/tables/sample_probe_summary.json`
- Ignition summary: `v3/outputs/research/branch_dynamics_2026-04-08_v1/tables/ignition_summary.json`

## Main findings

### 1. `L1 collapse` is solved, but depth is still capped

Full-export summary:

- `extended40`: mean `1.05`, `L1 ratio 0.9734`, `p95 1`, `max 8`
- `branchfix`: mean `2.73`, `L1 ratio 0.8234`, `p95 14`, `max 26`
- `branchfix_v2`: mean `6.16`, `L1 ratio 0.6895`, `p95 26`, `max 35`
- `structfix`: mean `18.96`, `L1 ratio 0.0154`, `p95 25`, `max 30`

Interpretation:

- Collapse was fixed.
- But the upper tail is still narrow relative to a genuine long-horizon deliberation process.

### 2. The current bottleneck is not early death. It is `hard cap + late ignition`

Sample probe on `200` texts:

- mean branch length: `18.915`
- mean ticks run: `40.0`
- `hit_max_ticks_ratio`: `1.0`
- `p95 ticks run`: `40.0`
- termination reason: `200 / 200 = max_ticks`

Interpretation:

- Current runs are not ending because the convergence rule says "enough".
- They are hitting the hard ceiling every time.
- Therefore the system is no longer branch-starved; it is horizon-capped.

### 3. Dominant-path extraction is not the main bottleneck anymore

Sample probe:

- mean path coverage: `0.9894`
- median path coverage: `1.0`

Interpretation:

- Once activity starts, the selected dominant path almost fully covers the active window.
- So the current problem is not "the extractor throws away most of the thought".
- The deeper issue is "activity starts too late, then runs until the cap".

### 4. The real structural problem is delayed ignition

Ignition summary:

- mean first active tick: `20.89`
- median first active tick: `20`
- `p90 first active tick`: `26`
- `late_ignition_ratio_ge_15`: `0.935`
- mean active window: `18.92`
- no-activity rows: `2 / 200`

Interpretation:

- In most samples, nothing meaningful happens for roughly the first half of the run.
- Then the graph activates late and stays active almost until tick 39.
- That is why branch length clusters around `~19-20` even though every run uses all `40` ticks.

## Code-level explanation

### Hard cap

- `max_ticks=40`: `v3/emonet/core.py`
- `run_until_converged(...)` stops at `max_ticks`: `v3/emonet/core.py`

### Why ignition is late

At reset, every neuron starts with:

- `K = 0`
- no pending signals
- only base stimulus, self-state, parent-state, and intrinsic bias mixed into `stim_vec`

Then in each tick, before any real branch propagation exists, `K` grows mainly through:

- `input_strengths[node]` from pending signals
- `hysteresis_k_bonus * recent_activity`
- `+ 0.3*dopamine + 0.3*norepinephrine`
- `- 0.3*serotonin - 0.3*melatonin`

This means the initial ignition process is effectively a slow accumulation process against a fixed threshold:

- `k_threshold_base = 0.72`
- `k_decay = 0.99`
- `pending_signals = {}` at the beginning

So the model spends many ticks integrating weak base drive before the first real activation wave appears.

## What this means for the next design

The next step should **not** be "more parameter tuning".

The right order is:

1. Add an explicit ignition mechanism
   - Example: a short-lived external seed pulse or text-conditioned starter set
   - Goal: bring first active tick from `~21` down to the first few ticks

2. Then increase the horizon
   - Raise `max_ticks` only after ignition is fixed
   - Otherwise we just add more empty ticks before activation

3. Then re-evaluate convergence
   - Right now convergence is not the limiting factor because nothing converges before the cap
   - After ignition is improved, convergence criteria become relevant again

## Concrete research claim supported by the data

The present EmoNet no longer fails because dominant branches collapse to length `1`.
It now fails to achieve genuinely long deliberation because the current dynamics ignite too late and then saturate against a fixed `40`-tick ceiling.
