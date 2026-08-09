# v5.9.1 pre-run protocol amendment

Status: **committed before the first GitHub Actions benchmark run and before any v5.9.1 result was observed.**

The preregistered primary question, state generators, vector worlds, tasks, seeds, train/test splits, primary event-trace self-similarity readout, causal controls, and all acceptance gates are unchanged.

One **non-primary** diagnostic is amended for computational cost:

Original wording:

> Full raw episode trajectory — concatenate all 7 event traces and fit the same ridge probe directly.

The concatenated trajectory has:

```text
7 events × 16 ticks × 128 neurons = 14,336 coordinates
```

Repeated high-dimensional ridge Gram-matrix construction across three leave-one-world-out folds, five recurrent seeds, four tasks, and two frozen dynamics would dominate the run without affecting the primary trajectory-similarity test.

Amended diagnostic:

> Concatenate the same 14,336 raw coordinates, then apply one fixed deterministic 256-dimensional signed feature hash before the downstream ridge probe.

Properties:

- hash seed fixed at `5_091_2026`;
- bucket/sign mapping depends only on raw coordinate index and the fixed seed;
- no class label, world ID, task ID, pair ID, train/test split, or result affects the projection;
- every raw episode coordinate contributes exactly once;
- this diagnostic is not an acceptance gate and cannot substitute for event-trace self-similarity.

Interpretation wording is correspondingly narrowed from **full raw episode** to **hashed full-episode diagnostic**.

No scientific result existed when this amendment was committed.
