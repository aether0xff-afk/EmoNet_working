# Result Artifact Policy

Status date: 2026-06-12

This policy defines what happens to generated experiment outputs from `v7`
runs. The goal is to keep the repository reproducible and reviewable without
turning git into storage for checkpoints, caches, and bulk logs.

## Default Rule

Generated run directories are local artifacts. Do not commit `runs/`,
`outputs/`, `artifacts/`, model checkpoints, embedding caches, raw run logs, or
bulk generated figures by default.

Commit the reviewed interpretation instead:

- the exact command or runner used
- code commit hash
- machine or backend details when relevant
- model identifiers for LM Studio or embedding backends
- seeds and key hyperparameters
- selected metrics needed for the decision
- the conservative claim boundary
- path or storage location for the full local or remote run output

## Commit to Git

Commit these when they are small, reviewed, and useful for future readers:

- docs that summarize an experiment or decision
- fixture files and configs required to reproduce a benchmark
- source code and tests
- small curated evidence tables under `docs/` when they are directly referenced
  by a decision document
- selected final figures under `docs/` only when a document discusses them and
  the image is needed for review

Curated evidence should be deterministic or manually reviewed. It should not
include secrets, private prompts, local machine paths that expose credentials,
raw LM Studio conversations beyond what the document needs, or unbounded
generated text.

## Keep Local or External

Keep these out of git:

- `best_checkpoint.pt`, `.pt`, `.pth`, `.npz`, `.joblib`
- `embedding_cache.json`
- `run_log.jsonl`
- full `history.csv` files from long runs
- full `runs.jsonl` response logs
- raw `metadata.json` when it mainly records local runtime state
- intermediate sweep directories
- generated PNGs from exploratory runs
- copied model bundles or large package archives

If a later experiment needs a checkpoint from an earlier run, record the source
run name, host, commit hash, and command in the consuming document. Do not copy
the checkpoint into the repository.

## Promote a Result

Use this promotion path when a local run becomes evidence for a Linear issue or
project decision:

1. Verify the run completed and, when applicable, that `used_device_fallback` is
   false.
2. Read the generated summary, comparison, or decision report.
3. Copy only the decision-relevant metrics into a documentation section.
4. Record the output directory, host/backend, command, commit hash, seeds, and
   date.
5. State what the result does not prove.
6. Leave the full generated directory in `runs/` or external storage.

For multi-seed benchmark decisions, prefer a compact summary table in docs over
committing per-seed `history.csv` or checkpoint files.

## File-Type Guidance

| Artifact | Default | Notes |
| --- | --- | --- |
| `summary.json` | Do not commit from `runs/` | Copy key fields into docs; commit a curated summary under `docs/` only if needed. |
| `decision_report.json` | Usually summarize in docs | Commit only if it is small, stable, and directly reviewed. |
| `comparison.json` | Usually summarize in docs | Useful fields should be copied into the relevant evidence section. |
| `history.csv` | Do not commit | Long and usually reproducible from command plus seed. |
| `run_log.jsonl` | Do not commit | Diagnostic log, not the durable evidence layer. |
| `embedding_cache.json` | Do not commit | Rebuildable cache and can be backend-specific. |
| `best_checkpoint.pt` | Do not commit | Large binary state; store externally if a later run depends on it. |
| `*.png` figures | Do not commit by default | Commit selected final figures under `docs/` only when needed for review. |
| `metadata.json` | Do not commit by default | Copy relevant environment fields into docs. |

## Current Local Convention

The repository currently ignores generated `v7/runs/`, `v7/outputs/`, and
`v7/artifacts/`. Existing run evidence for AET-25 and AET-26 is preserved as
documentation summaries rather than committed run directories.

This means future long runs should leave full outputs local or external, then
promote only reviewed summaries into the documentation and Linear comments.
