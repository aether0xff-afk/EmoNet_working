# v3.1 TRACE Formation Experiment Design

## Goal

Confirm whether neural activation traces in `v3.1` form an emotion-relevant
TRACE representation.

The experiment follows this flow:

```text
emotion sentence -> neural network -> output
                         |
                         v
              neural activation record
                         |
                         v
      extract synapse/neuron firing and activation strength
                         |
                         v
              confirm TRACE formation
```

TRACE formation is defined representation-level, not by response quality:
sentences with the same emotion/appraisal axes should produce closer neural
activation traces than sentences with different axes.

## Scope

Use the `v3.1` research line. The experiment should extend the existing neural
trace tools instead of creating a separate runtime.

In scope:

- Input CSV of emotion sentences and appraisal labels.
- Neural activation export using the existing v3 EmoNet network.
- Tick-level activation records, dominant branch route, branch tensor, active
  count series, and `z` vector extraction.
- Geometry probing across multiple neural feature views.
- A compact JSON or Markdown result summary that states whether TRACE formation
  was observed.

Out of scope:

- Ruca/Rookie `v6` runtime behavior.
- LLM response generation quality.
- Human evaluation.
- Training a new neural model.

## Inputs

The primary input is a CSV with one row per emotion sentence.

Required columns:

- `record_id`
- `text`
- `valence`
- `arousal`
- `target`
- `control_state`
- `social_orientation`
- `action_tendency_class`

Optional columns:

- `episode_family`
- `appraisal_family`

The existing `v3.1/scripts/export_neural_activation_traces.py` already supports
these fields and can keep unknown or optional columns as labels for probing.

## Pipeline

### 1. Export Neural Activation Records

Run each emotion sentence through the v3 EmoNet network. For every record, save:

- `activation`: tick-by-neuron activation matrix.
- `branch_tensor`: branch-level representation.
- `z`: encoded stimulus vector.
- `stim_vec`: neurotransmitter-like input vector.
- `dominant_branch_ids`: dominant node route.
- `active_counts`: number of active nodes per tick.

The existing exporter already writes compressed `.npz` files plus
`neural_trace_summary.csv`. The implementation should keep this output format
so existing probes remain compatible.

### 2. Extract TRACE Features

Use multiple feature views because TRACE may appear in different parts of the
network record:

- `activation_temporal`: temporal mean/max pools over tick-level activation.
- `branch_plus_temporal`: branch tensor, temporal pooling, active stats, route
  histogram, and transition hash.
- `route_histogram`: dominant branch node distribution.
- `active_stats`: summary of firing density and activation dynamics.
- `z`: encoded stimulus baseline.

The primary feature view is `branch_plus_temporal` because it combines route,
intensity, and temporal dynamics.

### 3. Confirm TRACE Formation

For each feature view, compute:

- nearest-neighbor label consistency.
- majority-label baseline.
- balanced nearest-neighbor consistency.
- balanced random baseline.
- intra-label distance.
- inter-label distance.

Evaluate these axes:

- `target`
- `control_state`
- `social_orientation`
- `action_tendency_class`
- `valence`
- `arousal`
- `episode_family`
- `appraisal_family`

## Success Criteria

TRACE formation is considered observed when the primary feature view
`branch_plus_temporal` satisfies both conditions:

1. At least two of `target`, `control_state`, `social_orientation`, and
   `action_tendency_class` have positive balanced nearest-neighbor lift.
2. At least two of those same axes have positive distance separation, meaning
   mean inter-label distance is greater than mean intra-label distance.

Secondary support is stronger if `activation_temporal` or `route_histogram`
shows the same pattern.

The result should be reported as:

- `confirmed`: success criteria met.
- `partial`: one condition met, or only secondary feature views pass.
- `not_confirmed`: neither condition met.

## Outputs

Expected output directory:

```text
v3.1/outputs/trace_formation_v1/
```

Expected files:

- `neural_trace_summary.csv`
- `traces_npz/*.npz`
- `trace_formation_report.json`
- optional `trace_formation_report.md`

The report should include:

- input path and record count.
- model settings such as neuron count and seed.
- branch health summary.
- per-feature TRACE verdict.
- strongest axes.
- weakest axes.
- final verdict.

## Error Handling

The exporter should continue when an individual row fails, collect row-level
errors, and write them to the manifest. The reporter should fail clearly if no
trace files are available or if required label columns are missing.

If a label axis has too few examples for balanced evaluation, the report should
mark it as insufficient rather than treating it as failed evidence.

## Testing

Add or update tests around the pure analysis layer where possible:

- verdict classification from synthetic metric dictionaries.
- required-column validation.
- report aggregation across feature views.

Full neural export can remain a smoke/integration command because it is heavier
and depends on the v3 model.

## Implementation Notes

Prefer extending `v3.1/scripts/probe_neural_trace_geometry.py` or adding a small
wrapper script that calls existing export and probe logic. Avoid duplicating
feature extraction code unless the existing script becomes hard to reuse.

The implementation should preserve existing v3.1 outputs and default behavior.
