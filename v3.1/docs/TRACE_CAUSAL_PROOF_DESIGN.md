# Trace Causal Proof Design

## 1. Purpose

Representation-level evidence showed that normalized trace axes form emotion-like structure. The next question is causal:

> if a trace axis is removed or changed, does the emotional direction of the response change predictably?

This proof is stronger than clustering. Clustering says similar traces are near each other. Causal manipulation asks whether trace fields actually control appraisal and response direction.

## 2. Conditions

Each base record is expanded into nine causal conditions:

| Condition | Type | Meaning |
|---|---|---|
| `trace_full` | control | all trace fields preserved |
| `ablate_target` | ablation | remove emotion target |
| `ablate_social_orientation` | ablation | remove social orientation |
| `ablate_control_state` | ablation | remove control/agency state |
| `ablate_action_tendency_class` | ablation | remove canonical action tendency |
| `perturb_target` | perturbation | change target to a contrasting value |
| `perturb_social_orientation` | perturbation | change social orientation |
| `perturb_control_state` | perturbation | change control state |
| `perturb_action_tendency_class` | perturbation | change action tendency class |

## 3. Expected Effects

| Manipulated axis | Expected failure or shift |
|---|---|
| `target` | blame/self/other direction should weaken or flip |
| `social_orientation` | defend/approach/withdraw tone should shift |
| `control_state` | helplessness, agency, and planning tone should shift |
| `action_tendency_class` | suggested action direction should change |

## 4. Evidence Criteria

A causal effect is supported if:

1. Full trace scores higher than ablated trace on the matching metric.
2. Perturbed trace shifts response direction toward the new manipulated value.
3. Naturalness does not fully explain the result.
4. The effect appears within the same `record_id`, not only across different examples.

## 5. Current Probe Set

The first causal probe set uses 24 base records and creates 216 rows:

```text
24 trace_full
96 ablation rows
96 perturbation rows
```

Each manipulated axis has 48 rows:

```text
24 ablations + 24 perturbations
```

## 6. Next Execution Step

The next script should generate responses for every causal row using the same generation backend as v4 `episode_trace_v3`, but with the manipulated trace payload.

Then a causal judge should score:

- appraisal fidelity
- target direction fit
- social orientation fit
- control state fit
- action tendency fit
- raw affect preservation
- naturalness

The analysis should be paired within `record_id`:

```text
trace_full - ablated_axis
perturbed_axis direction match
```

