# Neural Trace-As-Emotion Design

## 1. Corrected Definition

In v3.1, `trace` means the neural activation trajectory produced when a stimulus vector flows through the EmoNet network.

It does not primarily mean:

```text
episode_label
target
control_state
action_tendency
```

Those are external interpretation labels.

The actual object of study is:

```text
stimulus vector
-> network propagation
-> tick-by-tick neuron activations
-> dominant branch / branch tensor / z
-> emotion-state trace
```

## 2. Core Claim

The strong EmoNet claim is:

> emotion is the activation trace formed as a stimulus travels through the neural network.

The symbolic appraisal fields are useful only as probes for that trace.

## 3. Evidence Plan

Representation evidence:

- same appraisal/action labels should have similar neural traces
- different labels should separate in neural trace space
- dominant branches should show reusable neuron routes
- z embeddings should preserve emotion-relevant geometry

Causal evidence:

- perturbing neural trace features should shift emotion interpretation
- ablating high-contribution neurons should reduce label separability
- increasing neuron count should improve trace geometry only if the added capacity forms stable clusters

## 4. Current Implementation

`scripts/export_neural_activation_traces.py` exports:

- `activation`: tick x neuron K matrix
- `branch_tensor`: dominant branch feature tensor
- `z`: encoded trace embedding
- `stim_vec`: original 4D stimulus vector
- `dominant_branch_ids`: route through the network
- `active_counts`: active neuron count per tick

The output is stored under:

```text
outputs/neural_trace_probe_v1/
```

## 5. Relationship To Previous v3.1 Work

The previous normalized fields are not discarded. They become evaluation labels:

```text
neural trace = object being tested
target/control/social/action labels = external probes
```

This is the correct hierarchy for the user's original idea.

