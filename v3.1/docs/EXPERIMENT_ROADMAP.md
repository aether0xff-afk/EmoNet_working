# v3.1 Experiment Roadmap

## Phase 1: Static Trace Space Probe

Goal: check whether existing trace fields already form emotion-like structure.

Tasks:

- build mixed numeric/categorical trace vectors
- compute nearest-neighbor consistency for each emotional axis
- compute intra-group vs inter-group distance
- report weak and strong axes

Success signal:

- nearest-neighbor consistency is clearly above majority-label baseline for at least `target`, `social_orientation`, and `action_tendency`

## Phase 2: Cluster Discovery

Goal: test whether clusters appear without using emotion labels directly.

Tasks:

- run k-means or agglomerative clustering on trace vectors
- compare clusters against `target`, `control_state`, `social_orientation`, `action_tendency`
- inspect cluster exemplars

Success signal:

- clusters map to appraisal/action patterns rather than only surface topic

## Phase 3: Trace Ablation

Goal: prove that trace is causally useful, not just correlated.

Tasks:

- generate responses with full trace
- remove one trace axis at a time
- score appraisal fidelity, raw affect preservation, anti-softening, and action tendency fit

Success signal:

- removing `target`, `avoid`, or `action_tendency` causes predictable drops in the matching judge metric

## Phase 4: Trace Perturbation

Goal: test whether changing trace changes the emotional interpretation.

Tasks:

- keep the same stimulus text
- alter `target`, `control_state`, `avoid`, or `action_tendency`
- generate responses from perturbed traces
- ask judge or humans whether the intended emotion direction changed

Success signal:

- perturbations produce controlled changes without destroying naturalness

## Phase 5: Neural Trace Expansion

Goal: test whether more neurons or learned trace dynamics improve emotion-state geometry.

Tasks:

- compare neuron counts such as 256, 512, 1024
- evaluate branch collapse, trace separability, and response fidelity
- measure whether learned clusters become more stable

Success signal:

- larger models improve trace separability and targeted fidelity without reintroducing branch collapse

## Reporting Rule

v3.1 should report representation evidence separately from generation evidence:

```text
representation proof: trace behaves like emotion space
generation proof: trace improves responses
```

This prevents the claim from becoming too broad too early.

