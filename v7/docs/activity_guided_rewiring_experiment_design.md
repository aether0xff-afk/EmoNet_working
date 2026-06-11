# Activity-Guided Rewiring Experiment Design

Status date: 2026-06-11

This document defines the current M3 experiment path for self-organizing
community and rewiring checks. The implementation is an ablation, not a final
rewiring rule.

## Question

Can neuron-local memory profiles discover useful communities and rewire sparse
adjacency while preserving context/semantic readability under controlled
fixtures?

## Current Rule

The current rule in `src/emonet_v7/activity_guided_rewiring.py`:

1. collects train-episode memory-strength profiles,
2. computes positive neuron-neuron coactivity,
3. discovers functional communities by spectral features and modularity search,
4. removes weak inter-community edges,
5. adds high-coactivity intra-community edges,
6. preserves the directed edge budget,
7. resets optimizer moments only for changed recurrent-weight entries.

Semantic labels are not used for training, rewiring, or community discovery.

## Experiment Stack

Minimum comparison path:

```text
memory-threshold baseline sweep
-> activity-guided rewiring stability OFAT sweep
-> semantic-preserving config selection
-> rewired adjacency-community diagnostic
-> visualization of community assignment and diagnostic metrics
```

Entry points:

- `experiments/run_memory_threshold_parameter_sweep.py`
- `experiments/summarize_memory_threshold_parameter_sweep.py`
- `experiments/run_activity_guided_rewiring_stability_sweep.py`
- `experiments/summarize_activity_guided_rewiring_stability_sweep.py`
- `experiments/run_activity_guided_rewiring_emergent_cluster_benchmark.py`
- `experiments/summarize_activity_guided_rewiring_emergent_cluster_benchmark.py`
- `experiments/run_activity_guided_rewiring_pipeline.py`
- `experiments/visualize_activity_guided_rewiring_clusters.py`

## One-Command Pipeline

```powershell
python experiments/run_activity_guided_rewiring_pipeline.py `
  --encoder lmstudio `
  --base-url http://127.0.0.1:1234 `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --seeds 7 13 21 42 100 `
  --output runs/activity_guided_rewiring_pipeline_lmstudio
```

Then render figures:

```powershell
python experiments/visualize_activity_guided_rewiring_clusters.py `
  --input runs/activity_guided_rewiring_pipeline_lmstudio/rewired_cluster
```

## Output Files

Pipeline-level:

```text
pipeline_report.json
stability_sweep/decision_report.json
rewired_cluster/decision_report.json
rewired_cluster/summary_metrics.csv
rewired_cluster/by_seed_cluster.csv
rewired_cluster/seed_*/cluster_diagnostic.json
rewired_cluster/seed_*/neuron_communities.csv
rewired_cluster/figures/*.png
rewired_cluster/figures/visualization_manifest.json
```

The visualizer writes:

```text
seed_<seed>_community_assignment.png
community_sizes_by_seed.png
rewiring_cluster_metrics.png
visualization_manifest.json
```

## Decision Checks

The stability sweep selects a rewiring region only when:

- rewiring actually changes edges,
- real-history targeted MAE does not materially regress from the selected memory
  baseline,
- shuffled and reset histories degrade relative to real history,
- direction and pair-order accuracy exceed chance,
- positive-rate checks are stable across seeds,
- memory strength is not saturated.

The cluster diagnostic separates:

- structural evidence: modularity versus initialization and shuffled-weight
  nulls,
- functional evidence: within-community memory-response coherence versus
  between-community and label-permutation nulls,
- descriptive semantic-axis correlation, used only after community discovery.

## Interpretation Boundary

Passing this stage can support a narrow claim:

```text
Under the controlled fixture, the current activity-guided rewiring rule found a
semantic-preserving topology change and produced adjacency communities with
diagnostic structure.
```

It does not establish emotional ground truth, stable neuron roles, biological
fidelity, or broad real-world generalization.
