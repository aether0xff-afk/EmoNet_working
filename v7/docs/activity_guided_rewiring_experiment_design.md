# Activity-Guided Rewiring Experiment Design

Status date: 2026-06-12

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

## 2026-06-12 AET-29 Pipeline Result

The activity-guided rewiring pipeline was run on remote host `DESKTOP-MMLRCFK`
with LM Studio embeddings and CUDA execution for the train/evaluation stages.

```text
Code commit: 1ca029227e6471bdbec88a800e1b5b09dbc7e657
Python env: C:/Users/remote/miniconda3/envs/picasso-gpu/python.exe
GPU: NVIDIA GeForce RTX 4090
Encoder: lmstudio
Embedding model: text-embedding-nomic-embed-text-v1.5
Base URL: https://desktop-mmlrcfk.tail93ffc6.ts.net
Epochs: 30
Seeds: 7, 13, 21, 42, 100
Null permutations: 64
Output: runs/activity_guided_rewiring_pipeline_lmstudio
```

Pipeline command:

```powershell
python experiments/run_activity_guided_rewiring_pipeline.py `
  --encoder lmstudio `
  --base-url https://desktop-mmlrcfk.tail93ffc6.ts.net `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --seeds 7 13 21 42 100 `
  --null-permutations 64 `
  --device cuda `
  --output runs/activity_guided_rewiring_pipeline_lmstudio `
  --skip-baseline-auto-create
```

The pipeline found a semantic-preserving rewiring region:

```text
stability_sweep stage_verdict: semantic_preserving_rewiring_region_found
selected config: fraction_0.0100__start_10__interval_10
rewiring_fraction: 0.01
rewiring_start_epoch: 10
rewiring_interval: 10
baseline memory model real targeted MAE: 0.26881304606795314
rewired real targeted MAE: 0.26550635248422627
baseline_minus_rewired_mae: 0.003306693583726872
real_direction_accuracy: 0.675
real_pair_order_accuracy: 0.825
shuffled_minus_real_mae: 0.06898729503154755
reset_minus_real_mae: 0.1703202120959758
rewiring_event_count: 2.4
rewired_edges_total: 38.8
objective_memory_strength_mean_abs: 0.3491812162101269
semantic_preservation_checks_pass: true
```

Stable regions identified by the sweep:

```text
rewiring_fraction: 0.005, 0.02
rewiring_interval: 10
rewiring_start_epoch: 15
```

The rewired cluster diagnostic did **not** establish community evidence:

```text
pipeline stage_verdict: rewiring_community_evidence_not_established
rewired_cluster stage_verdict: rewiring_community_evidence_not_established
selected_cluster_count: 6.4
trained_modularity: 0.1990827530622482
initial_modularity: 0.19982807338237757
null_modularity: 0.20186002082191407
trained_minus_initial_modularity: -0.0007453203201293854
trained_minus_null_modularity: -0.0027772677596658198
response_coherence_gap: -0.0092938501703614
trained_minus_null_response_coherence_gap: -0.0096060820338958
trained_minus_null_modularity positive rate: 0.2
trained_minus_null_response_coherence_gap positive rate: 0.2
```

Generated figure manifest:

```text
runs/activity_guided_rewiring_pipeline_lmstudio/rewired_cluster/figures/visualization_manifest.json
```

The manifest lists five seed-level community-assignment figures plus
`community_sizes_by_seed.png` and `rewiring_cluster_metrics.png`.

Interpretation:

The current activity-guided rule found a narrow topology-change regime that
preserved, and slightly improved, the controlled semantic-readability metric
relative to the selected memory-threshold baseline. However, the resulting
weighted adjacency did not beat shuffled-weight nulls and did not produce
within-community memory-response coherence. The rule should therefore remain a
controlled ablation and search heuristic, not a final rewiring rule or evidence
for stable emotion clusters.
