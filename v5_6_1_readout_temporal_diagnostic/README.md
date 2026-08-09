# EmoNet v5.6.1 — Readout & Temporal-Control Diagnostic

## Version boundary

- v5.6 dual-timescale architecture test: frozen failed result
- **v5.6.1 changes neither the fast recurrent generator nor the slow EMA generator**
- diagnostic only

This version addresses two ambiguities exposed by v5.6:

1. slow-only semantic memory reached `0.800`, but naive high-dimensional concatenation fell to `0.675`;
2. the temporal-order benchmark was solved perfectly by slow EMA, so it could not demonstrate unique fast-state value.

## A. Semantic integration diagnostic

State generators are frozen:

```text
fast = v5.0 recurrent trace, 16 × 128 = 2048 raw features
slow = EMA embedding state, 384 features
```

Diagnostic readouts:

- fast only
- slow only
- raw concat
- balanced random projection concat at dimensions 8, 16, 32, and 64 per block
- independent fast/slow probe-score fusion

Random projections are deterministic and label-free. They are fit to no data and use no benchmark labels.

Probe-score fusion is explicitly a downstream diagnostic: separate fast and slow ridge probes are fit on the training split and their standardized decision scores are averaged. This is not proposed as an unsupervised core mechanism; it only tests whether the two blocks contain complementary recoverable information that raw concatenation obscures.

The existing v5.4 semantic fixture is reused because this version is diagnostic only.

## B. Identity-disjoint structural temporal benchmark

The old order benchmark used global ALPHA/BETA identities, and EMA solved it perfectly through recency weighting.

The new task uses pair-specific event identities that never repeat across pairs. Train and test identities are disjoint.

For each pair, two sequences use the same multiset `2A + 2B` and the same current input:

```text
class 0: A B A B
class 1: A A B B
```

The label describes sequence structure, not whether a globally fixed token A or B occurred recently.

Each pair receives unique synthetic event strings such as `sigil-kappa-17` / `sigil-orbit-17`; test-pair strings are unseen during training. The same frozen semantic encoder is used.

Compare:

- fast only
- slow only
- raw dual
- balanced projected dual
- fast reset
- slow reset

## Diagnostic questions

### Semantic

- Does balancing block dimensionality recover the slow semantic signal lost by raw concat?
- Do independent fast/slow scores show any complementarity beyond slow-only?

### Temporal

- Does slow EMA remain sufficient once token identities are pair-specific and unseen?
- Does fast recurrent state preserve a sequence-structure signal that generalizes across identities?

## Interpretation rule

This is not a success/failure version for the EmoNet hypothesis. It classifies the v5.6 failure source.

Possible outcomes:

- balanced concat ≈ slow-only → raw concat/readout dimensionality was the semantic integration problem;
- all dual readouts remain below slow-only → fast block adds mostly nuisance for semantic probing;
- fast > slow on identity-disjoint structure → evidence for unique fast temporal information;
- slow still matches fast → EMA-like memory already captures the tested temporal structure, so fast recurrence remains unjustified.
