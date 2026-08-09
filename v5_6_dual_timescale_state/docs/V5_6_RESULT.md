# EmoNet v5.6 Result — Dual-Timescale State

검증일: 2026-08-09 KST

## Version boundary

- v5.5 predictive-state memory: frozen failed result
- v5.6 dual-timescale architecture test: this result
- branch: `feature/v5.6-dual-timescale-state`
- PR: #14
- workflow run: `31291499434`
- job: `93189374989`
- artifact: `v5.6-dual-timescale-state-benchmark`
- artifact id: `9031557125`

v5.6 introduced no new learning objective. The state was split into:

```text
fast = frozen v5.0 fixed recurrent trace
slow = EMA semantic embedding memory, decay 0.80
dual = [fast, slow]
```

Fast and slow resets were implemented independently and verified by regression tests.

## Engineering validation

```text
v5.0 frozen fast-core tests: 5 passed
v5.6 fast/slow reset tests: 4 passed
```

Therefore the experiment executed with the intended independent reset semantics.

## Semantic-context result

Mean across five recurrent seeds on the v5.4 five-domain development fixture:

| Condition | Accuracy |
| --- | ---: |
| fast recurrent only | 0.595 |
| slow EMA only | **0.800** |
| naive dual concat | 0.675 |
| dual after fast reset | 0.495 |
| dual after slow reset | 0.495 |
| dual after both reset | 0.500 |

Gaps:

```text
dual - fast       = +0.080
dual - slow       = -0.125
dual - slow reset = +0.180
```

The slow state clearly carries semantic information, but naive concatenation does not preserve the full slow-only performance.

## Temporal-order result

Controlled same-multiset A-before-B vs B-before-A task:

| Condition | Accuracy |
| --- | ---: |
| fast recurrent only | 0.905 |
| slow EMA only | **1.000** |
| dual | **1.000** |
| dual after fast reset | 0.650 |
| dual after slow reset | 0.625 |
| dual after both reset | 0.500 |

Gaps:

```text
dual - fast       = +0.095
dual - slow       =  0.000
dual - fast reset = +0.350
```

Although fast reset strongly damages the dual probe, slow EMA alone already solves the order benchmark perfectly. Therefore this task cannot establish incremental temporal value from the fast recurrent component.

## Predeclared gate

```text
dual_semantic_macro_at_least_0_78 = false
dual_semantic_beats_fast_by_0_10 = false
slow_reset_reduces_semantic_by_0_10 = true
dual_order_at_least_0_80 = true
dual_order_beats_slow_by_0_05 = false
fast_reset_reduces_order_by_0_10 = true
dual_adds_value_without_semantic_regression = false
all_primary_gates = false
```

**v5.6 fails the preregistered dual-timescale development gate.**

## Interpretation

Two different limitations are exposed.

### 1. Naive concatenation is not a good integration test

The slow state alone reaches `0.800`, while the raw concatenated dual representation reaches only `0.675`.

The fast trace contributes 2048 raw features (`16 ticks × 128 units`) while the slow MiniLM state contributes 384 features. With only 24 training arms per domain, the high-dimensional fast block can dominate the downstream ridge readout even though the slow block is individually more useful.

This does **not** show that the slow state stopped carrying semantic information. Slow-only performance directly shows the information remains present. It shows that naive feature concatenation is an inadequate integration/readout method.

### 2. The temporal-order benchmark is too easy for EMA

The slow EMA state achieves `1.000` on the A-before-B vs B-before-A task. EMA is itself order-sensitive because recent events receive larger weights. Therefore the benchmark does not isolate a kind of temporal structure that requires fast recurrent trajectory dynamics.

The fact that fast reset hurts the dual probe is not sufficient evidence of unique fast-state value when slow-only already solves the task perfectly.

## What v5.6 does support

The cleanest conclusion is architectural rather than performance-based:

> persistent slow semantic memory is useful, but the current experiment does not yet justify a separate fast recurrent component or a naive concatenation interface between the two timescales.

## Next version

**v5.6.1 — Readout & Temporal-Control Diagnostic** should keep the v5.6 states frozen and diagnose both failures separately.

### A. Semantic integration diagnostic

Do not change either state generator.

Compare:

- slow only
- fast only
- raw concatenation
- deterministic equal-dimensional random projection of fast and slow blocks before concatenation
- independent fast/slow probe-score fusion as a diagnostic readout

No projection may use task labels.

If a balanced fixed representation recovers slow-only performance, the v5.6 semantic failure is primarily an integration/readout dimensionality problem.

### B. Harder temporal-control benchmark

Replace fixed global `ALPHA/BETA` with pair-specific unseen token identities and classify sequence **structure** rather than fixed token order.

Example patterns with the same multiset:

```text
pattern 0: A B A B
pattern 1: A A B B
```

Each pair gets unique A/B event identities, and test pairs use identities never seen during training. The probe therefore cannot memorize a global ALPHA/BETA direction. It must generalize sequence structure across identities.

Compare slow-only, fast-only, dual, fast-reset, and slow-reset.

v5.6.1 is diagnostic only; v5.6 remains a failed development version.
