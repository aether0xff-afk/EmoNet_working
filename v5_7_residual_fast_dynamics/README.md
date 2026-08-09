# EmoNet v5.7 — Residual-Driven Fast Dynamics

## Version boundary

- v5.6 dual-timescale concat experiment: frozen failed result
- v5.6.1 readout/temporal diagnostic: frozen
- **v5.7 changes only what drives the fast channel**

Previous version directories are not modified.

## Motivation

v5.6.1 established a clean separation:

```text
slow EMA semantic retention = 0.800
slow identity-disjoint structure = 0.5125
fast raw-embedding semantic = 0.595
fast identity-disjoint structure = 0.565
```

The raw fast channel redundantly receives the full semantic embedding even though slow memory already preserves semantics better. This makes the fast channel compete with slow memory instead of specializing in transient change.

## Architecture

Slow memory is unchanged:

```text
m_t = 0.8 m_(t-1) + 0.2 e_t
```

The v5.0 recurrent equations/topology are also unchanged. Only the fast input changes.

Old fast input:

```text
fast <- e_t
```

v5.7 fast input:

```text
r_t = e_t - m_(t-1)
fast <- r_t
```

Then slow memory is updated with `e_t`.

Thus the two channels have explicit roles:

- slow: persistent context / semantic baseline
- fast: deviation from that baseline / transient change

No emotion labels, semantic-state labels, task labels, learned gates, or new optimizer are introduced.

## Why residual input is relevant

The v5.6.1 structural benchmark uses pair-specific unseen event identities and asks whether the sequence is:

```text
ABAB  vs  AABB
```

Both classes contain the same `2A + 2B` multiset. The difference is the pattern of change.

A residual relative to slow context naturally emphasizes transitions:

- repeated event -> smaller residual
- changed event -> larger/different residual

This provides an identity-independent signal without defining A/B semantics by hand.

## Development benchmarks

### A. Identity-disjoint temporal structure

Reuse the frozen v5.6.1 `ABAB vs AABB` construction:

- 80 train pairs
- 40 held-out test pairs
- pair-specific A/B identities
- train/test identities disjoint
- same event multiset
- same current input

Compare:

- old v5.0 raw-input fast state
- v5.7 residual-driven fast state
- slow EMA state
- residual fast reset before current
- a direct residual-change diagnostic baseline
- relational structure validity baseline

### B. Semantic preservation

Reuse the v5.4 five-domain development fixture only as a diagnostic.

The slow semantic state must remain unchanged. We report:

- slow EMA semantic macro
- residual-fast semantic macro

v5.7 does not require the fast channel to carry semantics.

## Predeclared development gates

Residual-fast development passes only if all are true:

1. structural temporal accuracy >= `0.70`;
2. residual-fast improves by >= `+0.10` over old raw-input fast;
3. residual-fast improves by >= `+0.12` over slow EMA on the structural task;
4. fast reset reduces structural accuracy by >= `0.12`;
5. relational task-validity baseline remains >= `0.95`;
6. slow semantic macro remains >= `0.78`.

A separate complexity note must report whether the direct label-free residual-change baseline matches or beats the recurrent fast channel. If it does, the recurrent dynamics are still not justified over a simpler transient representation.

## Claim boundary

v5.7 is development-only. Passing it would show that driving fast recurrence by deviation from slow context is a better transient representation than feeding the full semantic embedding redundantly. It would not establish emotion or semantic-memory confirmation.
