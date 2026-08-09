# EmoNet v5.6 — Dual-Timescale State

## Version boundary

- v5.0: fixed-random temporal dynamics — frozen
- v5.1–v5.5: semantic-memory/objective experiments — frozen
- **v5.6: architecture/control experiment**

v5.6 does not modify any previous version directory and introduces no new training objective.

## Motivation

The clean-line results now show a repeated separation:

- fast recurrent dynamics preserve temporal trajectory/order well;
- recurrent-only semantic state is weak and unstable;
- simple persistent EMA embedding memory is consistently strong on semantic context.

Forcing one homogeneous recurrent vector to act simultaneously as transient dynamics and persistent semantic memory may therefore be the wrong architecture.

v5.6 explicitly separates two timescales.

## Architecture

```text
frozen semantic embedding
        │
        ├───────────────> slow persistent state
        │                    EMA decay = 0.80
        │
        └──> fixed recurrent fast dynamics
                 ↓
             fast trace

internal state = [slow state, fast trace]
```

### Fast state

Exactly the frozen v5.0 fixed-random recurrent substrate:

- 128 units
- event ticks 16
- stimulation ticks 6
- no learning
- no affect labels

### Slow state

Exactly the simple EMA embedding-memory baseline already used in v5.2–v5.5:

```text
m_t = 0.8 m_(t-1) + 0.2 e_t
```

The final slow vector is normalized at readout.

There is no learned gate and no hand-authored emotion meaning.

## Why this is a fair architecture test

v5.6 does not claim novelty merely by concatenating two representations.

The dual state is useful only if each component contributes something the other does not:

- slow state should prevent semantic information loss;
- fast state should provide temporal information beyond slow memory.

If EMA alone matches the dual state on both tasks, the dual architecture is not justified.

## Development benchmarks

### A. Semantic-context retention

Reuse the already-inspected v5.4 five-domain fixture as development data.

Compare:

- fast recurrent only
- slow EMA only
- dual fast + slow
- dual with fast reset before current
- dual with slow reset before current
- dual with both reset

### B. Temporal-order retention

Use a controlled A-before-B vs B-before-A benchmark with:

- identical event multiset
- identical current event
- same train/test construction

Compare the same fast / slow / dual conditions.

## Reset semantics

- `reset_fast`: erase recurrent state only; preserve slow memory
- `reset_slow`: erase slow memory only; preserve recurrent state
- `reset_both`: erase both before the current event

This is required to identify where each kind of information actually lives.

## Predeclared development gates

The dual-timescale hypothesis passes only if all are true:

### Semantic preservation

1. dual semantic macro >= `0.78`;
2. dual semantic macro >= fast-only semantic + `0.10`;
3. resetting slow memory reduces semantic accuracy by >= `0.10`.

### Fast-state incremental value

4. dual temporal-order accuracy >= `0.80`;
5. dual temporal-order accuracy exceeds slow-only order accuracy by >= `0.05`;
6. resetting fast state reduces order accuracy by >= `0.10`.

### Complexity rule

7. dual state must beat EMA/slow-only on at least one preregistered task without falling more than `0.02` below slow-only on semantic retention.

If the slow EMA state solves both tasks by itself, v5.6 fails: the recurrent component has not justified its complexity.

## Claim boundary

v5.6 is a development architecture test on already-inspected data. Passing it would justify a dual-timescale state as a candidate substrate, not confirm semantic memory or emotion.
