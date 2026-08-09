# EmoNet v5.5 — Predictive State Memory

## Version boundary

- v5.0: temporal-memory sanity baseline — frozen
- v5.1–v5.1.2: semantic-memory diagnostics — frozen
- v5.2: delayed cosine reconstruction — frozen failed result
- v5.3: delayed event identity retrieval — frozen development pass
- v5.4: fresh confirmatory test — frozen failed result
- v5.4.1: abstraction diagnostic — frozen
- **v5.5: predictive-state development**

v5.5 does not modify any previous version directory.

## Motivation

v5.4.1 showed that exact delayed-event retrieval mainly learned instance/domain identity:

```text
domain identity = 1.000
exact semantic-event retrieval = 0.380
state polarity = 0.675
```

Remembering which sentence happened is therefore not sufficient to create a stable abstract state.

v5.5 changes the learning question:

> Instead of remembering the past event itself, can the recurrent state preserve the aspects of history that are useful for predicting what happens next?

This is closer to a predictive-state representation.

## Label-free training protocol

Each development sequence has the form:

```text
prefix
→ latent-state event
→ neutral event
→ neutral/shared event
→ SAME CURRENT SITUATION
→ observed FUTURE CONSEQUENCE
```

The recurrent core receives events only up to the current situation. A prediction head must identify the actually observed next consequence among candidate future-event embeddings.

The optimizer receives:

- frozen sentence embeddings
- the identity of the actually observed next event

The optimizer never receives:

- positive/negative state labels
- usable/blocked labels
- emotion labels
- valence/arousal labels
- downstream probe labels

The future event is ordinary sequence data, not an explicit state label.

## Why this may create abstraction

Different paraphrases that lead to the same kind of consequence should become useful in similar ways because the state is trained for **future prediction**, not exact reconstruction of the past wording.

Histories with opposite consequences must remain distinguishable even when they belong to the same domain.

## Architecture

The recurrent substrate is the same leaky-tanh architecture used in v5.2–v5.4:

- hidden size 128
- 16 ticks per event
- 6 stimulation ticks
- update rate 0.35
- spectral stabilization <= 0.98

v5.5 adds only a training-time linear future-prediction head from recurrent state to frozen embedding space.

## Development data

v5.5 reuses the v5.4 five-domain histories as development material, because v5.4 is already inspected and cannot be confirmatory data anymore.

For v5.5 only, each arm receives a future-consequence sentence. Train and held-out test consequence paraphrases are disjoint.

Domains:

- connectivity
- capacity
- integrity
- route
- assignment

The current sentence remains identical between positive/negative arms of every pair. The future consequence differs because the earlier state differs.

## Evaluation point

Semantic-state probes read the recurrent trace **after the same current situation and before the future consequence is shown to the model**.

Therefore future information cannot leak into the evaluated state.

## Baselines

Mandatory comparisons:

- v5.0 frozen random recurrent
- historical v5.4 contrastive-past-memory result (`0.630` on this fixture)
- v5.5 predictive recurrent
- simple EMA embedding memory
- reset-before-current trace
- opposite-arm/wrong trace

Also report zero-shot held-out future-consequence retrieval.

## Predeclared development gates

v5.5 development succeeds only if all are true:

1. held-out future-consequence retrieval top-1 >= `0.30`;
2. predictive recurrent semantic macro >= `0.72`;
3. predictive recurrent improves by >= `+0.10` over v5.0 random recurrent;
4. predictive recurrent improves by >= `+0.08` over historical v5.4 contrastive-past-memory (`0.630`);
5. predictive recurrent beats reset by >= `+0.15`;
6. predictive recurrent beats wrong/opposite trace by >= `+0.15`;
7. at least 4 of 5 recurrent seeds achieve semantic macro >= `0.68`.

EMA remains a separate complexity baseline. v5.5 may pass semantic-state development while still failing to justify recurrent complexity over EMA.

## Claim boundary

v5.5 uses previously inspected history domains and is development only.

If it passes, the result supports only:

> future-prediction self-supervision creates a more stable semantic state than past-event identity memory on the development fixture.

A later fresh version is still required for confirmation. Emotion/affect probing remains premature until semantic-state confirmation succeeds.
