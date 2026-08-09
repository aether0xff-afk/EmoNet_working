# EmoNet v5.4 Result — Fresh Confirmatory Semantic Memory

검증일: 2026-08-09 KST

## Version boundary and preregistration

- base: frozen v5.3 development pass
- branch: `feature/v5.4-fresh-confirmatory`
- PR: #11
- preregistration commit existed before the first benchmark result
- first completed confirmatory run: `31290774060`
- first-result artifact: `v5.4-fresh-confirmatory-first-result`
- artifact id: `9031305315`
- fixture SHA-256: `0fa99d76d7163124ae0479e1916c7c3f48532e89194fead4f479ae870a7804a9`

The architecture, objective, temperature, optimizer, epoch count, seeds, probe type, controls, baselines, spacing, and acceptance gates were frozen before this result was inspected.

The v5.4 fixture used five domains not present in the v5.1 development fixture:

- connectivity
- capacity
- integrity
- route
- assignment

Train and held-out paraphrase semantic templates were disjoint. Fixture regression tests also verified that the v5.4 semantic sentences were not reused from v5.1.

## First-result metrics

Mean across five recurrent seeds:

| Metric | Result |
| --- | ---: |
| held-out lag-3 exact event retrieval top-1 | **0.365** |
| held-out lag-3 cosine | 0.373 |
| contrastive recurrent semantic macro | **0.630** |
| reset trace semantic macro | 0.500 |
| wrong/opposite trace semantic macro | 0.370 |
| v5.0 random recurrent macro | 0.595 |
| EMA embedding memory macro | **0.800** |
| semantic-event input diagnostic macro | 0.800 |

Gaps:

```text
contrastive - random = +0.035
contrastive - reset  = +0.130
contrastive - wrong  = +0.260
contrastive - EMA    = -0.170
```

## Seed-wise semantic macro

| Seed | Contrastive | Random | EMA | Retrieval top-1 |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.625 | 0.575 | 0.800 | 0.400 |
| 13 | 0.575 | 0.575 | 0.800 | 0.325 |
| 21 | 0.700 | 0.625 | 0.800 | 0.325 |
| 42 | 0.575 | 0.475 | 0.800 | 0.450 |
| 100 | 0.675 | 0.725 | 0.800 | 0.325 |

Only 2 of 5 seeds reached the preregistered `>= 0.65` seed threshold.

## Domain pattern

Semantic input / EMA / contrastive performance is not uniform across the fresh domains.

The clearest positive domain is `connectivity`:

```text
semantic input = 1.000
EMA            = 1.000
contrastive    ≈ 0.850 mean across seeds
```

Other domains are substantially weaker, especially `capacity` and `assignment`. This shows that the method can preserve some semantic distinctions well, but the effect does not generalize uniformly across fresh state types.

## Predeclared confirmatory gate

```text
heldout_lag3_retrieval_top1_at_least_0_20 = true
contrastive_semantic_macro_at_least_0_70 = false
contrastive_beats_random_by_0_10 = false
contrastive_beats_reset_by_0_15 = false
contrastive_beats_wrong_by_0_15 = true
at_least_4_of_5_seeds_at_or_above_0_65 = false
confirmatory_semantic_memory_pass = false
```

**v5.4 fails the preregistered confirmatory semantic-memory test.**

The GitHub Actions workflow itself succeeded; the scientific hypothesis gate did not.

## Interpretation

The fresh result is mixed rather than null:

1. exact delayed-event retrieval remains strong (`0.365` over a 75-event held-out vocabulary), so the contrastive objective does generalize as a delayed identity-memory mechanism;
2. real trace beats wrong/opposite trace by `+0.260`, so the trace contains sample-specific semantic signal;
3. however, real trace improves only `+0.035` over the random recurrent baseline and only `+0.130` over reset, below preregistered thresholds;
4. EMA reaches `0.800`, outperforming the recurrent trace by `0.170`.

Therefore the development result from v5.3 (`0.725` semantic macro) did **not** reproduce strongly enough on a fresh fixture.

The defensible conclusion is:

> contrastive delayed-event training reliably improves exact temporal event retrieval, and its recurrent trace carries some semantic-state information, but the current recurrent-state geometry does not yet provide robust cross-domain semantic memory beyond a random reservoir or simple EMA baseline.

This weakens the semantic-memory claim and prevents moving on to affect/emotion probing yet.

## What this rules out

It would be incorrect after v5.4 to claim:

- semantic-context memory has been confirmed;
- learned EmoNet dynamics beat a random recurrent baseline generally;
- recurrent EmoNet memory is preferable to simple embedding memory;
- affective structure should now be probed as if semantic state were solved.

## Next version

Any diagnosis or improvement must move to a new version; v5.4 remains frozen as the failed first confirmatory result.

A useful next diagnostic is to explain the mismatch:

```text
exact delayed identity retrieval = 0.365   (works)
semantic downstream trace        = 0.630   (weak)
EMA semantic memory              = 0.800   (strong)
```

The next version should determine whether the problem is:

1. the recurrent state encodes event identity in a geometry poorly aligned with compositional semantic state;
2. the final-event trace overwrites information that the lag memory head can still retrieve;
3. the exact-event objective encourages instance memorization rather than a stable state abstraction;
4. the recurrent architecture itself is inferior to direct persistent embedding memory for this task.
