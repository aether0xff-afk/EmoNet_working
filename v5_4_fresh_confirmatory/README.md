# EmoNet v5.4 — Fresh Semantic-Memory Confirmatory Test

## Version boundary

- v5.0: temporal-memory baseline — frozen
- v5.1–v5.1.2: semantic diagnostics/calibration — frozen
- v5.2: cosine-memory development — frozen failed result
- v5.2.1: objective/readout diagnosis — frozen
- v5.3: contrastive-memory development — frozen development pass
- **v5.4: fresh confirmatory protocol**

v5.4 changes no v5.3 architecture or training hyperparameter. It introduces a completely new semantic fixture that has not been used in any previous version.

## Frozen before first v5.4 result

The following are fixed before the first GitHub Actions benchmark is inspected:

- encoder: `sentence-transformers/all-MiniLM-L6-v2`
- recurrent architecture: v5.2 `LearnedLeakyRecurrentCore`, unchanged
- hidden size: 128
- event ticks: 16
- stimulation ticks: 6
- update rate: 0.35
- recurrent stabilization: spectral norm <= 0.98
- objective: v5.3 exact delayed-event contrastive retrieval
- memory lags: 1 / 2 / 3
- temperature: 0.07
- optimizer: AdamW
- epochs: 150
- learning rate: 0.002
- weight decay: 1e-5
- recurrent seeds: `7, 13, 21, 42, 100`
- downstream readout: domain-conditioned standardized ridge binary probe, alpha 3.0
- reset and opposite-arm controls
- v5.0 random recurrent baseline
- simple EMA embedding-memory baseline, decay 0.80
- evaluation spacing: semantic event followed by two shared neutral events before the same current event

No benchmark task label or emotion label enters core training.

## Fresh domains

The v5.4 fixture uses five domains not present in v5.1:

1. `connectivity` — communication link reachable vs unavailable
2. `capacity` — required capacity available vs exhausted
3. `integrity` — item/system integrity verified vs compromised
4. `route` — required route open vs obstructed
5. `assignment` — resource/task assigned to the actor vs assigned elsewhere

Each domain has disjoint train and held-out paraphrase templates.

Each pair keeps constant:

- prefix
- neutral intervening events
- last history event
- current input

Only the earlier latent semantic-state statement differs between the two arms.

## Confirmatory baselines

The fresh fixture is evaluated with:

- `v5.4_contrastive_recurrent`
- `v5.0_random_recurrent`
- `ema_embedding_memory`
- contrastive reset trace
- contrastive wrong/opposite trace

The core optimizer sees only frozen event embeddings and self-supervised event identities.

## Predeclared confirmatory gates

A **confirmatory semantic-memory pass** requires all of:

1. held-out lag-3 exact-event retrieval top-1 >= `0.20`;
2. contrastive recurrent semantic macro >= `0.70`;
3. contrastive recurrent improves by >= `+0.10` over v5.0 random recurrent;
4. contrastive recurrent beats reset by >= `+0.15`;
5. contrastive recurrent beats wrong/opposite trace by >= `+0.15`;
6. at least 4 of 5 recurrent seeds achieve semantic macro >= `0.65`.

EMA is a mandatory complexity baseline but is **not** part of the semantic-memory confirmation gate. Whether recurrent dynamics beat EMA is reported as a separate claim.

## No post-result tuning rule

After the first completed v5.4 benchmark is inspected:

- the fixture is frozen;
- the above hyperparameters and gates are frozen;
- a failed gate means v5.4 fails;
- performance tuning must move to a new version;
- only implementation bugs that make the declared protocol execute incorrectly may be fixed in v5.4, and any such rerun must be explicitly documented.

## Claim boundary

If v5.4 passes, the defensible claim is:

> a label-free contrastive recurrent memory protocol selected during development reproduces semantic-context retention on a fresh held-out natural-language fixture, with the information disappearing under reset and wrong-trace controls.

This still does **not** establish emotion representation or superiority over simple EMA memory.
