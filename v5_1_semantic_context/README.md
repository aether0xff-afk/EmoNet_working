# EmoNet v5.1 — Semantic Context Memory

## Version boundary

`v5_clean/` is frozen as **v5.0: Temporal Memory Baseline**.

This directory is **v5.1** and tests one new claim only:

> A frozen v5.0 recurrent trace can preserve natural-language semantic context across paraphrases, even when the current text and final history event do not reveal that context.

No v5.0 core files are modified for this experiment.

## What v5.1 does NOT test

- emotion representation
- valence/arousal
- learned recurrent dynamics
- EmoNet superiority over GRU/LSTM/ESN
- causal behavior changes

Those are later versions.

## Input representation

v5.1 uses a frozen sentence embedding model (`sentence-transformers/all-MiniLM-L6-v2`) only as a neutral semantic input representation. The embedding model is never fine-tuned on the benchmark labels.

The benchmark deliberately separates train/test paraphrase templates so a probe cannot rely only on memorizing the exact training sentence.

## Controlled semantic task

Each pair contains two histories that describe opposite latent world states using natural language. The current sentence and final history event are identical across the pair.

Examples of latent context domains include:

- access permitted vs access blocked
- resource available vs resource unavailable
- device operational vs device unusable
- schedule still valid vs expired
- authorization active vs revoked

The binary label is a task-state label (`usable/proceedable` vs `blocked/unusable`), not an emotion label.

## Conditions

- `current_text_only`
- `last_event_only`
- `history_bag_embedding`
- `full_history_embedding`
- `trace_only_real`
- `text_plus_real_trace`
- `text_plus_temporal_shuffle`
- `text_plus_wrong_trace`
- `text_plus_reset_trace`

The main trace probe is trained only on real traces. The same probe is evaluated on wrong/reset/shuffled controls.

## Interpretation rule

A successful v5.1 result can support only:

> the frozen recurrent trace preserves decodable semantic context beyond the current input.

It cannot yet support an emotion claim.
