# Research Gaps (2026-04-09)

## 1. Surface softening remains the main user-facing failure

Evidence:

- `outputs/paper/refresh_2026-04-09_calref_v1/tables/paper_refresh_summary.json`
- `outputs/research/trajectory_batch_v1_gpt54/episode_summary.csv`

What we know:

- raw trajectory and GPT-5.4 episode interpretation are substantially more honest than the final generated responses
- the final response still often drifts toward calm, advisory, counseling-like surface tone

Implication:

- the bottleneck is no longer only branch generation
- it is the final `episode -> response surface` conversion

## 2. Style supervision is still heavily biased toward safe/cooperative tone

Evidence:

- `softness=0.9276`
- `calmness=0.9132`
- `cooperativeness=0.9202`
- `positivity=0.9051`
- `hostility=0.0003`
- `resentment=0.0003`
- `despair=0.0044`

Implication:

- even a better internal affect signal can be re-softened by the learned style target

## 3. `z -> s` remains weaker than the text baseline

Evidence:

- `text_tfidf` mean MAE `0.114628`
- `structfix_learned_z64` mean MAE `0.117328`

Implication:

- the branch/trajectory system improved, but its compressed predictor path is still not the strongest route to style control

## 4. Positive arousal and anticipatory states remain fragile

Evidence:

- `s_001913` needed GPT-5.4 episode interpretation to avoid anger-like heuristic misread
- `s_000527` showed true no-ignition / suspended anticipation

Implication:

- the system is better at persistent negative high-arousal episodes than at
  - positive approach tension
  - suspended anticipation
  - weakly ignited uncertainty

## 5. Upper-tail saturation is reduced but still present

Evidence:

- branch `p90=126`, `p95=126`, `max=126`

Implication:

- some runs still sit near the ceiling
- this may flatten later-phase differentiation

## 6. v4 code structure is improved, but `legacy_cli.py` is still too large

Evidence:

- public facade exists, and active modules were extracted
- generation-related logic still largely lives in `emonet/legacy_cli.py`

Implication:

- maintainability risk remains
- future bugs will cluster around generation path edits unless command groups are separated further

## Priority next steps

1. Run a full `episode_trace` / `hybrid_episode` matrix and score it with GPT-5.4
2. Compare whether `episode_trace` improves `naturalness` without collapsing rawness
3. If softening remains, simplify the final response prompt further and reduce style influence when episode risk is high
4. Revisit style supervision with a split between `felt_state` and `response_style`
5. Split generation command code out of `legacy_cli.py`
