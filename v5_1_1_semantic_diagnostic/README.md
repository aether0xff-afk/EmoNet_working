# EmoNet v5.1.1 — Semantic Failure Diagnostic

## Version boundary

- **v5.0**: temporal-memory baseline — frozen
- **v5.1**: first natural-language semantic-context test — frozen failed result
- **v5.1.1**: diagnostic/calibration only

v5.1.1 does not change the scientific claim and does not modify the v5.0 recurrent core or the v5.1 benchmark fixture.

## Question

Why did v5.1 produce only 55.5% real-trace accuracy?

The diagnostic separates four possible bottlenecks:

1. semantic input representation is not sufficiently decodable;
2. recurrent state loses the semantic signal as neutral events intervene;
3. the v5.0 handcrafted trace summary discards information that remains in the raw trace;
4. a simple linear readout cannot recover the information.

## Diagnostic matrix

The same held-out paraphrase split from v5.1 is reused.

Semantic-event-to-current gap:

- `gap0`: semantic event immediately precedes current input
- `gap1`: one shared neutral event intervenes
- `gap2`: two shared neutral events intervene; this reproduces v5.1 spacing

Trace readout:

- `final_state`
- `summary_features` (the v5.0 sanity summary)
- `raw_flattened_trace`

Input ceiling:

- semantic event embedding alone
- history bag embedding

Controls:

- reset before current
- opposite-arm/wrong trace

## Interpretation

v5.1.1 is successful if it identifies a bottleneck. It is not required to make the semantic-memory hypothesis pass.
