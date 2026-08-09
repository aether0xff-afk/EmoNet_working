# EmoNet v5.4.1 — Contrastive Memory Readout Diagnostic

## Version boundary

- v5.4 fresh confirmatory test: frozen failed result
- **v5.4.1 changes no recurrent architecture, self-supervised objective, optimizer, epochs, seeds, or fresh fixture**
- diagnostic only

## Why this diagnostic exists

v5.4 produced a striking mismatch:

```text
held-out exact lag-3 event retrieval = 0.365
raw trace semantic macro             = 0.630
EMA semantic macro                   = 0.800
```

The contrastive memory head can identify the exact delayed event far above chance, yet the frozen recurrent trace does not expose a robust abstract semantic-state geometry.

## Questions

v5.4.1 separates three possibilities:

1. **trace-readout bottleneck** — the trained lag-3 memory head contains stronger semantic state than the raw recurrent trace;
2. **geometry distortion** — semantic state is decodable from the memory-head output only after learning a new probe, while the original frozen-embedding semantic direction does not transfer;
3. **instance-identity objective** — the model retrieves specific sentence identity without forming a stable positive/negative state abstraction.

## Diagnostic readouts

Using the exact frozen v5.4 training protocol:

- original semantic-event embedding probe
- raw recurrent trace probe
- lag-3 memory-head output with a native semantic probe
- semantic probe trained on original embeddings and transferred directly to memory-head output
- semantic-only event retrieval top-1
- retrieval polarity accuracy: whether the retrieved semantic candidate has the correct latent state even if the exact paraphrase is wrong
- EMA semantic probe

Task-state labels are used only after core training for diagnostic evaluation.

## Interpretation

- memory-head semantic high, raw trace low → trace/readout geometry bottleneck;
- native memory-head high but geometry transfer low → semantic geometry distortion;
- exact/semantic retrieval high but polarity low → instance-identity objective is preserving the wrong abstraction;
- memory-head semantic itself low → contrastive objective still does not create a useful semantic state representation.

No result in v5.4.1 can retroactively turn v5.4 into a confirmatory pass.
