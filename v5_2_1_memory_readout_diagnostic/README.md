# EmoNet v5.2.1 — Memory Readout Diagnostic

## Version boundary

- v5.2 learned-memory core: frozen failed result
- **v5.2.1 changes no core architecture, optimizer, training objective, epoch count, or benchmark fixture**
- this version inspects only what the already-trained lag-3 memory head can recover from the frozen recurrent state

## Question

v5.2 achieved held-out lag-3 embedding cosine around 0.64 but only about 56% semantic-state accuracy from a raw-trace probe.

Two explanations remain:

1. **readout/sample-efficiency bottleneck** — the hidden state contains useful semantic memory, but a small downstream probe cannot discover the direction efficiently;
2. **objective-information bottleneck** — cosine reconstruction preserves broad sentence meaning while discarding the fine semantic distinction needed by the benchmark.

## Readouts

For every trained seed and held-out paraphrase:

- original semantic-event embedding
- lag-3 memory-head reconstruction
- raw final-event trace
- simple EMA memory

Two probes are evaluated on the reconstructed embedding:

1. `reconstruction_native_probe`: train and test on reconstructed embeddings;
2. `semantic_geometry_transfer`: train on the original semantic-event embeddings, then apply that frozen probe directly to reconstructed embeddings.

The second test asks whether the memory head preserves the original semantic geometry rather than merely creating a new decodable coordinate system.

## Interpretation

- reconstruction probe high, raw trace low → readout/sample-efficiency problem;
- reconstruction probe low despite high cosine → delayed cosine objective preserves the wrong information;
- transfer low but reconstruction-native high → information survives but semantic geometry is distorted;
- both reconstruction tests high → raw-trace probe was the main bottleneck.

This is diagnostic only and cannot rescue v5.2 retroactively.
