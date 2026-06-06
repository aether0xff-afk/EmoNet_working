# v8 Brainstorming

## Always-On EmoNet

EmoNet should not reset neurons on every execution. The neural system should remain alive across ticks, and every tick should receive stimulation.

Stimulation sources include:

- user message
- user typing state
- EmoNet processing state
- LLM answering state
- silence
- elapsed time
- other environmental state

The important architectural shift is:

```text
bad:  run(input) -> reset neurons -> response -> stop
good: always-on neurons -> tick stimulation -> trace -> episode -> response
```

## Per-Neuron Memory

Memory should not exist only as a separate memory node. Each neuron should be able to carry local memory.

If memory strength `K` crosses `remember_threshold`, the neuron stores a local trace.

```text
if K > remember_threshold:
    neuron.store(memory_trace)
```

The memory should be distributed across neurons and reinforced through activation, connection strength, and repeated stimulation.

## 8D Hormone Stim Vector

The primary stimulus vector should be 8D. It should act as neural pressure, not as final emotion labels.

Current working axis names:

```text
valence
arousal
threat
novelty
agency
social_pressure
fatigue
coherence
```

These names can change later. The important rule is that the vector is an input to neural dynamics, not the final interpretation.

## Clustering System

Clusters should represent functional neural states, not fixed emotion labels.

```text
neural activity -> cluster state -> trace -> episode
```

Clusters are evidence. Episode formation interprets that evidence. The response should not be generated directly from cluster names.

## Mini Cell Idea

Each neuron can be imagined as a mini cell.

A neuron lives if it is connected. If it becomes isolated, it decays or dies. Neurons may have a survival pressure: stay connected, communicate with other neurons, preserve useful activity, and survive as long as possible.

This is not yet an implementation plan. It is a v8 idea to explore after the persistent runtime and per-neuron memory system are stable.
