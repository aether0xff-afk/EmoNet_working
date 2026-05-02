# EmoNet v3.1

v3.1 is the research branch for the idea:

> trace is not just metadata about emotion; trace is the emotion-state representation itself.

v4 remains the app and evaluation implementation branch. v3.1 focuses on the scientific question behind EmoNet: whether neural/appraisal traces form a stable, structured emotional state space.

## Core Hypothesis

The previous pipeline used trace as an explanation or conditioning payload:

```text
stimulus -> episode trace -> response prompt -> generated response
```

v3.1 treats trace as the emotion itself:

```text
stimulus -> trace dynamics -> emotion-state representation -> appraisal/action constraints -> response
```

This changes the proof target. Instead of asking only whether generated responses sound better, v3.1 asks whether trace space has emotion-like structure:

- similar emotional situations produce nearby traces
- different appraisal/action patterns separate in trace space
- trace perturbation changes emotional interpretation
- trace removal weakens appraisal fidelity and affect preservation

## Directory Layout

```text
v3.1/
  README.md
  docs/
    TRACE_AS_EMOTION_DESIGN.md
    EXPERIMENT_ROADMAP.md
  scripts/
    trace_emotion_probe.py
  outputs/
    .gitkeep
```

## First Experiment

The first probe is deliberately lightweight. It reads a targeted record CSV, converts trace fields into a simple mixed categorical/numeric representation, and reports whether nearest neighbors in trace space share the same emotional attributes.

Example input:

```text
../v4/outputs/experiments/superiority_targeted_v1/targeted_records.csv
```

Example output:

```text
outputs/trace_emotion_probe_summary.json
```

## Why This Exists

The v4 targeted superiority result suggests that trace contains useful emotional information. But response quality alone is indirect evidence. v3.1 is where that claim becomes testable at the representation level.

