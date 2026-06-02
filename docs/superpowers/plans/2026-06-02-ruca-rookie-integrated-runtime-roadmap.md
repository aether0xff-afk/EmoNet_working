# Ruca/Rookie integrated runtime roadmap

**Date:** 2026-06-02  
**Working branch:** `feature/ruca-rookie-integrated-runtime`

## Goal

Implement the full Ruca/Rookie character runtime discussed in this project folder without reducing emotion to a hand-written script.

The runtime may provide events, elapsed time, context, memory cues, relationship state, scene pressure, and delivery gates. EmoNet must remain the felt-state engine.

```text
events + elapsed time + context
  -> EmoNet felt-state trace
  -> interpretable semantic report / memory cues
  -> inner-voice shadows
  -> response gate
  -> character expression
```

Inner voices are not substitute emotion engines. They are language shadows of the internal trace and may be used for interpretation, imagination, private deliberation, and final-expression guidance.

## Current baseline restored

The integration branch starts by restoring an executable v6 baseline.

- `no_reply` is the standard autonomous time event.
- Legacy `silence_tick` and `long_silence` inputs remain compatibility aliases.
- New event text and remembered reference text are kept separate.
- Ruca/Ricky/Rocky inner voices, Rookie context, trait EMA, plot pressure, relationship graph, memory records, response gate, visible-speaker selection, and session persistence are connected through one pipeline.
- EmoNet mode fails loudly if required artifacts or dependencies are unavailable unless an explicit development fallback is enabled.
- The rule-based emotion path and rule composer remain development scaffolding only.

## Phase 0 — Stabilize the executable baseline

### Deliverables

- Remove unresolved merge markers from active v6 runtime files.
- Keep `RucaPipeline` as the orchestration boundary.
- Standardize event transport:
  - `user_message`
  - `no_reply`
  - legacy `silence_tick`
  - legacy `long_silence`
- Preserve the distinction between:
  - `source_text`: newly received user text
  - `reference_text`: previous useful text recalled only as context
  - `elapsed_minutes`
- Add focused integration tests and CLI smoke tests.

### Exit criteria

- A normal user event produces a visible response.
- A short `no_reply` event updates state without speaking.
- A sufficiently long `no_reply` event may produce a low-pressure check-in.
- Session files persist emotion state, trait state, Rookie plot state, relationship graph, recent history, and visible speaker.

## Phase 1 — Make EmoNet authoritative for felt state

The current rule path must not become the final emotion architecture. It is only a test scaffold.

### Work items

- Define an event adapter that converts both user messages and no-reply time events into EmoNet inputs.
- Preserve raw trace output before semantic interpretation.
- Select one dominant branch after the whole response episode rather than at every timestep.
- Keep neuron path information and cluster-path information together.
- Feed the global history `H` and the dominant branch into the history encoder.
- Retain z-dimension ablations for `32`, `64`, and `128`.

### Semantic report

Expose an interpretable intermediate report for downstream character runtime use. It is not a ground-truth emotion label.

Suggested fields:

```text
episode_label
valence
arousal
target
control_state
social_orientation
preserve
avoid
action_tendency
```

The semantic report interprets raw trace segments and state flow. It does not replace the raw trace.

### Exit criteria

- `--emonet` is the default production path.
- A no-reply event changes EmoNet state through a designed time-event encoding rather than a rule-only delta.
- Debug output contains raw trace, dominant branch, semantic report, and selected downstream cues.

## Phase 2 — Add neural-addressed memory

Reuse the design direction in the existing `v7 계획`.

### Principle

Do not store full memory text inside individual neurons.

```text
event
  -> EmoNet trace
  -> activation signature
  -> MemoryItem
  -> NeuralMemoryIndex binding
```

Later:

```text
new trace
  -> activation signature
  -> neural similarity recall
  -> merge with lexical recall
  -> context, relationship, plot, and gate pressure
```

### Work items

- Add `ActivationSignature`.
- Add `NeuralMemoryBinding`, recall scoring, cooldown, and fatigue suppression.
- Keep `MemoryStore` as the record source of truth.
- Bind memories by id to trace signatures.
- Merge neural recall with lexical recall.
- Persist neural-memory index separately.

### Exit criteria

- Similar emotional/neural patterns recall related memories even when lexical overlap is weak.
- Repeated recalls decay through cooldown or fatigue.
- Debug output explains why a memory was recalled.

## Phase 3 — Build character-specific internal life

### Ruca

- Main relational character.
- Maintains felt-state trace, memory recall, relationship interpretation, and private inner voice.
- Can misunderstand, imagine, hesitate, and revise interpretations over time.
- Does not send every internal movement to the user.

### Ricky

- Interpretation, balance, and structure pressure.
- Helps prevent Ruca from turning every uncertainty into an immediate external reaction.
- May surface for explicit analysis and structured explanation requests.

### Rocky

- Protective and action pressure.
- Pushes toward concrete action when urgency is high.
- May surface for urgent execution requests.

### Rookie

- Scene, plot, unresolved-thread, and narrative-pressure layer.
- Does not become a generic narrator by default.
- Tracks what remains unresolved and which scene transition is plausible.

### Work items

- Give each character a separate internal state record and memory view.
- Separate persistent traits from episode state.
- Add cross-character tension and alignment edges.
- Add private imagination records derived from trace + recalled memory, not direct hand-authored emotions.
- Add confidence and uncertainty so inner interpretations can be wrong.

### Exit criteria

- Characters can disagree internally without exposing all deliberation.
- Ruca can generate internal-only interpretations during no-reply time.
- Character surface speech remains controlled and sparse.

## Phase 4 — Expand Rookie into a real plot manager

### Work items

- Replace simple scene pressure with scene objects.
- Track unresolved threads, stakes, participants, relationship shifts, and scene transition candidates.
- Introduce plot beats without forcing them.
- Preserve narrative continuity across sessions.
- Add lightweight episode summaries for long-running stories.
- Distinguish ordinary chat mode from Rookie-managed story mode.

### Exit criteria

- Rookie can maintain a Zeta-style unfolding narrative without flattening characters into exposition.
- Scene transitions remain explainable in debug records.
- Ruca, Ricky, and Rocky can appear as characters inside Rookie-managed plots.

## Phase 5 — Add the autonomous scheduler

### Principle

No-reply time is an event stream, not an excuse to spam messages.

### Work items

- Add a local scheduler loop or service.
- Convert elapsed time into periodic `no_reply` events.
- Track last user message, last visible response, cooldown, unread state, and relationship pressure.
- Add message budget and quiet hours.
- Distinguish:
  - internal tick
  - imagination update
  - memory consolidation
  - scene pressure update
  - visible check-in candidate
- Let the response gate decide whether an external message is sent.

### Exit criteria

- Ruca may think and update internally without user input.
- Ruca may occasionally send a spontaneous message.
- Repeated ticks do not create message floods.

## Phase 6 — Connect the GUI

### Work items

- Connect the browser GUI to `RucaPipeline`.
- Show conversation normally.
- Keep internal state hidden by default.
- Add a developer panel for:
  - raw trace summary
  - semantic report
  - dominant branch
  - recalled memories
  - inner-voice shadows
  - plot state
  - relationship graph
  - response-gate decision
- Add controls for manual no-reply ticks and elapsed-time simulation.

### Exit criteria

- GUI and CLI execute the same runtime path.
- Developer inspection is available without leaking internal state into ordinary character dialogue.

## Phase 7 — Route LLM expression models and evaluate quality

The LLM is the expression layer, not the felt-state engine.

### Work items

- Keep OpenAI-compatible routing for `gpt-oss:120b-cloud`.
- Keep Anthropic routing for Claude Haiku expression experiments.
- Add provider-specific configuration files rather than hard-coding one model.
- Evaluate expression quality with trace-sensitive conditions.
- Compare at least:
  - no trace
  - stimulus-only
  - episode trace
  - targeted semantic-report conditioning
- Retain caution: broad benchmark superiority is not assumed merely because targeted episode fidelity improves.

### Evaluation axes

- appraisal fidelity
- raw affect preservation
- anti-softening
- action-tendency fit
- emotional specificity
- naturalness
- overall preference

### Exit criteria

- Provider routing is swappable without changing EmoNet state logic.
- Expression-model comparisons are reproducible.
- Targeted advantages and limitations are reported separately.

## Phase 8 — Storytelling runtime

### Work items

- Add story-mode sessions managed by Rookie.
- Add character cards and role constraints for Ruca, Ricky, Rocky, and future characters.
- Add location, time, relationship, and unresolved-event state.
- Add memory retrieval scoped by character and story arc.
- Preserve emotional asynchrony, pauses, body language, imperfect dialogue, and misunderstanding rather than over-explaining every feeling.

### Exit criteria

- The system can sustain a multi-character story over multiple sessions.
- Plot continuity, character memory, and emotional trace influence dialogue without direct prompt dumping.

## Verification strategy

Use focused tests before expanding scope.

### Baseline tests

- Event normalization.
- Short no-reply internal-only update.
- Long no-reply quiet check-in.
- Source text versus reference text separation.
- Session persistence.
- Trait drift persistence.
- Rookie unresolved-thread accumulation.
- Relationship graph accumulation.
- Visible speaker selection.
- CLI no-reply smoke.

### EmoNet tests

- Required artifacts fail loudly when absent.
- Trace output contains required fields.
- Time events alter trace through the event adapter.
- Dominant-branch extraction happens after full episode completion.
- Semantic report remains an interpretation layer, not a replacement for raw trace.

### Memory tests

- Neural binding persistence.
- Similarity recall.
- Cooldown and fatigue suppression.
- Lexical and neural recall merge behavior.
- Character-scoped recall.

### Scheduler tests

- Cooldown.
- Quiet hours.
- Message budget.
- Internal tick without visible output.
- No repeated check-in flood.

## Non-goals

- Do not manually script Ruca's emotion as a collection of if-statements.
- Do not expose raw private inner voices directly to ordinary users.
- Do not let Rookie narrate every interaction.
- Do not treat a single broad benchmark win as proof of universal superiority.
- Do not store full memory prose directly inside neurons.

## 미정/추가 논의 필요

- Exact EmoNet encoding for elapsed-time and no-reply events.
- Whether trait EMA updates should consume raw trace, semantic report, or both.
- Persistent versus episode memory split boundaries.
- Neural-memory signature dimensions and suppression schedule.
- Rookie story-mode activation rule.
- Quiet-hour defaults and message budget.
- GUI developer-panel visibility and export format.
- Final expression-model routing defaults for Ruca.
