# EmoNet v6 — Ruca/Rookie autonomous runtime

`v6` is the executable integration line for the Ruca/Rookie character runtime.

The target is not a rule-scripted chatbot. EmoNet owns the felt-state trace. The runtime only provides events, elapsed time, memory cues, relationship context, and safe delivery gates. Inner voices are a language shadow of the trace: they may explain or surface pressure, but they must not become the component that decides emotion.

## Current executable slice

```text
user_message or no_reply event
  -> EmoNet trace when --emonet is enabled
     (development rule emotion fallback only when EmoNet is not enabled,
      or when --allow-rule-emotion-fallback is explicitly requested)
  -> memory retrieval
  -> Rookie turn context and plot pressure
  -> Ruca / Ricky / Rocky inner-voice shadows
  -> slow trait EMA and relationship graph updates
  -> response gate
  -> visible speaker selection
  -> LLM expression layer or development rule composer
  -> persistent memory and session state
```

The default visible speaker is Ruca. Ricky can surface for analysis and structure requests. Rocky can surface for urgent execution requests. Rookie manages scene and plot pressure rather than speaking directly by default.

## Event model

`no_reply` is the standard autonomous event. It preserves the distinction between:

- `source_text`: the new user text for this event, which is empty during no-reply time
- `reference_text`: the last useful user text carried only as context
- `elapsed_minutes`: time since the last user message
- `response_decision`: `send_message`, `update_internal_only`, or `stay_silent`

Legacy `silence_tick` and `long_silence` events remain available as compatibility aliases while the GUI and scheduler migrate to `no_reply`.

## Package layout

- `ruca_engine/pipeline.py`: full event orchestration boundary
- `ruca_engine/event_scheduler.py`: user-message, no-reply, and legacy silence normalization
- `ruca_engine/emonet_adapter.py`: v5 EmoNet trace runtime bridge using v6 artifacts
- `ruca_engine/emotion.py`: development fallback and no-reply signal scaffolding
- `ruca_engine/memory.py`: short-term, long-term, relationship, and emotional memory records
- `ruca_engine/context.py`: Rookie turn context
- `ruca_engine/inner_voice.py`: Ruca/Ricky/Rocky language shadows
- `ruca_engine/spontaneous.py`: spontaneous reaction candidate
- `ruca_engine/response_gate.py`: visible-message versus internal-only decision
- `ruca_engine/trait_state.py`: slow character trait EMA
- `ruca_engine/plot_manager.py`: Rookie scene pressure and unresolved threads
- `ruca_engine/relationship_graph.py`: typed relationship edges
- `ruca_engine/character_runtime.py`: controlled visible-speaker selection
- `ruca_engine/session.py`: persistent runtime state
- `ruca_engine/prompt_builder.py`: LLM expression prompt
- `ruca_engine/cli.py`: one-event CLI

## Run tests

From the repository root:

```powershell
python -m unittest discover -s v6/tests -v
```

## CLI smoke runs

Normal user message:

```powershell
cd .\v6
python -m ruca_engine.cli "실제로 구현하려면 어떻게 해야 할지 알려줘" --debug
```

Internal-only no-reply tick:

```powershell
python -m ruca_engine.cli --event-type no_reply --elapsed-minutes 45 --debug
```

Longer no-reply event that may produce a low-pressure check-in:

```powershell
python -m ruca_engine.cli --event-type no_reply --elapsed-minutes 180 --debug
```

Persistent state:

```powershell
python -m ruca_engine.cli "나 지금 너무 불안하고 무서워" `
  --memory .\outputs\ruca_memory.json `
  --session .\outputs\ruca_session.json `
  --debug

python -m ruca_engine.cli --event-type no_reply --elapsed-minutes 180 `
  --memory .\outputs\ruca_memory.json `
  --session .\outputs\ruca_session.json `
  --debug
```

## EmoNet-authoritative mode

Use `--emonet` to make the EmoNet adapter the visible runtime emotion source.

```powershell
python -m ruca_engine.cli "지금 너무 복잡해" --emonet --debug
```

If required artifacts or dependencies are missing, `--emonet` fails loudly. The explicit `--allow-rule-emotion-fallback` flag exists only for development diagnostics.

## LLM expression layer

The LLM is the final wording layer, not the emotion engine.

OpenAI-compatible endpoint:

```powershell
python -m ruca_engine.cli "짧게 답해줘" `
  --llm `
  --base-url http://127.0.0.1:11434/v1 `
  --model-name gpt-oss:120b-cloud `
  --debug
```

Anthropic endpoint:

```powershell
$env:ANTHROPIC_API_KEY = "..."
python -m ruca_engine.cli "짧게 답해줘" `
  --llm `
  --provider anthropic `
  --base-url https://api.anthropic.com `
  --api-key-env ANTHROPIC_API_KEY `
  --model-name claude-haiku-4-5-20251001 `
  --debug
```

## Next implementation line

The full integration roadmap is tracked in `docs/plans/2026-06-02-ruca-rookie-integrated-runtime-roadmap.md`.
