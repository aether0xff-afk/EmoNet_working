# EmoNet v6

`v6` is the Ruca & Rookie autonomous character runtime line. It extends the
`v5` character-chat MVP toward characters that keep an internal emotional life
across time: user messages, silence ticks, internal voices, response gating,
relationship memory, and optional EmoNet trace conditioning.

```text
user message or silence
  -> scheduled Ruca event
  -> emotion tick
  -> memory and relationship lookup
  -> internal voices
  -> response gate
  -> Ruca message, quiet check-in, or internal-only state update
```

The LLM is the required expression layer whenever Ruca is scheduled to speak.
EmoNet, memory, relationship state, and scene state decide what Ruca is carrying
before the LLM turns it into dialogue.

## Current v6 Slice

- `ruca_engine/event_scheduler.py`: normalizes user messages, short silence, and long silence into Ruca events.
- `ruca_engine/pipeline.py`: runs one full Ruca event and records debug state.
- `ruca_engine/emotion.py`: rule-based emotional trace update when EmoNet is not requested.
- `ruca_engine/memory.py`: short-term and relationship memory persistence.
- `ruca_engine/inner_voice.py`: Ruca/Ricky/Rocky private candidate voices.
- `ruca_engine/spontaneous.py`: response gate for check-ins and silence.
- `local_gui.py`: local browser GUI for character chat and AI dialogue tests.

## Run

From this directory:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $PY -m unittest discover -s tests -v
```

One normal turn:

```powershell
& $PY -m ruca_engine.cli "v6를 더 개발해줘" --llm --debug
```

One internal-only silence tick:

```powershell
& $PY -m ruca_engine.cli --silence --elapsed-minutes 10 --debug
```

A long silence that may produce a quiet check-in:

```powershell
& $PY -m ruca_engine.cli --elapsed-minutes 60 --llm --debug
```

Persistent memory/session:

```powershell
& $PY -m ruca_engine.cli "지금 너무 불안하고 무서워" --llm --memory .\outputs\ruca_memory.json --session .\outputs\ruca_session.json --debug
& $PY -m ruca_engine.cli --elapsed-minutes 60 --llm --memory .\outputs\ruca_memory.json --session .\outputs\ruca_session.json --debug
```

## LLM Expression Layer

When Ruca is scheduled to speak, `--llm` is required. If the LLM call fails, the
turn fails loudly instead of fabricating a rule-based reply. Supported providers
are `openai_compatible` and `anthropic`.

```powershell
$env:OPENAI_API_KEY = "..."
& $PY -m ruca_engine.cli "Ruca처럼 짧게 답해줘" --llm --debug
```

For local Ollama/OpenAI-compatible servers:

```powershell
& $PY -m ruca_engine.cli "Ruca처럼 짧게 답해줘" --llm --base-url http://127.0.0.1:11434/v1 --model-name gpt-oss:120b-cloud --debug
```

## Notes

The repository keeps optional research tooling for figures and ridge encoding.
Runtime paths should fail explicitly when required model, perception, plotting,
or trace dependencies are missing.
