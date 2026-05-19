# EmoNet v5

`v5` is the character-chat MVP line. It uses the EmoNet runtime to turn an
input message into an internal trace, treats that trace as the character's
current emotion-state representation, and then asks an LLM to express that
state as character dialogue.

## Active Code

- `emonet/chat_service.py`: combines EmoNet trace, character context, session
  state, and the LLM call.
- `emonet/character.py`: character card, session memory, felt-state helpers,
  and response validation.
- `local_gui.py`: local browser GUI for character chat and AI dialogue tests.
- `data/characters/default_luca_like.json`: default character card.

## Default LLM

The local GUI defaults to the local Ollama OpenAI-compatible endpoint:

- provider: `openai_compatible`
- endpoint: `http://127.0.0.1:11434/v1`
- model: `gpt-oss:20b`
- API key: normally empty

You can override the defaults with environment variables:

```powershell
$env:EMONET_LLM_PROVIDER = "openai_compatible"
$env:EMONET_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
$env:EMONET_LLM_MODEL = "gpt-oss:20b"
$env:EMONET_LLM_API_KEY = ""
```

Check local Ollama models:

```powershell
ollama list
```

## Run

From the repository root, with the Codex bundled Python:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v5
& $PY -m unittest discover -s tests -v
& $PY .\local_gui.py
```

The GUI opens at:

```text
http://127.0.0.1:8788/
```

## Notes

`v5` does not train a new LLM. The LLM is only the final expression layer. The
EmoNet trace, character card, session state, and relationship memory are built
locally and passed into the generation prompt.
