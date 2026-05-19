# EmoNet Working Tree

EmoNet is a research and prototype workspace for emotion-state modeling.
The repository keeps several historical implementation lines side by side, so
the root is mainly an index. Run code and tests inside the version directory
you are working on.

## Current Focus

- `v6`: Ruca & Rookie autonomous character runtime line, starting from v5 and
  adding no-reply ticks, internal voice, spontaneous response gates, and
  Rookie-ready scene/story state.
- `v4`: active research, evaluation, and local GUI line.
- `v5`: character-chat MVP built on top of the v4 runtime and the v3.1
  trace-as-emotion idea.
- `v3.1`: representation-level experiments for the hypothesis that a neural
  trace is the emotion-state representation itself.

Older directories are kept for continuity:

- `v1`: initial `emotion_z_pipeline` and GUI experiments.
- `v2`: early modular PyTorch MVP.
- `v3`: legacy self-contained research/CLI line.

## Directory Map

```text
.
  v1/                  early emotion-z pipeline
  v2/                  modular MVP
  v3/                  legacy research line
  v3.1/                trace-as-emotion experiments
  v4/                  active research/evaluation app line
  v5/                  character-chat MVP line
  v6/                  Ruca/Rookie autonomous character runtime line
  Dataset/             shared source dataset
  blueprints/          design notes and older architecture sketches
  encoder-LLM-testing/ LLM labeling benchmarks
  encoder-ML testing/  ML encoder benchmark material
  output/, outputs/    generated figures and experiment outputs
  tmp/                 temporary document/poster build material
```

## Python Environment

On this machine, the `python` command may resolve to the Windows Store alias
instead of a real interpreter. Use the Codex bundled Python directly:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $PY --version
& $PY -m pip install -r requirements.txt
```

The shared root requirements cover the common runtime:

```text
joblib, matplotlib, networkx, numpy, pandas, scikit-learn, streamlit, torch
```

For a normal local setup outside Codex, create a virtual environment and install
the same root requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick Start

Run v4 tests:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v4
& $PY -m unittest discover -s tests -v
```

Run v5 tests:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v5
& $PY -m unittest discover -s tests -v
```

Run v6 tests:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v6
& $PY -m unittest discover -s tests -v
```

Start the v4 local GUI:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v4
& $PY .\local_gui.py
```

Start the v5 character-chat GUI:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v5
& $PY .\local_gui.py
```

Start the v6 Ruca/Rookie GUI:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v6
& $PY .\local_gui.py
```

## API Keys

`v5` and `v6` use the local Ollama OpenAI-compatible endpoint by default and normally
does not need an API key. `v4` still has Claude-oriented GUI paths; set
`ANTHROPIC_API_KEY` only when running those Claude-backed flows.

Do not commit API keys, local progress files, or generated scratch outputs.

## Git Hygiene

This repository intentionally contains selected research artifacts, but new
bulk outputs should not be added casually. The `.gitignore` blocks common
generated directories and large model/archive files for future work. Existing
tracked artifacts remain tracked until they are intentionally moved or removed
in a separate cleanup.
