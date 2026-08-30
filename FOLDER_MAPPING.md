# Folder mapping — pre-PyPI-restructuring → current

Reference for anyone (human or AI tool) working from memory of the old
layout. Everything now lives inside one importable package, `maya_voice_os/`.

## Folder-level

| Old path | New path |
|---|---|
| `asr-service/` | `maya_voice_os/asr_service/` |
| `llm-service/` | `maya_voice_os/llm_service/` |
| `tts-service/` | `maya_voice_os/tts_service/` |
| `orchestration-service/` | `maya_voice_os/orchestration_service/` |
| `telephony-service/` | `maya_voice_os/telephony_service/` |
| `shared/` | `maya_voice_os/shared/` |
| `identity/` | `maya_voice_os/identity/` |
| `playbooks/` | `maya_voice_os/playbooks/` |

## Entry points

| Old command | New command |
|---|---|
| `python run_local.py` | `maya-voice-os local` |
| `python run_orchestrator.py` | `maya-voice-os orchestrator` |
| `python run_telephony.py` | `maya-voice-os telephony` |

`run_orchestrator.py` / `run_telephony.py` no longer exist as separate
files — both are now handled by `maya_voice_os/cli.py`, which is what
`pyproject.toml`'s `[project.scripts]` wires the `maya-voice-os` command to.
`run_local.py` still exists as its own file (inside the package now), since
`cli.py` just calls into it.

## Config

| Old | New |
|---|---|
| `requirements.txt` | `pyproject.toml` → `[project]` → `dependencies = [...]` |
| N/A | `pyproject.toml` → `[project.scripts]` (console entry point) |
| N/A | `pyproject.toml` → `[tool.setuptools.package-data]` (bundles playbooks/identity/UI into the installed package) |

## Removed (no longer needed)

| File | Why |
|---|---|
| `shared/module_loader.py` | Existed only to work around hyphenated folder names blocking real Python imports. Now that folders are underscored and nested under one package, every cross-module import is a normal `from maya_voice_os.x.y import z` — no dynamic file-path loading needed anywhere. |
| `telephony_service/sip_reference/` (from the pip package specifically) | Still in the GitHub source tree, just not bundled into what `pip install` ships — it was always a non-functional reference sample, never meant to run. |

## Import pattern, if you're editing code

Every cross-module reference now looks like this, everywhere:
```python
from maya_voice_os.llm_service.llm_router import LLMRouter
from maya_voice_os.shared.audio_utils import resample_audio
```
Not this (old pattern, gone):
```python
_mod = load_module(ROOT / "llm-service" / "llm_router.py", "llm_router")
LLMRouter = _mod.LLMRouter
```
