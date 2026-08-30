"""
Loads an identity/*.yaml file (persona, system prompt, voice, language,
which playbooks to use). This is the "drop identity files in and it just
works" hook the pipeline is built around.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class Identity:
    name: str = "Assistant"
    system_prompt: str = "You are a helpful voice assistant. Keep answers short."
    voice: str = "en-IN-NeerjaNeural"
    language: Optional[str] = None
    playbooks: List[str] = field(default_factory=list)
    greeting: str = "Hello! How can I help you?"


def load_identity(path: Optional[str] = None) -> Identity:
    """
    Resolution order:
      1. Explicit `path` argument
      2. IDENTITY_FILE env var, if set (can be a path anywhere on disk —
         e.g. your own custom identity file outside the installed package)
      3. The package's own bundled identity/default.yaml — resolved
         relative to this FILE's installed location, not the current
         working directory. A plain relative string like "identity/
         default.yaml" only works if you happen to be running from the
         repo root; once this is pip-installed and run from anywhere else,
         that would silently fail and fall back to a blank Identity().
    """
    env_path = os.getenv("IDENTITY_FILE")
    if path:
        p = Path(path)
    elif env_path:
        p = Path(env_path)
    else:
        p = Path(__file__).resolve().parent.parent / "identity" / "default.yaml"

    if not p.exists():
        return Identity()

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return Identity(
        name=data.get("name", "Assistant"),
        system_prompt=data.get("system_prompt", "You are a helpful voice assistant.").strip(),
        voice=data.get("voice", "en-IN-NeerjaNeural"),
        language=(data.get("language") or None),
        playbooks=data.get("playbooks") or [],
        greeting=data.get("greeting", "Hello! How can I help you?"),
    )
