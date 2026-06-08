"""Environment-based configuration for LM Studio experiment endpoints."""

from __future__ import annotations

import os


ENV_NAME = "EMONET_LMSTUDIO_BASE_URL"


def load_default_lmstudio_base_url() -> str | None:
    """Return a configured LM Studio endpoint without embedding it in the repo."""

    value = os.environ.get(ENV_NAME, "").strip()
    return value.rstrip("/") or None
