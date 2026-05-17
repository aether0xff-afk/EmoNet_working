from __future__ import annotations

# Public v4 CLI facade.
# Keep the large legacy implementation isolated while exposing extracted modules here.

from .legacy_cli import *  # noqa: F401,F403
from .legacy_cli import _init_parallel_model, _require_parallel_model  # noqa: F401
from .legacy_cli import main as _legacy_main
from .llm_api import (  # noqa: F401
    call_openai_compatible_chat,
    extract_json_block,
    request_json_response,
    request_plain_text_response,
)
from .episode_conditioning import (  # noqa: F401
    augment_profile_with_episode,
    build_episode_generation_prompt,
    build_episode_lines,
    build_episode_summary_text,
    build_episode_v3_generation_prompt,
    build_episode_v3_lines,
    build_hybrid_episode_generation_prompt,
    load_episode_payload,
    resolve_episode_payload_path,
)


def main() -> None:
    _legacy_main()


if __name__ == "__main__":
    main()
