from __future__ import annotations

import json
from pathlib import Path

from .models import CharacterProfile


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_profiles_path() -> Path:
    return package_root() / "data" / "characters" / "ruca_rookie_profiles.json"


def load_character_profiles(path: Path | None = None) -> dict[str, CharacterProfile]:
    active_path = path or default_profiles_path()
    payload = json.loads(active_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"character profile file must contain a list: {active_path}")
    profiles = [CharacterProfile.from_mapping(item) for item in payload if isinstance(item, dict)]
    profile_map = {profile.character_id: profile for profile in profiles}
    required = {"ruca", "rookie", "ricky", "rocky"}
    missing = sorted(required.difference(profile_map))
    if missing:
        raise ValueError(f"missing required character profiles: {', '.join(missing)}")
    return profile_map
