"""Structured console and JSONL logging for EmoNet experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class RunLogger:
    """Emit readable console logs and append matching JSONL records."""

    def __init__(self, *, output_dir: str | Path, verbose: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "run_log.jsonl"
        self.verbose = bool(verbose)
        self._sequence = 0

    def log(self, event: str, message: str, **fields: Any) -> None:
        self._sequence += 1
        record = {
            "sequence": self._sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "message": message,
            **fields,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if self.verbose:
            field_text = " ".join(
                f"{key}={json.dumps(value, ensure_ascii=False, default=str)}"
                for key, value in fields.items()
            )
            suffix = f" | {field_text}" if field_text else ""
            print(f"[EmoNet][{self._sequence:03d}][{event}] {message}{suffix}", flush=True)

    def section(self, title: str) -> None:
        self.log("section", f"=== {title} ===")
