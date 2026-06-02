from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ruca_engine import RucaPipeline, make_event
from ruca_engine.memory import MemoryStore


class IntegratedRuntimeBaselineTests(unittest.TestCase):
    def test_no_reply_event_has_empty_source(self) -> None:
        event = make_event(event_type="no_reply", text="context", elapsed_minutes=45)
        self.assertEqual(event.event_type, "no_reply")
        self.assertEqual(event.user_text, "")
        self.assertEqual(event.elapsed_minutes, 45.0)

    def test_short_no_reply_is_internal_only(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        pipeline.run_turn("Thanks. I will return later.")
        result = pipeline.run_event(event_type="no_reply", elapsed_minutes=45)
        self.assertEqual(result.response_decision.action, "update_internal_only")
        self.assertEqual(result.assistant_text, "")
        self.assertIsNone(result.visible_speaker)


if __name__ == "__main__":
    unittest.main()
