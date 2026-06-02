from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ruca_engine import RucaPipeline, SessionStore, make_event
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
        self.assertEqual(result.debug_record["event"]["source_text"], "")
        self.assertTrue(result.debug_record["event"]["reference_text"])

    def test_long_no_reply_can_send_check_in(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        pipeline.run_turn("Thank you. I will be back later.")
        result = pipeline.run_event(event_type="no_reply", elapsed_minutes=180)
        self.assertEqual(result.response_decision.action, "send_message")
        self.assertEqual(result.spontaneous_reaction.reaction_type, "quiet_check_in")
        self.assertTrue(result.assistant_text)

    def test_short_silence_alias_is_internal_only(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        result = pipeline.run_turn("", elapsed_minutes=10, force_silence=True)
        self.assertEqual(result.event.event_type, "silence_tick")
        self.assertEqual(result.response_decision.action, "update_internal_only")
        self.assertEqual(result.assistant_text, "")

    def test_visible_speaker_selection(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        default = pipeline.run_turn("Hello there.")
        analysis = pipeline.run_turn("What is the structure?")
        action = pipeline.run_turn("Build it now!")
        self.assertEqual(default.visible_speaker.character_id, "ruca")
        self.assertEqual(analysis.visible_speaker.character_id, "ricky")
        self.assertEqual(action.visible_speaker.character_id, "rocky")

    def test_session_persists_visible_speaker_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.json"
            pipeline = RucaPipeline(memory_store=MemoryStore.from_items(), session_store=SessionStore(path))
            pipeline.run_turn("What is the structure?")
            pipeline.run_event(event_type="no_reply", elapsed_minutes=45)
            stored = SessionStore(path).load()
        self.assertEqual(stored.turn_index, 2)
        self.assertEqual(stored.recent_history[0]["visible_speaker"], "ricky")
        self.assertIsNone(stored.recent_history[1]["visible_speaker"])
        self.assertIn("trait_state", stored.to_record())
        self.assertIn("plot_state", stored.to_record())
        self.assertIn("relationship_graph", stored.to_record())


if __name__ == "__main__":
    unittest.main()
