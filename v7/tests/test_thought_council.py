import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime import ThoughtCouncil


class ThoughtCouncilTests(unittest.TestCase):
    def test_council_runs_multiple_emotion_models(self) -> None:
        council = ThoughtCouncil()

        lines = council.tick(event_kind="user_message", text="생각들이 서로 대화하게 해줘")

        self.assertEqual(len(council.agents), 4)
        self.assertGreaterEqual(len(lines), 2)
        self.assertEqual(len(council.snapshot_records()), 4)
        self.assertTrue(all(item["tick_index"] == 0 for item in council.snapshot_records()))

    def test_council_keeps_internal_dialogue_history(self) -> None:
        council = ThoughtCouncil(max_lines=8)

        for _ in range(12):
            council.tick(event_kind="silence_tick", elapsed_seconds=1.0)

        records = council.to_records(limit=20)
        self.assertLessEqual(len(records), 8)
        self.assertTrue(all("speaker_name" in item for item in records))


if __name__ == "__main__":
    unittest.main()
