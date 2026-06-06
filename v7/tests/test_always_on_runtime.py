import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime import AlwaysOnEmoNetRuntime, RuntimeEvent


class AlwaysOnRuntimeTests(unittest.TestCase):
    def test_runtime_keeps_neuron_state_between_ticks(self) -> None:
        runtime = AlwaysOnEmoNetRuntime(neuron_count=8, remember_threshold=0.05)

        first = runtime.tick(RuntimeEvent(kind="user_message", text="구조를 계속 생각하고 있어"))
        first_activation = first.neurons[0].activation

        second = runtime.tick(RuntimeEvent(kind="typing", elapsed_seconds=1.0))

        self.assertGreater(second.tick_index, first.tick_index)
        self.assertNotEqual(second.neurons[0].activation, 0.0)
        self.assertNotEqual(second.neurons[0].activation, first_activation)

    def test_typing_state_is_environmental_stimulation(self) -> None:
        runtime = AlwaysOnEmoNetRuntime(neuron_count=8)

        snapshot = runtime.tick(RuntimeEvent(kind="typing", elapsed_seconds=2.0))

        self.assertEqual(snapshot.event_kind, "typing")
        self.assertEqual(len(snapshot.stim_vec), 8)
        self.assertGreater(sum(abs(value) for value in snapshot.stim_vec), 0.0)

    def test_neuron_stores_local_memory_when_k_crosses_threshold(self) -> None:
        runtime = AlwaysOnEmoNetRuntime(neuron_count=8, remember_threshold=0.01)

        snapshot = runtime.tick(RuntimeEvent(kind="user_message", text="반드시 기억해야 하는 설계"))

        remembered = [memory for neuron in snapshot.neurons for memory in neuron.local_memory]
        self.assertTrue(remembered)
        self.assertIn("user_message", remembered[0]["event_kind"])
        self.assertEqual(len(remembered[0]["stim_vec"]), 8)

    def test_cluster_summary_is_built_from_live_neuron_activity(self) -> None:
        runtime = AlwaysOnEmoNetRuntime(neuron_count=12, cluster_count=3)

        snapshot = runtime.tick(RuntimeEvent(kind="user_message", text="클러스터와 기억 시스템"))

        self.assertEqual(len(snapshot.clusters), 3)
        self.assertEqual(sum(cluster.size for cluster in snapshot.clusters), 12)
        self.assertIsNotNone(snapshot.dominant_cluster_id)


if __name__ == "__main__":
    unittest.main()
