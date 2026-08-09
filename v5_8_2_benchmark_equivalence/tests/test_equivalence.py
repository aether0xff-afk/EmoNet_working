from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VERSION_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "v5_2_learned_memory" / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "v5_6_1_readout_temporal_diagnostic" / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "v5_7_residual_fast_dynamics"))
sys.path.insert(0, str(REPO_ROOT / "v5_8_adaptive_fast_dynamics"))

from run_benchmark_equivalence import PAIR_COUNT, TRAIN_PAIRS, new_case, old_case


def transient(sequence):
    return tuple(sequence[1:5])


def test_old_and_new_renderers_have_identical_latent_abab_vs_aabb_logic() -> None:
    for pair_id in (0, TRAIN_PAIRS - 1, TRAIN_PAIRS, PAIR_COUNT - 1):
        for factory in (old_case, new_case):
            c0, c1, current, event_a, event_b, _ = factory(pair_id)
            assert transient(c0) == (event_a, event_b, event_a, event_b)
            assert transient(c1) == (event_a, event_a, event_b, event_b)
            assert Counter(transient(c0)) == Counter(transient(c1))
            assert current == "The identical current observation is now presented."


def test_train_test_pair_ids_are_disjoint() -> None:
    assert set(range(TRAIN_PAIRS)).isdisjoint(set(range(TRAIN_PAIRS, PAIR_COUNT)))


def test_renderer_difference_is_surface_only_not_label_specific() -> None:
    for pair_id in (3, 83):
        oc0, oc1, _, _, _, _ = old_case(pair_id)
        nc0, nc1, _, _, _, _ = new_case(pair_id)
        assert len(oc0) == len(nc0) == 6
        assert len(oc1) == len(nc1) == 6
        assert oc0[-1] == nc0[-1]
        assert oc1[-1] == nc1[-1]
        assert Counter(transient(oc0)) != Counter(transient(nc0))
        # Within each renderer, however, the two labels remain multiset matched.
        assert Counter(transient(oc0)) == Counter(transient(oc1))
        assert Counter(transient(nc0)) == Counter(transient(nc1))
