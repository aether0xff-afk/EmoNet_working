# v6 Complete Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full v6 Ruca/Rookie runtime from the integrated design: autonomous Ruca, trait drift, Rookie plot state, character memory separation, relationship graph, controlled multi-character surface speech, and CLI access.

**Architecture:** Keep `RucaPipeline` as the orchestration boundary and add small state modules that return serializable dataclasses. The LLM remains only the final expression layer; local runtime state decides emotion, memory, plot pressure, response gating, and visible speaker selection before any LLM call.

**Tech Stack:** Python standard library dataclasses/JSON, `unittest`, and the existing `v6/ruca_engine` package. GUI integration is explicitly out of scope for this pass.

---

## File Structure

- Create `v6/ruca_engine/trait_state.py`: character trait EMA state, event deltas, and persistence helpers.
- Create `v6/ruca_engine/plot_manager.py`: Rookie scene state, unresolved threads, plot pressure, and next-scene hints.
- Create `v6/ruca_engine/relationship_graph.py`: typed relationship edges between user/Ruca/Rookie/Ricky/Rocky.
- Create `v6/ruca_engine/character_runtime.py`: speaker selection and conflict resolution for Ruca/Ricky/Rocky direct speech.
- Modify `v6/ruca_engine/models.py`: add serializable runtime dataclasses used by the new modules and `TurnResult`.
- Modify `v6/ruca_engine/session.py`: persist trait state, plot state, relationship graph, and visible speaker history.
- Modify `v6/ruca_engine/pipeline.py`: run the new modules each turn and expose records in debug output.
- Modify `v6/ruca_engine/prompt_builder.py`: include plot, traits, relationships, and visible speaker instructions.
- Modify `v6/ruca_engine/cli.py`: support `--event-type`, `--elapsed-minutes`, and full v6 event execution.
- Modify `v6/README.md`: document the completed v6 runtime.
- Test in `v6/tests/test_ruca_engine.py`: focused unit and integration tests for all runtime additions.

---

### Task 1: Trait EMA State

**Files:**
- Create: `v6/ruca_engine/trait_state.py`
- Modify: `v6/ruca_engine/models.py`
- Modify: `v6/ruca_engine/session.py`
- Test: `v6/tests/test_ruca_engine.py`

- [ ] **Step 1: Write the failing test**

Add tests that import `CharacterTraitState`, run one warm turn and one alarm turn, then assert Ruca/Rocky trait values move and persist through `SessionStore`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest v6.tests.test_ruca_engine.RucaEngineTests.test_trait_state_updates_and_persists -v`
Expected: FAIL because `CharacterTraitState` is not defined.

- [ ] **Step 3: Write minimal implementation**

Implement `CharacterTraitState.from_profiles()`, `update_trait_state()`, `to_record()`, `from_mapping()`, and session persistence fields.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest v6.tests.test_ruca_engine.RucaEngineTests.test_trait_state_updates_and_persists -v`
Expected: PASS.

### Task 2: Rookie Plot Manager

**Files:**
- Create: `v6/ruca_engine/plot_manager.py`
- Modify: `v6/ruca_engine/models.py`
- Modify: `v6/ruca_engine/session.py`
- Modify: `v6/ruca_engine/pipeline.py`
- Test: `v6/tests/test_ruca_engine.py`

- [ ] **Step 1: Write the failing test**

Add tests that repeated implementation requests add an unresolved thread and no-reply events increase plot pressure without forcing a direct reply.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest v6.tests.test_ruca_engine.RucaEngineTests.test_rookie_plot_state_tracks_threads_and_pressure -v`
Expected: FAIL because `plot_state` is not exposed.

- [ ] **Step 3: Write minimal implementation**

Implement `RookiePlotState`, `update_plot_state()`, session persistence, and debug output.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest v6.tests.test_ruca_engine.RucaEngineTests.test_rookie_plot_state_tracks_threads_and_pressure -v`
Expected: PASS.

### Task 3: Relationship Graph

**Files:**
- Create: `v6/ruca_engine/relationship_graph.py`
- Modify: `v6/ruca_engine/models.py`
- Modify: `v6/ruca_engine/session.py`
- Modify: `v6/ruca_engine/pipeline.py`
- Test: `v6/tests/test_ruca_engine.py`

- [ ] **Step 1: Write the failing test**

Add tests that warm input raises user-Ruca trust, alarm/no-reply raises reassurance need, and Ruca-Rocky tension rises during protective events.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest v6.tests.test_ruca_engine.RucaEngineTests.test_relationship_graph_accumulates_edges -v`
Expected: FAIL because `relationship_graph` is not exposed.

- [ ] **Step 3: Write minimal implementation**

Implement `RelationshipGraph`, `RelationshipEdge`, `update_relationship_graph()`, session persistence, and debug output.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest v6.tests.test_ruca_engine.RucaEngineTests.test_relationship_graph_accumulates_edges -v`
Expected: PASS.

### Task 4: Multi-character Runtime

**Files:**
- Create: `v6/ruca_engine/character_runtime.py`
- Modify: `v6/ruca_engine/models.py`
- Modify: `v6/ruca_engine/pipeline.py`
- Modify: `v6/ruca_engine/composer.py`
- Modify: `v6/ruca_engine/prompt_builder.py`
- Test: `v6/tests/test_ruca_engine.py`

- [ ] **Step 1: Write the failing test**

Add tests that default visible speaker remains Ruca, Ricky can surface for analysis requests, Rocky can surface for urgent action, and internal-only no-reply has no visible speaker.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest v6.tests.test_ruca_engine.RucaEngineTests.test_character_runtime_selects_visible_speaker -v`
Expected: FAIL because `visible_speaker` is not exposed.

- [ ] **Step 3: Write minimal implementation**

Implement `VisibleSpeakerDecision`, `select_visible_speaker()`, composer prefix handling for non-Ruca speakers, prompt instructions, and debug output.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest v6.tests.test_ruca_engine.RucaEngineTests.test_character_runtime_selects_visible_speaker -v`
Expected: PASS.

### Task 5: CLI v6 Integration

**Files:**
- Modify: `v6/ruca_engine/cli.py`
- Modify: `v6/README.md`
- Test: `v6/tests/test_ruca_engine.py`

- [ ] **Step 1: Write the failing tests**

Add tests for CLI event execution, no-reply session semantics, and debug output using v6 session records. Do not add or modify GUI tests in this task.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest v6.tests.test_ruca_engine.RucaEngineTests.test_cli_can_run_no_reply_event -v`
Expected: FAIL because CLI no-reply support is not implemented.

- [ ] **Step 3: Write minimal implementation**

Add CLI event options, persist v6 memory/session paths, and expose debug records. Leave `v6/ruca_gui.py` outside the v6 runtime integration scope.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest v6.tests.test_ruca_engine -v`
Expected: PASS.

### Task 6: Full Verification and Docs

**Files:**
- Modify: `v6/README.md`
- Test: all v6 tests

- [ ] **Step 1: Run full v6 test suite**

Run: `python -m unittest discover -s v6/tests -v`
Expected: PASS with zero failures.

- [ ] **Step 2: Run CLI smoke tests**

Run:
`python -m v6.ruca_engine.cli "나 지금 너무 불안하고 무서워" --debug`
`python -m v6.ruca_engine.cli --event-type no_reply --elapsed-minutes 180 --debug`
Expected: both exit 0 and print debug JSON.

- [ ] **Step 3: Review git diff**

Run: `git diff -- v6 docs/superpowers/plans/2026-05-22-v6-complete-runtime.md`
Expected: only planned v6/runtime docs changes appear, with no `v6/ruca_gui.py` integration diff.

---

## Self-Review

- Spec coverage: autonomous response gate, trait drift, Rookie plot layer, character memory separation through memory records, relationship graph, controlled multi-character surface speech, CLI access, and documentation are covered.
- Placeholder scan: no unresolved placeholders are intentionally left in the plan.
- Type consistency: new runtime records are dataclass-style objects with `to_record()`/`from_mapping()` patterns matching existing v6 code.
