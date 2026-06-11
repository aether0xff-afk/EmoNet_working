# Adaptive Thought Module Protocol

Status date: 2026-06-11

This document defines the first multi-module thought protocol for EmoNet v7.
It is a design contract for integration work, not a claim that multi-module
reasoning has already been validated.

## Purpose

Each thought module is modeled as:

```text
module_id
+ private EmoNet state
+ local chat client / model configuration
+ local memory summary
+ participation budget
+ message inbox/outbox
```

The LLM text is a language shadow of the module state. It may propose an
internal thought or candidate response, but it must not become the source of
emotional ground truth.

## Message Types

The protocol uses the existing v7 event categories:

```text
user_message
internal_thought
module_message
elapsed_time
```

Minimum message envelope:

```text
message_id: str
kind: user_message | internal_thought | module_message | elapsed_time
source_module_id: str | null
target_module_id: str | null
round_index: int
text: str
state_report: dict | null
created_from_event_id: str | null
```

`text` is natural language. `state_report` is a neutral numeric report from the
module's own trace state. Message envelopes do not contain emotion labels.

## Module State

Minimum module state:

```text
module_id: str
role_hint: str
emonet_state: SNNState or MemoryThresholdSNNState
last_trace_report: dict
local_memory_summary: str
participation_budget_remaining: int
status: active | quiet | saturated | retired
```

`role_hint` is a prompt prior, not a permanent identity. Examples:

```text
module_planner: checks sequence, constraints, and next action
module_skeptic: checks uncertainty, missing evidence, and risk
```

## Two-Module Minimum Interface

A minimal discussion uses two modules and one shared round buffer:

```text
start user_event
-> broadcast user_message to module_planner and module_skeptic
-> each module runs its private EmoNet event window
-> each module builds a neutral state report
-> each module emits one internal_thought
-> modules exchange module_message summaries
-> each module updates private state from received module_message
-> modules emit candidate answer fragments
-> shared termination rule selects output or silence
```

No central emotional summarizer is introduced. A coordinator may enforce turn
limits and output formatting, but it should not assign hidden emotion labels or
overwrite module states.

## Discussion Rounds

One round contains:

```text
round_index
input_messages
per_module_state_reports
per_module_internal_thoughts
per_module_candidate_outputs
termination_vote
```

Allowed termination votes:

```text
answer_ready
needs_one_more_round
stay_silent
blocked_by_missing_context
```

Stop when the first rule matches:

1. all active modules vote `answer_ready`,
2. any module votes `stay_silent` and no module has an urgent action proposal,
3. max rounds is reached,
4. total token or participation budget is exhausted,
5. every active module is `saturated` or `quiet`.

## State Transition Example

```text
user_message: "친구가 답장을 하지 않았다."

module_planner
  receives user_message
  updates private SNN state
  report: active_ratio=0.08, trace_persistence=0.31
  thought: "가능한 설명을 나누어 보고, 바로 단정하지 않는 답이 필요하다."

module_skeptic
  receives user_message
  updates private SNN state
  report: active_ratio=0.11, trace_persistence=0.42
  thought: "정보가 부족하므로 관계 단절로 결론내리면 안 된다."

exchange
  module_planner -> module_skeptic: "상황적 가능성을 먼저 제시하자."
  module_skeptic -> module_planner: "확신 표현은 낮추고 확인 질문을 넣자."

candidate output
  planner: "바빠서 못 봤을 수도 있으니 조금 기다려 보자."
  skeptic: "단정하지 말고, 필요하면 짧게 확인해 보는 편이 낫다."

termination
  both answer_ready
  visible response can be composed from agreed candidate fragments
```

## Fixed Versus Learnable Elements

Fixed in the first integration prototype:

- message envelope fields
- neutral state-report requirement
- max-round and budget enforcement
- no central emotion-label aggregator
- module-private EmoNet state
- LLM as expression/thought layer only

Learnable or later-adaptive:

- module role specialization
- participation policy
- module creation and retirement policy
- state-sharing gates
- candidate ranking
- whether module disagreement improves downstream metrics

## Shared State Boundary

Modules may share:

- natural-language `module_message` summaries,
- explicit candidate output fragments,
- neutral numeric reports when attached to a message.

Modules must not share:

- raw private hidden tensors by default,
- optimizer state,
- private memory buffers,
- synthetic emotion labels created by another module.

Future experiments may add a learned state-sharing gate, but that gate must be
tested as an ablation.

## Current Implementation Mapping

Already implemented:

- `ThoughtModule` builds one internal thought from user text and a neutral state
  report.
- `build_neutral_state_report` creates LLM-facing reports without emotion
  labels.
- `run_internal_thought_ablation.py` tests injected thought conditions offline.
- `run_lmstudio_thought_feedback_suite.py` tests repeated local LLM-generated
  thought conditions.

Not yet implemented:

- persistent multi-module runtime loop,
- module creation/retirement,
- learned participation policy,
- shared answer arbitration beyond a fixed coordinator rule.

## Evaluation Plan

Minimum metrics for a two-module prototype:

- round count before termination,
- token cost per round,
- per-module trace distance after receiving peer messages,
- candidate-output agreement rate,
- response change versus single-module baseline,
- silence/update/send decision agreement with a fixed response gate,
- failure cases: runaway discussion, repeated paraphrase, overconfident
  speculation, and prompt-condition leakage.

## Interpretation Boundary

Passing this protocol test would show that multiple private EmoNet states can
exchange natural-language summaries and terminate under a fixed coordination
contract. It would not show that modules have stable personalities, emotions,
or biologically grounded roles.
