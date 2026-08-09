from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
V54_ROOT = REPO_ROOT / "v5_4_fresh_confirmatory"
sys.path.insert(0, str(V54_ROOT))

from fresh_fixture import FreshArm, FreshPair, build_fresh_pairs, flatten_pairs as flatten_fresh  # noqa: E402


@dataclass(frozen=True)
class PredictiveArm:
    pair_id: str
    split: str
    domain: str
    label: int
    history: tuple[str, ...]
    current_text: str
    future_text: str


@dataclass(frozen=True)
class PredictivePair:
    pair_id: str
    split: str
    domain: str
    positive: PredictiveArm
    negative: PredictiveArm


FUTURES = {
    "connectivity": {
        "train_positive": [
            "The live transfer reaches the remote receiver and an acknowledgment comes back.",
            "The outgoing data is delivered successfully through the link to the distant endpoint.",
            "The transmission completes and the remote system confirms receipt.",
            "Packets from the live transfer arrive at the destination without a reachability failure.",
            "The data stream crosses the network path and the far receiver accepts it.",
            "The send operation finishes with confirmation from the remote side.",
        ],
        "train_negative": [
            "The live transfer cannot reach the remote receiver and no acknowledgment comes back.",
            "The outgoing data fails to reach the distant endpoint because the link is unavailable.",
            "The transmission fails and the remote system never confirms receipt.",
            "Packets from the live transfer do not arrive at the destination because reachability is absent.",
            "The data stream cannot cross the network path and the far receiver gets nothing.",
            "The send operation ends without confirmation because the remote side cannot be reached.",
        ],
        "test_positive": [
            "The remote endpoint receives the new data and returns a successful delivery response.",
            "The live payload traverses the connection and appears at the destination.",
            "The attempted transmission succeeds, with the far side reporting receipt.",
            "Communication proceeds normally and the outgoing information reaches its receiver.",
        ],
        "test_negative": [
            "The remote endpoint receives no new data and returns no successful delivery response.",
            "The live payload cannot traverse the connection and never appears at the destination.",
            "The attempted transmission fails, with no receipt reported by the far side.",
            "Communication cannot proceed and the outgoing information does not reach its receiver.",
        ],
    },
    "capacity": {
        "train_positive": [
            "The scheduler admits the new workload and allocates the capacity it requested.",
            "The batch enters the system without exceeding the allocation limit.",
            "Resources are assigned to the incoming work and processing begins normally.",
            "The request is accepted because enough headroom remains for its allocation.",
            "The new workload is placed successfully and the capacity counters stay within quota.",
            "Admission succeeds and the requested resource units are reserved for the batch.",
        ],
        "train_negative": [
            "The scheduler rejects the new workload because the requested capacity cannot be allocated.",
            "The batch is refused because admitting it would exceed the allocation limit.",
            "Resources cannot be assigned to the incoming work, so processing does not begin.",
            "The request is rejected because there is not enough headroom for its allocation.",
            "The new workload cannot be placed because the capacity quota would be exceeded.",
            "Admission fails and the requested resource units cannot be reserved for the batch.",
        ],
        "test_positive": [
            "The pending work is accepted and receives enough resources to start.",
            "Allocation succeeds, allowing the additional workload into the system.",
            "The system admits the batch while remaining inside its capacity ceiling.",
            "The request obtains the required resource allocation and begins execution.",
        ],
        "test_negative": [
            "The pending work is rejected because enough resources cannot be assigned.",
            "Allocation fails, preventing the additional workload from entering the system.",
            "The system refuses the batch because its capacity ceiling would be exceeded.",
            "The request does not obtain the required resource allocation and cannot begin execution.",
        ],
    },
    "integrity": {
        "train_positive": [
            "The next procedure accepts the package as verified and continues without an integrity hold.",
            "Processing proceeds because the item is treated as intact and trusted.",
            "The package passes the handoff check and moves into the normal next stage.",
            "The verified item is released for further handling without quarantine.",
            "The inspection result allows the package to continue through the standard procedure.",
            "The system records the item as trustworthy and permits the next operation.",
        ],
        "train_negative": [
            "The next procedure places the package on hold because its integrity cannot be trusted.",
            "Processing stops because the item is treated as potentially compromised.",
            "The package fails the handoff check and is diverted from the normal next stage.",
            "The item is quarantined instead of being released for further handling.",
            "The inspection result prevents the package from continuing through the standard procedure.",
            "The system records an integrity concern and blocks the next operation.",
        ],
        "test_positive": [
            "The package advances to the following step because its verified condition is accepted.",
            "Normal handling continues with the item regarded as uncompromised.",
            "The item clears the integrity gate and proceeds through the workflow.",
            "No protective hold is applied, so the package moves onward normally.",
        ],
        "test_negative": [
            "The package cannot advance because its condition triggers an integrity hold.",
            "Normal handling stops with the item regarded as potentially compromised.",
            "The item fails the integrity gate and is removed from the normal workflow.",
            "A protective hold is applied, preventing the package from moving onward normally.",
        ],
    },
    "route": {
        "train_positive": [
            "I continue along the planned passage and reach the next waypoint on the route.",
            "Travel proceeds through the required corridor toward the destination.",
            "The journey continues on the intended path without being stopped by a closure.",
            "I cross the required section and move closer to the destination.",
            "Navigation continues through the open route to the following waypoint.",
            "The vehicle proceeds along the planned corridor without needing a detour.",
        ],
        "train_negative": [
            "I cannot continue along the planned passage and stop before the blocked section.",
            "Travel cannot proceed through the required corridor toward the destination.",
            "The journey is stopped on the intended path by the active closure.",
            "I cannot cross the required section and must remain before the obstruction.",
            "Navigation halts because the route to the following waypoint is closed.",
            "The vehicle cannot proceed along the planned corridor and must seek another route.",
        ],
        "test_positive": [
            "Movement continues through the intended path and the next route point is reached.",
            "I pass through the required corridor and continue toward the destination.",
            "The open path lets the trip proceed without interruption at this section.",
            "The planned route carries me beyond the transfer point toward the next waypoint.",
        ],
        "test_negative": [
            "Movement stops at the intended path and the next route point cannot be reached this way.",
            "I cannot pass through the required corridor and cannot continue toward the destination.",
            "The closed path prevents the trip from proceeding through this section.",
            "The planned route cannot carry me beyond the transfer point because passage is blocked.",
        ],
    },
    "assignment": {
        "train_positive": [
            "I begin handling the work item because it is present in my assigned queue.",
            "The workflow allows me to take ownership actions on the task as its current assignee.",
            "I proceed with the job and my account is recorded as the responsible operator.",
            "The task appears in my active workload and I start the required work.",
            "I can perform the assignee-only operation because responsibility belongs to me.",
            "The work item moves forward under my account as the designated owner.",
        ],
        "train_negative": [
            "I do not begin handling the work item because it belongs in another operator's queue.",
            "The workflow prevents me from taking ownership actions because someone else is the assignee.",
            "I cannot proceed as the responsible operator because another account owns the job.",
            "The task does not appear in my active workload, so I do not start the required work.",
            "I cannot perform the assignee-only operation because responsibility belongs to someone else.",
            "The work item remains under another account rather than moving forward under mine.",
        ],
        "test_positive": [
            "I act on the task and the workflow accepts me as its responsible assignee.",
            "The item enters my active work because ownership is currently assigned to my account.",
            "I am permitted to carry out the owner-specific step for this work item.",
            "Processing continues under my responsibility as the current assignee.",
        ],
        "test_negative": [
            "I cannot act on the task as its owner because the workflow recognizes another assignee.",
            "The item does not enter my active work because ownership belongs to another account.",
            "I am not permitted to carry out the owner-specific step for this work item.",
            "Processing remains under someone else's responsibility rather than mine.",
        ],
    },
}


def _attach_future(arm: FreshArm, future_text: str) -> PredictiveArm:
    return PredictiveArm(
        pair_id=arm.pair_id,
        split=arm.split,
        domain=arm.domain,
        label=arm.label,
        history=arm.history,
        current_text=arm.current_text,
        future_text=future_text,
    )


def build_predictive_pairs() -> tuple[list[PredictivePair], list[PredictivePair]]:
    fresh_train, fresh_test = build_fresh_pairs()
    outputs: list[list[PredictivePair]] = []
    for split, pairs in (("train", fresh_train), ("test", fresh_test)):
        result: list[PredictivePair] = []
        for pair in pairs:
            spec = FUTURES[pair.domain]
            local_index = int(pair.pair_id.rsplit("-", 1)[-1])
            positive_templates = spec[f"{split}_positive"]
            negative_templates = spec[f"{split}_negative"]
            template_index = local_index % len(positive_templates)
            result.append(
                PredictivePair(
                    pair_id=pair.pair_id,
                    split=split,
                    domain=pair.domain,
                    positive=_attach_future(pair.positive, positive_templates[template_index]),
                    negative=_attach_future(pair.negative, negative_templates[template_index]),
                )
            )
        outputs.append(result)
    return outputs[0], outputs[1]


def flatten_pairs(pairs: list[PredictivePair]) -> list[PredictiveArm]:
    arms: list[PredictiveArm] = []
    for pair in pairs:
        arms.extend([pair.positive, pair.negative])
    return arms
