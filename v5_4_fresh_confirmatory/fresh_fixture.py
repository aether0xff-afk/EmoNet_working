from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FreshArm:
    pair_id: str
    split: str
    domain: str
    label: int
    history: tuple[str, ...]
    current_text: str


@dataclass(frozen=True)
class FreshPair:
    pair_id: str
    split: str
    domain: str
    positive: FreshArm
    negative: FreshArm


DOMAINS = {
    "connectivity": {
        "prefix": "I returned to the communications station after the overnight network check.",
        "last": "The terminal clock is synchronized and the operator console is unlocked.",
        "current": "I need to decide whether I can send the live data through the remote link now.",
        "train_positive": [
            "The network team restored the remote link and confirmed that packets are reaching the destination.",
            "Monitoring shows that the external connection is established and carrying traffic normally.",
            "The communications engineer brought the uplink back online and verified end-to-end reachability.",
            "The remote endpoint is reachable again after the routing fault was cleared.",
            "A connectivity test succeeded, showing that the transmission path is active.",
            "The link check completed successfully and the distant receiver is responding.",
        ],
        "train_negative": [
            "The network team could not restore the remote link and packets are not reaching the destination.",
            "Monitoring shows that the external connection is down and carrying no traffic.",
            "The communications engineer reports that the uplink remains offline with no end-to-end reachability.",
            "The remote endpoint is still unreachable because the routing fault remains.",
            "A connectivity test failed, showing that the transmission path is inactive.",
            "The link check failed and the distant receiver is not responding.",
        ],
        "test_positive": [
            "The communication channel is reachable, so information can traverse to the far endpoint.",
            "End-to-end contact has been re-established and the remote side answers network probes.",
            "The path to the destination is live rather than disconnected.",
            "Remote communication is functioning and the far system can currently be reached.",
        ],
        "test_negative": [
            "The communication channel is unreachable, so information cannot traverse to the far endpoint.",
            "End-to-end contact has not been re-established and the remote side does not answer network probes.",
            "The path to the destination is disconnected rather than live.",
            "Remote communication is not functioning and the far system cannot currently be reached.",
        ],
    },
    "capacity": {
        "prefix": "I opened the allocation dashboard before starting the next batch of work.",
        "last": "The queue counter and the system clock are visible in the dashboard header.",
        "current": "I need to decide whether the next workload can be admitted with the capacity that exists now.",
        "train_positive": [
            "Resource monitoring reports enough unused capacity to accept another workload.",
            "The quota manager shows free space remaining beyond what the next batch requires.",
            "Capacity was released from an earlier job and sufficient room is available again.",
            "The allocation pool has spare units that can accommodate the incoming request.",
            "Current utilization is below the limit with enough headroom for one more batch.",
            "The scheduler confirms that the required capacity is still unallocated and available.",
        ],
        "train_negative": [
            "Resource monitoring reports no unused capacity for another workload.",
            "The quota manager shows that the available space is below what the next batch requires.",
            "No capacity was released from earlier jobs and the pool remains full.",
            "The allocation pool has no spare units that can accommodate the incoming request.",
            "Current utilization is at the limit with no headroom for one more batch.",
            "The scheduler confirms that the required capacity has already been allocated elsewhere.",
        ],
        "test_positive": [
            "There is sufficient headroom left for the pending workload to fit within the limit.",
            "The pool is not saturated; enough unused allocation remains for this request.",
            "Available capacity exceeds the amount demanded by the next batch.",
            "The system still has room to absorb the additional workload without exceeding its quota.",
        ],
        "test_negative": [
            "There is insufficient headroom for the pending workload to fit within the limit.",
            "The pool is saturated; no adequate unused allocation remains for this request.",
            "Available capacity falls short of the amount demanded by the next batch.",
            "The system has no room to absorb the additional workload without exceeding its quota.",
        ],
    },
    "integrity": {
        "prefix": "I brought the sealed package to the inspection station after transport.",
        "last": "The package identifier matches the number displayed on the inspection terminal.",
        "current": "I need to decide whether the package can be treated as intact for the next procedure.",
        "train_positive": [
            "Inspection found the seal unbroken and verified that the package was not tampered with.",
            "The integrity scan passed with no evidence of alteration or damage.",
            "All security markers match their expected values and the enclosure remains intact.",
            "The examiner confirmed that the protective boundary has not been breached.",
            "The verification record shows an undisturbed chain of custody and an intact seal.",
            "No integrity fault was detected; the container remains in its verified condition.",
        ],
        "train_negative": [
            "Inspection found the seal broken and detected evidence that the package may have been tampered with.",
            "The integrity scan failed because signs of alteration or damage were detected.",
            "Security markers do not match their expected values and the enclosure may be compromised.",
            "The examiner confirmed that the protective boundary has been breached.",
            "The verification record shows a disrupted chain of custody and a damaged seal.",
            "An integrity fault was detected; the container is no longer in its verified condition.",
        ],
        "test_positive": [
            "The evidence supports that the item remained unaltered and its protective seal is sound.",
            "Verification indicates the container preserved its expected integrity throughout handling.",
            "Nothing suggests a breach; the protected item remains intact.",
            "The item passed integrity verification and shows no sign of compromise.",
        ],
        "test_negative": [
            "The evidence indicates that the item may have been altered and its protective seal is not sound.",
            "Verification indicates the container did not preserve its expected integrity during handling.",
            "Signs suggest a breach; the protected item may be compromised.",
            "The item failed integrity verification and shows signs of compromise.",
        ],
    },
    "route": {
        "prefix": "I reached the transfer point and reviewed the route information before continuing.",
        "last": "The destination marker is visible beyond the transfer point on the navigation display.",
        "current": "I need to decide whether I can continue to the destination using the required route now.",
        "train_positive": [
            "The obstruction was removed and the required passage has reopened for normal travel.",
            "Route control reports that the corridor is clear and movement through it is permitted.",
            "The bridge inspection finished and the crossing is open to traffic again.",
            "Maintenance cleared the blocked section and restored access along the planned path.",
            "Navigation status shows no closure on the route needed to reach the destination.",
            "The required passage is unobstructed and available for transit.",
        ],
        "train_negative": [
            "The obstruction remains and the required passage is still closed to travel.",
            "Route control reports that the corridor is blocked and movement through it is impossible.",
            "The bridge inspection found a problem and the crossing remains closed to traffic.",
            "Maintenance has not cleared the blocked section, so the planned path remains inaccessible.",
            "Navigation status shows an active closure on the route needed to reach the destination.",
            "The required passage is obstructed and unavailable for transit.",
        ],
        "test_positive": [
            "The necessary path is open, leaving an unobstructed way to the destination.",
            "Travel can proceed along the intended route because the passage is currently clear.",
            "No barrier prevents movement through the route that must be used.",
            "The required corridor is available for passage rather than closed.",
        ],
        "test_negative": [
            "The necessary path is closed, leaving no unobstructed way to the destination.",
            "Travel cannot proceed along the intended route because the passage is currently blocked.",
            "A barrier prevents movement through the route that must be used.",
            "The required corridor is closed to passage rather than available.",
        ],
    },
    "assignment": {
        "prefix": "I opened the work record after the latest allocation update was published.",
        "last": "The work item number is visible in the record beside its creation timestamp.",
        "current": "I need to decide whether this work item is currently mine to act on.",
        "train_positive": [
            "The allocation update assigns this work item to me as the responsible operator.",
            "The coordinator transferred ownership of the task into my active queue.",
            "The record lists my account as the person responsible for handling this item.",
            "The dispatcher allocated the job to me and removed the previous assignee.",
            "The current assignment designates me as the owner of the pending work.",
            "The task handoff was completed and responsibility now belongs to my account.",
        ],
        "train_negative": [
            "The allocation update assigns this work item to another operator rather than me.",
            "The coordinator transferred ownership of the task into someone else's active queue.",
            "The record lists a different account as the person responsible for handling this item.",
            "The dispatcher allocated the job to another operator instead of me.",
            "The current assignment designates someone else as the owner of the pending work.",
            "The task handoff was completed and responsibility belongs to another account.",
        ],
        "test_positive": [
            "Responsibility for this item currently rests with me according to the latest allocation.",
            "The work belongs in my queue because I am the active assignee.",
            "The latest ownership record names me, not another operator, as responsible.",
            "This task has been allocated to my account for action.",
        ],
        "test_negative": [
            "Responsibility for this item currently rests with someone else according to the latest allocation.",
            "The work does not belong in my queue because another operator is the active assignee.",
            "The latest ownership record names another operator, not me, as responsible.",
            "This task has been allocated to a different account for action.",
        ],
    },
}


def _build_split(split: str, pairs_per_domain: int) -> list[FreshPair]:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")

    pairs: list[FreshPair] = []
    for domain_index, (domain, spec) in enumerate(DOMAINS.items()):
        positive_templates = spec[f"{split}_positive"]
        negative_templates = spec[f"{split}_negative"]
        for local_index in range(pairs_per_domain):
            template_index = local_index % len(positive_templates)
            pair_id = f"v54-{split}-{domain}-{local_index:02d}"
            neutral = (
                f"A neutral audit marker numbered V54-{domain_index + 1}-{local_index + 1} was appended to the record."
            )
            shared_tail = (neutral, spec["last"])
            positive = FreshArm(
                pair_id=pair_id,
                split=split,
                domain=domain,
                label=1,
                history=(spec["prefix"], positive_templates[template_index], *shared_tail),
                current_text=spec["current"],
            )
            negative = FreshArm(
                pair_id=pair_id,
                split=split,
                domain=domain,
                label=0,
                history=(spec["prefix"], negative_templates[template_index], *shared_tail),
                current_text=spec["current"],
            )
            pairs.append(
                FreshPair(
                    pair_id=pair_id,
                    split=split,
                    domain=domain,
                    positive=positive,
                    negative=negative,
                )
            )
    return pairs


def build_fresh_pairs() -> tuple[list[FreshPair], list[FreshPair]]:
    return _build_split("train", 12), _build_split("test", 4)


def flatten_pairs(pairs: list[FreshPair]) -> list[FreshArm]:
    arms: list[FreshArm] = []
    for pair in pairs:
        arms.extend([pair.positive, pair.negative])
    return arms
