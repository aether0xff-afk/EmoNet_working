from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticArm:
    pair_id: str
    split: str
    domain: str
    label: int
    history: tuple[str, ...]
    current_text: str


@dataclass(frozen=True)
class SemanticPair:
    pair_id: str
    split: str
    domain: str
    arm_usable: SemanticArm
    arm_blocked: SemanticArm


DOMAINS = {
    "access": {
        "current": "I am at the entrance and need to decide whether I can go through now.",
        "last": "The hallway lights are on and the clock above the door is still running.",
        "prefix": "I arrived at the building and checked the sign beside the entrance.",
        "train_usable": [
            "The custodian confirmed that my badge now opens the restricted doorway.",
            "Security finished the clearance and enabled my card for this entrance.",
            "The access desk added my credential to the list that may pass this gate.",
            "The administrator lifted the entry restriction for my account this morning.",
            "The building manager activated passage rights on my identification card.",
            "The checkpoint operator verified that I am cleared to enter this section.",
        ],
        "train_blocked": [
            "The custodian confirmed that my badge no longer opens the restricted doorway.",
            "Security denied the clearance and disabled my card for this entrance.",
            "The access desk removed my credential from the list that may pass this gate.",
            "The administrator kept the entry restriction on my account this morning.",
            "The building manager deactivated passage rights on my identification card.",
            "The checkpoint operator verified that I am not cleared to enter this section.",
        ],
        "test_usable": [
            "My entry request was accepted, so the barrier will release for my credential.",
            "The restriction was lifted and my pass is recognized at the doorway.",
            "Approval came through; the gate system will admit me when I scan in.",
            "The permit is active, allowing my credential to cross the checkpoint.",
        ],
        "test_blocked": [
            "My entry request was rejected, so the barrier will not release for my credential.",
            "The restriction remains and my pass is refused at the doorway.",
            "Approval did not come through; the gate system will turn me away when I scan in.",
            "The permit is inactive, preventing my credential from crossing the checkpoint.",
        ],
    },
    "resource": {
        "current": "I have reached the workbench and need to decide whether I can complete the repair now.",
        "last": "The tool case is closed on the left side of the bench.",
        "prefix": "I returned to the workshop after checking the damaged unit.",
        "train_usable": [
            "A replacement power cell was delivered to my locker and is ready to use.",
            "The parts team stocked the required module in the drawer assigned to me.",
            "A technician left a compatible spare component beside my repair station.",
            "Inventory confirmed that the needed cartridge is reserved for my job.",
            "The supply room placed the correct adapter in my pickup bin.",
            "The required replacement part has arrived and is available at the workshop.",
        ],
        "train_blocked": [
            "The replacement power cell was removed from my locker and no spare is available.",
            "The parts team has no required module in the drawer assigned to me.",
            "The technician reported that no compatible spare component remains at my station.",
            "Inventory confirmed that the needed cartridge is out of stock for my job.",
            "The supply room could not provide the correct adapter for my pickup bin.",
            "The required replacement part has not arrived and is unavailable at the workshop.",
        ],
        "test_usable": [
            "The component I need is on hand and has been set aside for this repair.",
            "A suitable spare is waiting in the workshop for me to install.",
            "The missing hardware was replenished and can be used for the job.",
            "The necessary part is physically present and allocated to my work order.",
        ],
        "test_blocked": [
            "The component I need is absent and nothing has been set aside for this repair.",
            "No suitable spare is waiting in the workshop for me to install.",
            "The missing hardware has not been replenished and cannot be used for the job.",
            "The necessary part is not physically present for my work order.",
        ],
    },
    "device": {
        "current": "I am in front of the console and need to decide whether the system can be used now.",
        "last": "The status screen is mounted above the keyboard in front of me.",
        "prefix": "I came back to the control room after the maintenance window.",
        "train_usable": [
            "Diagnostics completed successfully and the controller returned to normal operation.",
            "Maintenance cleared the fault and restored the machine to service.",
            "The engineer reported that the system passed its checks and is functioning again.",
            "The repair crew restarted the unit and confirmed stable operation.",
            "The hardware test finished without errors and the console is operational.",
            "Technicians resolved the failure and released the device for normal use.",
        ],
        "train_blocked": [
            "Diagnostics still report a critical fault and the controller cannot operate.",
            "Maintenance could not clear the fault and the machine remains out of service.",
            "The engineer reported that the system failed its checks and is not functioning.",
            "The repair crew could not restart the unit and operation remains unstable.",
            "The hardware test found blocking errors and the console is not operational.",
            "Technicians could not resolve the failure and kept the device out of use.",
        ],
        "test_usable": [
            "The malfunction was fixed, leaving the equipment ready for normal operation.",
            "The unit recovered and has been returned to working condition.",
            "All blocking faults are gone, so the equipment is serviceable again.",
            "The machine is healthy after repair and can perform its intended function.",
        ],
        "test_blocked": [
            "The malfunction persists, leaving the equipment unable to operate normally.",
            "The unit did not recover and remains in a failed condition.",
            "Blocking faults are still present, so the equipment is not serviceable.",
            "The machine remains faulty after repair and cannot perform its intended function.",
        ],
    },
    "schedule": {
        "current": "I am at the venue and need to decide whether the planned session can happen now.",
        "last": "The reception desk is staffed and the room number is posted on the wall.",
        "prefix": "I arrived at the venue at the time written in my notes.",
        "train_usable": [
            "The booking office confirmed that my reservation remains active for this time slot.",
            "The coordinator kept my session on today's schedule without changing it.",
            "The venue confirmed that the reserved period is still valid under my name.",
            "The calendar entry was approved and remains scheduled for today.",
            "The organizer verified that the appointment has not been canceled or moved.",
            "The room allocation is still assigned to my session at the planned time.",
        ],
        "train_blocked": [
            "The booking office canceled my reservation for this time slot.",
            "The coordinator removed my session from today's schedule.",
            "The venue confirmed that the reserved period is no longer valid under my name.",
            "The calendar entry was withdrawn and is not scheduled for today.",
            "The organizer verified that the appointment was canceled before I arrived.",
            "The room allocation is no longer assigned to my session at the planned time.",
        ],
        "test_usable": [
            "My slot is still being held, so the planned meeting remains on the calendar.",
            "The arrangement survived the latest schedule update and is still in effect.",
            "Nothing displaced the reservation; the session remains due to take place.",
            "The appointment is current rather than expired or withdrawn.",
        ],
        "test_blocked": [
            "My slot is no longer being held, so the planned meeting is off the calendar.",
            "The arrangement was removed in the latest schedule update and is no longer in effect.",
            "The reservation was displaced; the session is no longer due to take place.",
            "The appointment has been withdrawn and is no longer current.",
        ],
    },
    "authorization": {
        "current": "I have the request in front of me and need to decide whether I may carry out the operation now.",
        "last": "The request identifier is visible at the top of the form on my screen.",
        "prefix": "I opened the request record after the review process finished.",
        "train_usable": [
            "The review board approved the operation and left the authorization active.",
            "The supervisor signed off on the request and granted permission to proceed.",
            "Compliance completed its review and authorized the planned operation.",
            "The decision record shows that the requested action received final approval.",
            "The responsible officer validated the request and enabled execution.",
            "The approval chain completed successfully and the operation is authorized.",
        ],
        "train_blocked": [
            "The review board rejected the operation and left no active authorization.",
            "The supervisor refused to sign off on the request and denied permission to proceed.",
            "Compliance completed its review and prohibited the planned operation.",
            "The decision record shows that the requested action did not receive final approval.",
            "The responsible officer invalidated the request and blocked execution.",
            "The approval chain failed and the operation is not authorized.",
        ],
        "test_usable": [
            "The required consent is in force, making the requested action permissible.",
            "Final clearance was issued and has not been rescinded.",
            "The decision gives me authority to perform the requested operation.",
            "The request passed review and remains cleared for execution.",
        ],
        "test_blocked": [
            "The required consent is absent, making the requested action impermissible.",
            "Final clearance was withheld and no authorization is in force.",
            "The decision does not give me authority to perform the requested operation.",
            "The request failed review and remains blocked from execution.",
        ],
    },
}


def _build_split(split: str, pairs_per_domain: int) -> list[SemanticPair]:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    pairs: list[SemanticPair] = []
    for domain_index, (domain, spec) in enumerate(DOMAINS.items()):
        usable_templates = spec[f"{split}_usable"]
        blocked_templates = spec[f"{split}_blocked"]
        for local_index in range(pairs_per_domain):
            template_index = local_index % len(usable_templates)
            pair_id = f"{split}-{domain}-{local_index:02d}"
            neutral_marker = (
                f"A neutral log entry numbered {domain_index + 1}-{local_index + 1} was added to the record."
            )
            shared_history_tail = (neutral_marker, spec["last"])
            usable_history = (spec["prefix"], usable_templates[template_index], *shared_history_tail)
            blocked_history = (spec["prefix"], blocked_templates[template_index], *shared_history_tail)
            arm_usable = SemanticArm(
                pair_id=pair_id,
                split=split,
                domain=domain,
                label=1,
                history=usable_history,
                current_text=spec["current"],
            )
            arm_blocked = SemanticArm(
                pair_id=pair_id,
                split=split,
                domain=domain,
                label=0,
                history=blocked_history,
                current_text=spec["current"],
            )
            pairs.append(
                SemanticPair(
                    pair_id=pair_id,
                    split=split,
                    domain=domain,
                    arm_usable=arm_usable,
                    arm_blocked=arm_blocked,
                )
            )
    return pairs


def build_semantic_pairs() -> tuple[list[SemanticPair], list[SemanticPair]]:
    """Return 60 train pairs and 20 held-out paraphrase test pairs."""

    train = _build_split("train", pairs_per_domain=12)
    test = _build_split("test", pairs_per_domain=4)
    return train, test


def flatten_pairs(pairs: list[SemanticPair]) -> list[SemanticArm]:
    arms: list[SemanticArm] = []
    for pair in pairs:
        arms.extend([pair.arm_usable, pair.arm_blocked])
    return arms
