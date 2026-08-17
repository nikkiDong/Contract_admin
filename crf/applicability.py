"""Layer 1: APPLIES / DOES_NOT_APPLY from project metadata.

Each checklist `Applicability_Rule` reduces to a predicate over
Project_Metadata.json. This layer is intentionally free of any model call: the
gate is a stated rule, so it should be executed, not inferred. It also
suppresses the largest source of over-flagging, since a DOES_NOT_APPLY row can
never produce a FLAG.
"""

from __future__ import annotations

from typing import Callable

from .models import Package

APPLIES = "APPLIES"
DOES_NOT_APPLY = "DOES_NOT_APPLY"

# requirement_id -> (predicate, reason when true, reason when false)
_RULES: dict[str, tuple[Callable[[Package], bool], str, str]] = {
    "CC-01": (
        lambda p: p.federal_aid,
        "Project metadata reports federal_aid=Yes, so the Title 23 federal-aid "
        "contract provisions requirement is in scope.",
        "Project metadata reports federal_aid=No, so FHWA-1273 incorporation is "
        "not triggered for this package.",
    ),
    "CC-02": (
        lambda p: True,
        "DelDOT bid package governed by Section 102.8; proposal guaranty applies.",
        "",
    ),
    "CC-03": (
        lambda p: True,
        "DelDOT proposal package; the non-collusive bidding certification applies.",
        "",
    ),
    "CC-04": (
        lambda p: True,
        "Contract execution package governed by Section 103.5; performance and "
        "payment bonds apply.",
        "",
    ),
    "CC-05": (
        lambda p: True,
        "Successful-bidder execution provisions are present, so Section 103.7 applies.",
        "",
    ),
    "CC-06": (
        lambda p: True,
        "Delaware construction/maintenance contractor context; the Contractor "
        "Registration Act notice applies.",
        "",
    ),
    "CC-07": (
        lambda p: True,
        "Delaware public works proposal; contractor and subcontractor licensing "
        "requirements apply.",
        "",
    ),
    "CC-08": (
        lambda p: bool(p.issued_addenda),
        "Project metadata lists issued addenda ({addenda}), so addenda currency applies.",
        "Project metadata lists no issued addenda or Q&A, so there is no addenda "
        "currency obligation to test.",
    ),
    "CC-09": (
        lambda p: p.baba_applicable,
        "Project metadata reports buy_america_baba_applicable=Yes.",
        "Project metadata reports buy_america_baba_applicable=No, so Buy America/BABA "
        "is outside scope for this package.",
    ),
    "CC-10": (
        lambda p: True,
        "Multiple contract documents are incorporated and may conflict, so Section "
        "105.6 coordination applies.",
        "",
    ),
    "CC-11": (
        lambda p: True,
        "Package contains contract-change provisions, so the Section 104.2 written "
        "change process applies.",
        "",
    ),
    "CC-12": (
        lambda p: True,
        "Package contains change-notification requirements, so Section 104.3 applies.",
        "",
    ),
    "CC-13": (
        lambda p: True,
        "Contract and subcontract performance records exist, so audit rights and "
        "retention apply.",
        "",
    ),
    "CC-14": (
        lambda p: p.subcontracting_planned,
        "Project metadata reports subcontracting_planned=Yes, so Section 108.1 "
        "subletting limits apply.",
        "Project metadata reports subcontracting_planned=No, so the subletting "
        "requirement is not engaged.",
    ),
    "CC-15": (
        lambda p: p.claim_event,
        "Project metadata reports claim_event=Yes, so the Section 105.15 claims "
        "procedure applies.",
        "Project metadata reports claim_event=No, so no claim scenario exists to review.",
    ),
    "CC-16": (
        lambda p: p.delay_event,
        "Project metadata reports delay_event=Yes, so Section 108.7 time extensions apply.",
        "Project metadata reports delay_event=No, so no delay/time-extension scenario "
        "exists to review.",
    ),
    "CC-17": (
        lambda p: True,
        "Package includes a liquidated-damages provision or governing schedule, so "
        "Section 108.9 applies.",
        "",
    ),
    "CC-18": (
        lambda p: p.changed_work_event,
        "Project metadata reports changed_work_event=Yes, so Section 109.4 "
        "compensation for changes applies.",
        "Project metadata reports changed_work_event=No, so no changed-work pricing "
        "scenario exists to review.",
    ),
}


def decide(package: Package, requirement_id: str) -> tuple[str, str]:
    """Return (applicability_decision, applicability_reason)."""
    rule = _RULES.get(requirement_id)
    if rule is None:
        return APPLIES, "No metadata gate defined; defaulting to in-scope for review."

    predicate, reason_true, reason_false = rule
    if predicate(package):
        reason = reason_true.format(addenda=", ".join(package.issued_addenda) or "none")
        return APPLIES, reason
    return DOES_NOT_APPLY, reason_false
