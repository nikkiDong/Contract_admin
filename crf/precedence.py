"""Layer 2: pick the governing clause for a requirement.

Two ordered mechanisms, in this priority:

1. **Addendum supersession.** An Addendum carrying "Revision to <Requirement>"
   plus REPLACEMENT TEXT validly revises that named provision. The
   latest-issued such Addendum governs. This must run before any deviation
   test, otherwise superseded base text produces false positives (the single
   biggest scoring trap in this dataset).

2. **DelDOT 105.6 order of precedence** for base documents:
   General Description > General Notices > Plans > Special Provisions >
   Standard Construction Details > Standard Specifications >
   Electronic Design Data Files.
"""

from __future__ import annotations

import re

from .models import Clause, Package
from .reference import ReferenceChecklist

# DelDOT 105.6 ladder. Lower number = higher authority.
PRECEDENCE_105_6: list[str] = [
    "General Description",
    "General Notices",
    "Plans",
    "Special Provisions",
    "Standard Construction Details",
    "Standard Specifications",
    "Electronic Design Data Files",
]

# Package document types mapped onto the 105.6 ladder.
_DOC_TYPE_RANK: dict[str, int] = {
    "general description": 0,
    "proposal and general notices": 1,
    "general notices": 1,
    "proposal": 1,
    "plans": 2,
    "special provisions": 3,
    "standard construction details": 4,
    "general conditions": 5,          # nearest analogue: Standard Specifications
    "standard specifications": 5,
    "electronic design data files": 6,
    "fhwa-1273 contract provisions attachment": 1,
}

_ADDENDUM_LETTER = re.compile(r"addendum[_\s]*([a-z])\b", re.IGNORECASE)
_ADDENDUM_NUMBER = re.compile(r"addendum[_\s]*(\d+)\b", re.IGNORECASE)


def is_addendum(clause: Clause) -> bool:
    return "addendum" in clause.doc_type.lower() or "addendum" in clause.file_name.lower()


def addendum_ordinal(clause: Clause) -> int:
    """Issue order of an addendum: Addendum_A -> 1, Addendum_B -> 2, ...

    The packages label files by letter and metadata by number; both map onto the
    same 1-based issue sequence.
    """
    for source in (clause.doc_type, clause.file_name):
        m = _ADDENDUM_NUMBER.search(source)
        if m:
            return int(m.group(1))
        m = _ADDENDUM_LETTER.search(source)
        if m:
            return ord(m.group(1).lower()) - ord("a") + 1
    return 0


def doc_rank(clause: Clause) -> int:
    key = clause.doc_type.strip().lower()
    if key in _DOC_TYPE_RANK:
        return _DOC_TYPE_RANK[key]
    for name, rank in _DOC_TYPE_RANK.items():
        if name in key:
            return rank
    return len(PRECEDENCE_105_6)


def candidates(
    package: Package, requirement_id: str, checklist: ReferenceChecklist
) -> list[Clause]:
    """Every clause in the package that addresses this requirement."""
    return [
        c for c in package.clauses
        if checklist.resolve_heading(c.heading) == requirement_id
    ]


def resolve(
    package: Package, requirement_id: str, checklist: ReferenceChecklist
) -> tuple[Clause | None, list[Clause], str]:
    """Return (governing_clause, superseded_clauses, resolution_note)."""
    found = candidates(package, requirement_id, checklist)
    if not found:
        return None, [], "No clause addressing this requirement was located in the package."

    revisions = [c for c in found if is_addendum(c) and c.is_replacement]
    if revisions:
        governing = max(revisions, key=addendum_ordinal)
        superseded = [c for c in found if c is not governing]
        note = (
            f"{governing.doc_type} explicitly revises this named provision with "
            f"replacement text and therefore governs over the earlier "
            f"{', '.join(sorted({c.doc_type for c in superseded})) or 'base'} text."
        )
        return governing, superseded, note

    base = sorted(found, key=lambda c: (doc_rank(c), c.page))
    governing = base[0]
    superseded = base[1:]
    if superseded:
        note = (
            f"No Addendum revises this provision. Under DelDOT 105.6 the "
            f"{governing.doc_type} text governs over "
            f"{', '.join(sorted({c.doc_type for c in superseded}))}."
        )
    else:
        note = f"Single governing occurrence, in {governing.doc_type}."
    return governing, superseded, note



