"""Reference checklist loading and heading -> requirement resolution.

The reference checklist is the scoring authority. Two things live here:

1. Loading the checklist into `Requirement` objects.
2. The heading vocabulary that maps a document heading onto a requirement ID.

(2) is what replaces vector retrieval. Package documents use the checklist
`Requirement_Name` verbatim as their section headings, so a normalised exact
match resolves the right clause with no embedding step and no chunking.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .models import Requirement

# Headings that do not literally repeat the Requirement_Name.
HEADING_ALIASES: dict[str, str] = {
    "federal requirements": "CC-01",
    "fhwa 1273 physical incorporation": "CC-01",
    "attachment status": "CC-01",
    "addenda and q a currency": "CC-08",
    "buy america baba applicability": "CC-09",
    "buy america babe applicability": "CC-09",
}

# Non-substantive headings that must never be treated as a clause.
IGNORED_HEADINGS = {
    "project summary",
    "proposal general notices",
    "official reference",
    "replacement text",
    "general conditions",
    "special provisions",
    "proposal and general notices",
    "federal contract provisions attachment",
}

REVISION_PREFIX = re.compile(r"^revision\s+to\s+", re.IGNORECASE)


def normalise(text: str) -> str:
    """Fold case, strip punctuation and collapse whitespace.

    Needed because the PDF text layer introduces cosmetic differences
    (e.g. checklist "Addenda and Q&A currency" vs rendered
    "Addenda and Q&A; currency").
    """
    text = text.replace("&", " ")
    text = re.sub(r"[^0-9a-zA-Z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


class ReferenceChecklist:
    """The 18 challenge requirements plus heading resolution."""

    def __init__(self, requirements: list[Requirement]):
        self.requirements = requirements
        self._by_id = {r.requirement_id: r for r in requirements}
        self._heading_index: dict[str, str] = {}
        for req in requirements:
            self._heading_index[normalise(req.requirement_name)] = req.requirement_id
        self._heading_index.update(
            {normalise(k): v for k, v in HEADING_ALIASES.items()}
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceChecklist":
        rows: list[Requirement] = []
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for raw in csv.DictReader(fh):
                rows.append(
                    Requirement(
                        requirement_id=raw["Requirement_ID"].strip(),
                        tier=raw["Tier"].strip(),
                        requirement_name=raw["Requirement_Name"].strip(),
                        reference_source=raw["Reference_Source"].strip(),
                        section=raw["Section"].strip(),
                        applicability_rule=raw["Applicability_Rule"].strip(),
                        review_expectation=raw["Review_Expectation"].strip(),
                        severity_guidance=raw["Severity_Guidance"].strip(),
                        evidence_required=raw["Evidence_Required"].strip(),
                        challenge_reference_rule=raw["Challenge_Reference_Rule"].strip(),
                    )
                )
        return cls(rows)

    # -- lookup -------------------------------------------------------------

    def __iter__(self):
        return iter(self.requirements)

    def __len__(self) -> int:
        return len(self.requirements)

    def get(self, requirement_id: str) -> Requirement:
        return self._by_id[requirement_id]

    @property
    def ids(self) -> list[str]:
        return [r.requirement_id for r in self.requirements]

    def resolve_heading(self, heading: str) -> str | None:
        """Map a document heading to a requirement ID, or None.

        Handles the Addendum form "Revision to <Requirement_Name>".
        """
        stripped = REVISION_PREFIX.sub("", heading.strip())
        key = normalise(stripped)
        if not key or key in IGNORED_HEADINGS:
            return None
        if key in self._heading_index:
            return self._heading_index[key]
        # Fall back to containment so minor heading drift still resolves.
        for known, req_id in self._heading_index.items():
            if known and (known in key or key in known):
                return req_id
        return None

    def is_known_heading(self, heading: str) -> bool:
        stripped = REVISION_PREFIX.sub("", heading.strip())
        return normalise(stripped) in self._heading_index

    @property
    def known_heading_keys(self) -> set[str]:
        return set(self._heading_index)
