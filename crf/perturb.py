"""Perturbation library for robustness testing.

The detectors in `detectors.py` were written with the development documents in
hand. That makes 108/108 on the development split a statement about sufficiency,
not about generalisation. This module manufactures inputs whose correct answer is
known *by construction*, so generalisation can be measured without more labels.

Two case classes:

* **invariance** - the transformation preserves meaning, so every decision must
  stay exactly as it was. Any change is a false positive or false negative
  caused by phrasing coupling.
* **directional** - the transformation changes meaning in a specific way, so one
  named requirement must move to a stated label and *nothing else may move*.

The second half of that sentence is the part that catches over-broad regexes:
a detector that fires on the wrong requirement shows up as collateral damage
even when its own target row is correct.

Transformations mutate a deep-copied `Package` in place. They operate on clause
text, clause headings and metadata - i.e. everything downstream of PDF
extraction. Extraction itself is out of scope here and is covered by the
137/137 heading-resolution check in SOLUTION.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .models import Clause, Package
from .reference import ReferenceChecklist

# ---------------------------------------------------------------------------
# Clause selection helpers
# ---------------------------------------------------------------------------


def clauses_for(
    pkg: Package, checklist: ReferenceChecklist, req_id: str
) -> list[Clause]:
    return [c for c in pkg.clauses if checklist.resolve_heading(c.heading) == req_id]


def edit_text(
    pkg: Package,
    checklist: ReferenceChecklist,
    req_id: str,
    fn: Callable[[str], str],
) -> None:
    """Rewrite the body of every clause addressing `req_id`."""
    for clause in clauses_for(pkg, checklist, req_id):
        clause.text = fn(clause.text)


def edit_all_text(pkg: Package, fn: Callable[[str], str]) -> None:
    """Rewrite the body of every clause in the package."""
    for clause in pkg.clauses:
        clause.text = fn(clause.text)


def edit_all_headings(pkg: Package, fn: Callable[[str], str]) -> None:
    for clause in pkg.clauses:
        clause.heading = fn(clause.heading)
        if clause.revises_heading:
            clause.revises_heading = fn(clause.revises_heading)


def add_addendum(
    pkg: Package,
    checklist: ReferenceChecklist,
    req_id: str,
    letter: str,
    text: str,
) -> None:
    """Append an Addendum carrying replacement text for `req_id`."""
    heading = checklist.get(req_id).requirement_name
    pkg.clauses.append(
        Clause(
            package_id=pkg.package_id,
            file_name=f"Addendum_{letter}.pdf",
            doc_type=f"Addendum {letter}",
            heading=heading,
            text=text,
            page=1,
            is_replacement=True,
            revises_heading=heading,
        )
    )
    ordinal = ord(letter.upper()) - ord("A") + 1
    issued = list(pkg.metadata.get("issued_addenda") or [])
    while len(issued) < ordinal:
        issued.append(f"Addendum {len(issued) + 1}")
    pkg.metadata["issued_addenda"] = issued


# ---------------------------------------------------------------------------
# Meaning-preserving text transformations
# ---------------------------------------------------------------------------

_SPELLED = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
    r"(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine|hundred|tenth))*"
)


def strip_parenthetical_digits(text: str) -> str:
    """`ten percent (10%)` -> `ten percent`; `three (3) years` -> `three years`.

    Removes the digit form the documents always supply alongside the spelled
    form. Detectors that read only digits lose their input entirely.
    """
    return re.sub(r"\s*\((\d[\d,]*)\s*%?\)", "", text)


def digits_only(text: str) -> str:
    """`ten percent (10%)` -> `10%`; `thirty (30) calendar days` -> `30 calendar days`."""
    text = re.sub(rf"\bone-tenth\s*\((\d+)\s*%\)", r"\1%", text, flags=re.IGNORECASE)
    text = re.sub(
        rf"\b{_SPELLED}\s+percent\s*\((\d+)\s*%\)", r"\1%", text, flags=re.IGNORECASE
    )
    text = re.sub(rf"\b{_SPELLED}\s*\((\d+)\)", r"\1", text, flags=re.IGNORECASE)
    return text


def years_to_months(text: str) -> str:
    """`three (3) years` -> `36 months`. Equivalent retention period."""
    def repl(m: re.Match) -> str:
        return f"{int(m.group(1)) * 12} months"

    return re.sub(rf"\b{_SPELLED}\s*\((\d+)\)\s*years?", repl, text, flags=re.IGNORECASE)


def reorder_sentences(text: str) -> str:
    """Reverse sentence order. Requirement content is unchanged."""
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(])", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        return text
    return " ".join(reversed(parts))


_SYNONYMS = [
    (r"\bshall\b", "must"),
    (r"\bfurnish(ed)?\b", "provide"),
    (r"\bprior to\b", "before"),
    (r"\bsufficient\b", "adequate"),
    (r"\brequired\b", "mandatory"),
    (r"\btotal bid price\b", "aggregate bid amount"),
    (r"\bremains? reviewable\b", "stays open to review"),
    (r"\bacceptable\b", "satisfactory"),
]


def synonym_swap(text: str) -> str:
    """Swap legal-drafting synonyms that do not change the obligation."""
    for pattern, replacement in _SYNONYMS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def prepend_boilerplate(text: str) -> str:
    """Add a non-operative preamble of the kind real drafts carry."""
    return "Notwithstanding any other provision of the contract documents, " + text


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", "  ", text).strip()


def heading_repunctuate(heading: str) -> str:
    """`Proposal guaranty / bid bond` -> `PROPOSAL GUARANTY - BID BOND`."""
    return heading.replace("/", "-").upper()


# ---------------------------------------------------------------------------
# Meaning-changing text transformations
# ---------------------------------------------------------------------------


def set_percent(target: int) -> Callable[[str], str]:
    """Rewrite every percentage to `target`, spelled and digit forms alike."""
    words = {
        5: "five", 10: "ten", 25: "twenty-five", 50: "fifty",
        75: "seventy-five", 80: "eighty", 100: "one hundred",
    }
    word = words.get(target, str(target))

    def fn(text: str) -> str:
        text = re.sub(
            rf"\b(?:one-tenth|{_SPELLED})\s+percent\s*\(\d+\s*%\)",
            f"{word} percent ({target}%)",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bone-tenth\s*\(\d+\s*%\)", f"{word} percent ({target}%)", text,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\b\d{1,3}\s*%", f"{target}%", text)

    return fn


def set_days(target: int) -> Callable[[str], str]:
    words = {7: "seven", 20: "twenty", 30: "thirty", 45: "forty-five", 60: "sixty"}
    word = words.get(target, str(target))

    def fn(text: str) -> str:
        return re.sub(
            rf"\b{_SPELLED}\s*\(\d+\)\s*((?:calendar|business|working)\s+)?(days?)",
            rf"{word} ({target}) \1\2",
            text,
            flags=re.IGNORECASE,
        )

    return fn


def set_years(target: int) -> Callable[[str], str]:
    words = {1: "one", 2: "two", 3: "three"}
    word = words.get(target, str(target))

    def fn(text: str) -> str:
        unit = "year" if target == 1 else "years"
        text = re.sub(
            rf"\b{_SPELLED}\s*\(\d+\)\s*years?",
            f"{word} ({target}) {unit}",
            text,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\b\d{1,3}\s*months?", f"{word} ({target}) {unit}", text)

    return fn


def replace_text(new_text: str) -> Callable[[str], str]:
    return lambda _text: new_text


# ---------------------------------------------------------------------------
# Structural transformations
# ---------------------------------------------------------------------------


def shuffle_clauses(seed: int = 1) -> Callable[[Package], None]:
    """Reverse-then-rotate clause order. Resolution must be order-independent."""

    def fn(pkg: Package) -> None:
        items = list(reversed(pkg.clauses))
        cut = seed % max(len(items), 1)
        pkg.clauses = items[cut:] + items[:cut]

    return fn


def set_metadata(key: str, value) -> Callable[[Package], None]:
    def fn(pkg: Package) -> None:
        pkg.metadata[key] = value

    return fn


def drop_fhwa_attachment(pkg: Package) -> None:
    """Remove the FHWA-1273 attachment file and its clause."""
    pkg.doc_files = [f for f in pkg.doc_files if "1273" not in f]
    pkg.clauses = [c for c in pkg.clauses if "1273" not in c.file_name]


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """One perturbation and its expected effect."""

    name: str
    kind: str                       # invariance | directional | applicability | precedence
    package: str                    # package directory name, e.g. "Pine_Grove"
    transform: Callable[[Package], None]
    rationale: str = ""
    target: str | None = None       # requirement expected to move
    expect_label: str | None = None
    expect_applicability: str | None = None
    expect_governing_contains: str | None = None
    # Requirements permitted to differ from baseline besides `target`.
    allow_drift: set[str] = field(default_factory=set)
