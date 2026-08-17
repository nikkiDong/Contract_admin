"""Robustness suite: run perturbation cases and report what breaks.

Each case is compared against a baseline run of the same package. Failures are
reported as one of:

  TARGET   the perturbed requirement did not reach the expected decision
  DRIFT    an untargeted requirement changed decision (collateral damage)
  GOVERN   precedence picked the wrong governing document
  DEGRADE  the decision survived but stopped resting on positive evidence

A DRIFT failure is usually more serious than a TARGET failure: it means a
detector is reading text belonging to a different requirement.

DEGRADE exists because label equality is not sufficient evidence of health. A
detector whose invariant test stops matching falls through to an `unclear` path
and returns NO_FLAG by default. On a compliant clause that produces the right
answer for the wrong reason, and it would produce the *wrong* answer the moment
the same clause carried a violation. Invariance cases therefore also require the
deciding rule to stay off the `unclear` / `no_value` / `missing` paths.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from . import perturb as P
from .extract import load_package
from .models import Finding, Package
from .pipeline import analyse_package
from .reference import ReferenceChecklist


# ---------------------------------------------------------------------------
# Case suite
# ---------------------------------------------------------------------------

def build_cases(checklist: ReferenceChecklist) -> list[P.Case]:
    """The perturbation suite.

    Package choices are deliberate: invariance cases run against packages whose
    baseline exercises the relevant detector, and directional cases run against
    packages where the target requirement is in scope and currently compliant
    (or currently violating, for repair cases).
    """
    cases: list[P.Case] = []

    # -- invariance: number formatting ----------------------------------
    for pkg_name in ("Harbor_Crossing", "Stone_Creek", "Riverbend", "Maple_Ridge"):
        cases.append(
            P.Case(
                name=f"strip_parenthetical_digits[{pkg_name}]",
                kind="invariance",
                package=pkg_name,
                transform=lambda p: P.edit_all_text(p, P.strip_parenthetical_digits),
                rationale="Documents that spell numbers without the digit form in "
                          "parentheses. Detectors reading only digits lose their input.",
            )
        )
        cases.append(
            P.Case(
                name=f"digits_only[{pkg_name}]",
                kind="invariance",
                package=pkg_name,
                transform=lambda p: P.edit_all_text(p, P.digits_only),
                rationale="Documents that give digits without the spelled form.",
            )
        )

    # -- invariance: equivalent units -----------------------------------
    cases.append(
        P.Case(
            name="years_to_months[Riverbend]",
            kind="invariance",
            package="Riverbend",
            transform=lambda p: P.edit_all_text(p, P.years_to_months),
            rationale="Retention stated as 36 months rather than three years. "
                      "Harbor Crossing already proves the reverse direction parses.",
        )
    )

    # -- invariance: phrasing -------------------------------------------
    for pkg_name in ("Pine_Grove", "Stone_Creek", "Riverbend"):
        cases.append(
            P.Case(
                name=f"reorder_sentences[{pkg_name}]",
                kind="invariance",
                package=pkg_name,
                transform=lambda p: P.edit_all_text(p, P.reorder_sentences),
                rationale="Clause sentences in a different order.",
            )
        )
        cases.append(
            P.Case(
                name=f"synonym_swap[{pkg_name}]",
                kind="invariance",
                package=pkg_name,
                transform=lambda p: P.edit_all_text(p, P.synonym_swap),
                rationale="shall/must, furnish/provide, required/mandatory and similar "
                          "drafting synonyms that do not change the obligation.",
            )
        )
        cases.append(
            P.Case(
                name=f"prepend_boilerplate[{pkg_name}]",
                kind="invariance",
                package=pkg_name,
                transform=lambda p: P.edit_all_text(p, P.prepend_boilerplate),
                rationale="Non-operative preamble ahead of the operative sentence.",
            )
        )

    cases.append(
        P.Case(
            name="collapse_whitespace[Northfield]",
            kind="invariance",
            package="Northfield",
            transform=lambda p: P.edit_all_text(p, P.collapse_whitespace),
            rationale="Irregular spacing from a different PDF text layer.",
        )
    )
    cases.append(
        P.Case(
            name="heading_repunctuate[Northfield]",
            kind="invariance",
            package="Northfield",
            transform=lambda p: P.edit_all_headings(p, P.heading_repunctuate),
            rationale="Headings in upper case with '/' replaced by '-'. Tests heading "
                      "normalisation rather than detector logic.",
        )
    )

    # -- invariance: structural ------------------------------------------
    for pkg_name in ("Harbor_Crossing", "Stone_Creek"):
        cases.append(
            P.Case(
                name=f"shuffle_clauses[{pkg_name}]",
                kind="invariance",
                package=pkg_name,
                transform=P.shuffle_clauses(3),
                rationale="Clause iteration order changed. Precedence resolution must "
                          "not depend on document traversal order.",
            )
        )

    # -- directional: numeric invariants --------------------------------
    cases.append(
        P.Case(
            name="bid_guaranty 10%->5% [Stone_Creek]",
            kind="directional",
            package="Stone_Creek",
            target="CC-02",
            expect_label="FLAG",
            transform=lambda p: P.edit_text(p, checklist, "CC-02", P.set_percent(5)),
            rationale="Reducing bid security below the reference 10% must flag.",
        )
    )
    cases.append(
        P.Case(
            name="bid_guaranty 5%->10% repair [Pine_Grove]",
            kind="directional",
            package="Pine_Grove",
            target="CC-02",
            expect_label="NO_FLAG",
            transform=lambda p: P.edit_text(p, checklist, "CC-02", P.set_percent(10)),
            rationale="Repairing a known violation must clear the flag. Guards against "
                      "a detector that flags on package identity rather than content.",
        )
    )
    cases.append(
        P.Case(
            name="bonds 100%->75% [Riverbend]",
            kind="directional",
            package="Riverbend",
            target="CC-04",
            expect_label="FLAG",
            transform=lambda p: P.edit_text(p, checklist, "CC-04", P.set_percent(75)),
            rationale="Reducing bond coverage below 100% must flag.",
        )
    )
    cases.append(
        P.Case(
            name="change_notice 7->30 days [Maple_Ridge]",
            kind="directional",
            package="Maple_Ridge",
            target="CC-12",
            expect_label="FLAG",
            transform=lambda p: P.edit_text(
                p, checklist, "CC-12",
                P.replace_text(
                    "Change Notification. Written follow-up documentation may be "
                    "submitted within thirty (30) calendar days after the alleged change."
                ),
            ),
            rationale="Extending the 7-day written follow-up must flag.",
        )
    )
    cases.append(
        P.Case(
            name="retention 3y->1y [Riverbend]",
            kind="directional",
            package="Riverbend",
            target="CC-13",
            expect_label="FLAG",
            transform=lambda p: P.edit_text(p, checklist, "CC-13", P.set_years(1)),
            rationale="Shortening record retention below three years must flag.",
        )
    )
    cases.append(
        P.Case(
            name="retention 1y->3y repair [Maple_Ridge]",
            kind="directional",
            package="Maple_Ridge",
            target="CC-13",
            expect_label="NO_FLAG",
            transform=lambda p: P.edit_text(
                p, checklist, "CC-13",
                P.replace_text(
                    "Right to Audit. Relevant prime and subcontract records shall remain "
                    "available for audit and be retained for three (3) years after final "
                    "payment."
                ),
            ),
            rationale="Restoring full audit scope and retention must clear the flag.",
        )
    )
    cases.append(
        P.Case(
            name="execution 20->45 days [Riverbend]",
            kind="directional",
            package="Riverbend",
            target="CC-05",
            expect_label="FLAG",
            transform=lambda p: P.edit_text(
                p, checklist, "CC-05",
                P.replace_text(
                    "Contract Execution and Insurance. Execution documents may be "
                    "returned within forty-five (45) calendar days."
                ),
            ),
            rationale="Extending the execution deadline past 20 days must flag.",
        )
    )
    cases.append(
        P.Case(
            name="subletting 50%->80% [Harbor_Crossing]",
            kind="directional",
            package="Harbor_Crossing",
            target="CC-14",
            expect_label="FLAG",
            transform=lambda p: P.edit_text(
                p, checklist, "CC-14",
                P.replace_text(
                    "Subcontracting. Up to eighty percent (80%) of the work may be "
                    "subcontracted without Department approval."
                ),
            ),
            rationale="Allowing the prime to self-perform under 50% must flag.",
        )
    )

    # -- directional: modal / polarity inversions -----------------------
    cases.append(
        P.Case(
            name="certification made optional [Riverbend]",
            kind="directional",
            package="Riverbend",
            target="CC-03",
            expect_label="FLAG",
            transform=lambda p: P.edit_text(
                p, checklist, "CC-03",
                P.replace_text(
                    "Non-Collusive Bidding Certification. Submission of the "
                    "certification is encouraged but optional."
                ),
            ),
            rationale="Making a required certification optional must flag.",
        )
    )
    cases.append(
        P.Case(
            name="BABA disclaimed [Stone_Creek]",
            kind="directional",
            package="Stone_Creek",
            target="CC-09",
            expect_label="FLAG",
            transform=lambda p: P.edit_text(
                p, checklist, "CC-09",
                P.replace_text(
                    "Buy America / BABA. Domestic-content requirements do not apply to "
                    "this project."
                ),
            ),
            rationale="Contradicting a stated applicability rule must flag Critical.",
        )
    )
    cases.append(
        P.Case(
            name="oral direction made binding [Northfield]",
            kind="directional",
            package="Northfield",
            target="CC-11",
            expect_label="FLAG",
            transform=lambda p: P.edit_text(
                p, checklist, "CC-11",
                P.replace_text(
                    "Contract Changes. Oral direction from an authorized representative "
                    "immediately modifies scope, price, or time even if never reduced "
                    "to writing."
                ),
            ),
            rationale="The CC-11 polarity pair. This is the case that caught the "
                      "original misplaced negative lookahead.",
        )
    )
    cases.append(
        P.Case(
            name="oral direction repaired [Stone_Creek]",
            kind="directional",
            package="Stone_Creek",
            target="CC-11",
            expect_label="NO_FLAG",
            transform=lambda p: P.edit_text(
                p, checklist, "CC-11",
                P.replace_text(
                    "Contract Changes. Scope, price, or time changes require the "
                    "documented written process; oral direction alone does not modify "
                    "the contract."
                ),
            ),
            rationale="The other half of the CC-11 polarity pair.",
        )
    )
    cases.append(
        P.Case(
            name="precedence hierarchy inverted [Riverbend]",
            kind="directional",
            package="Riverbend",
            target="CC-10",
            expect_label="FLAG",
            transform=lambda p: P.edit_text(
                p, checklist, "CC-10",
                P.replace_text(
                    "Coordination. In a conflict, Standard Specifications govern over "
                    "Special Provisions and General Notices."
                ),
            ),
            rationale="Reversing the DelDOT 105.6 ladder must flag.",
        )
    )
    cases.append(
        P.Case(
            name="automatic time extension [Stone_Creek]",
            kind="directional",
            package="Stone_Creek",
            target="CC-16",
            expect_label="FLAG",
            transform=lambda p: P.edit_text(
                p, checklist, "CC-16",
                P.replace_text(
                    "Time Extensions. Any delay automatically extends contract time for "
                    "the length of the delay without further demonstration or timely "
                    "supporting notice."
                ),
            ),
            rationale="Automatic extensions must flag.",
        )
    )
    cases.append(
        P.Case(
            name="flat LD rate [Harbor_Crossing]",
            kind="directional",
            package="Harbor_Crossing",
            target="CC-17",
            expect_label="FLAG",
            transform=lambda p: P.edit_text(
                p, checklist, "CC-17",
                P.replace_text(
                    "Liquidated Damages. A fixed rate of $10,000 per calendar day "
                    "applies to every contract regardless of contract value or "
                    "governing schedule."
                ),
            ),
            rationale="A universal invented flat daily rate must flag.",
        )
    )
    cases.append(
        P.Case(
            name="FHWA-1273 attachment removed [Stone_Creek]",
            kind="directional",
            package="Stone_Creek",
            target="CC-01",
            expect_label="FLAG",
            transform=P.drop_fhwa_attachment,
            rationale="Physical inclusion is an absence test: removing the attachment "
                      "file must flag even though no clause text changed.",
        )
    )

    # -- applicability: metadata gates ----------------------------------
    cases.append(
        P.Case(
            name="federal_aid Yes->No [Northfield]",
            kind="applicability",
            package="Northfield",
            target="CC-01",
            expect_applicability="DOES_NOT_APPLY",
            expect_label="NO_FLAG",
            transform=P.set_metadata("federal_aid", "No"),
            rationale="Flipping the CC-01 gate must take it out of scope.",
            allow_drift=set(),
        )
    )
    cases.append(
        P.Case(
            name="baba Yes->No [Northfield]",
            kind="applicability",
            package="Northfield",
            target="CC-09",
            expect_applicability="DOES_NOT_APPLY",
            expect_label="NO_FLAG",
            transform=P.set_metadata("buy_america_baba_applicable", "No"),
            rationale="Northfield's CC-09 is a Critical flag at baseline. Taking it out "
                      "of scope must suppress the flag entirely.",
        )
    )
    cases.append(
        P.Case(
            name="subcontracting No->Yes [Northfield]",
            kind="applicability",
            package="Northfield",
            target="CC-14",
            expect_applicability="APPLIES",
            expect_label="FLAG",
            transform=P.set_metadata("subcontracting_planned", "Yes"),
            rationale="Bringing CC-14 into scope where the package has no subletting "
                      "clause must raise a missing-provision flag.",
        )
    )
    cases.append(
        P.Case(
            name="delay Yes->No [Riverbend]",
            kind="applicability",
            package="Riverbend",
            target="CC-16",
            expect_applicability="DOES_NOT_APPLY",
            expect_label="NO_FLAG",
            transform=P.set_metadata("delay_event", "No"),
            rationale="Riverbend's CC-16 flags at baseline; removing the delay scenario "
                      "must suppress it.",
        )
    )
    cases.append(
        P.Case(
            name="addenda cleared [Maple_Ridge]",
            kind="applicability",
            package="Maple_Ridge",
            target="CC-08",
            expect_applicability="DOES_NOT_APPLY",
            expect_label="NO_FLAG",
            transform=P.set_metadata("issued_addenda", []),
            rationale="Maple Ridge's CC-08 flags at baseline; with no issued addenda "
                      "there is no currency obligation to test.",
        )
    )

    # -- precedence: supersession chains --------------------------------
    cases.append(
        P.Case(
            name="later addendum re-revises to violation [Harbor_Crossing]",
            kind="precedence",
            package="Harbor_Crossing",
            target="CC-04",
            expect_label="FLAG",
            expect_governing_contains="Addendum D",
            transform=lambda p: P.add_addendum(
                p, checklist, "CC-04", "D",
                "Performance and Payment Bonds. Bond coverage equal to fifty percent "
                "(50%) of the contract price is sufficient.",
            ),
            rationale="Harbor Crossing CC-04 is already revised by Addendum B to 100%. "
                      "A later Addendum D reducing it to 50% must govern and flag. "
                      "Tests latest-wins, not any-addendum-wins.",
        )
    )
    cases.append(
        P.Case(
            name="later addendum repairs violation [Northfield]",
            kind="precedence",
            package="Northfield",
            target="CC-12",
            expect_label="NO_FLAG",
            expect_governing_contains="Addendum A",
            transform=lambda p: P.add_addendum(
                p, checklist, "CC-12", "A",
                "Change Notification. Alleged changes require timely written notice and "
                "the reference follow-up documentation within the stated period.",
            ),
            rationale="Northfield CC-12 flags at baseline (30 days). An Addendum "
                      "restoring the reference workflow must clear it and be cited as "
                      "governing.",
            allow_drift={"CC-08"},  # adding an addendum brings CC-08 into scope
        )
    )
    cases.append(
        P.Case(
            name="addendum revises an unrelated provision [Pine_Grove]",
            kind="precedence",
            package="Pine_Grove",
            target="CC-16",
            expect_label="NO_FLAG",
            transform=lambda p: P.add_addendum(
                p, checklist, "CC-16", "A",
                "Time Extensions. Extensions require the reference conditions, timely "
                "support, and demonstrated effect on contract time; they are not "
                "automatic for every delay.",
            ),
            rationale="An addendum touching CC-16 must not disturb Pine Grove's six "
                      "baseline flags on other requirements.",
            allow_drift={"CC-08"},
        )
    )

    return cases


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class Failure:
    case: str
    kind: str
    mode: str           # TARGET | DRIFT | GOVERN | DEGRADE
    requirement: str
    detail: str


# Rule-id suffixes meaning "no invariant matched; defaulting".
_FALLTHROUGH = ("unclear", "no_value", "missing", "none")


def _is_fallthrough(rule_id: str) -> bool:
    return any(rule_id.endswith(s) or f".{s}" in rule_id for s in _FALLTHROUGH)


@dataclass
class Result:
    total: int = 0
    passed: int = 0
    failures: list[Failure] = field(default_factory=list)
    by_kind: dict[str, tuple[int, int]] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            "",
            "=" * 78,
            "  ROBUSTNESS SUITE  (perturbations with answers known by construction)",
            "=" * 78,
        ]
        for kind in ("invariance", "directional", "applicability", "precedence"):
            if kind in self.by_kind:
                ok, tot = self.by_kind[kind]
                bar = "PASS" if ok == tot else "FAIL"
                lines.append(f"  {kind:<16} {ok:>3}/{tot:<3}  {bar}")
        lines += [
            "  " + "-" * 66,
            f"  {'TOTAL':<16} {self.passed:>3}/{self.total:<3}  "
            f"{'ALL PASS' if not self.failures else str(len(self.failures)) + ' FAILING'}",
        ]

        if self.failures:
            modes: dict[str, int] = {}
            for f in self.failures:
                modes[f.mode] = modes.get(f.mode, 0) + 1
            lines += ["", "  failure modes: " + ", ".join(
                f"{k}={v}" for k, v in sorted(modes.items())
            )]
            lines += ["", "  FAILURES", "  " + "-" * 66]
            for f in self.failures:
                lines.append(f"    [{f.mode}] {f.case}")
                lines.append(f"           {f.requirement}: {f.detail}")
        lines.append("=" * 78)
        return "\n".join(lines)


def _index(findings: list[Finding]) -> dict[str, Finding]:
    return {f.requirement_id: f for f in findings}


def run_suite(
    data_root: str | Path,
    checklist: ReferenceChecklist,
    only: str | None = None,
) -> Result:
    """Execute every case and compare against its package baseline."""
    root = Path(data_root)
    cases = build_cases(checklist)
    if only:
        cases = [c for c in cases if only.lower() in c.name.lower()
                 or only.lower() == c.kind.lower()]
        if not cases:
            raise ValueError(
                f"filter {only!r} matched no cases. Valid kinds: invariance, "
                f"directional, applicability, precedence."
            )

    baselines: dict[str, tuple[Package, dict[str, Finding]]] = {}
    result = Result()

    for case in cases:
        # Locate and cache the unperturbed package + baseline decisions.
        if case.package not in baselines:
            matches = [
                d for split in ("Development", "Validation")
                for d in (root / split).iterdir()
                if d.is_dir() and d.name == case.package
            ]
            if not matches:
                result.total += 1
                result.failures.append(
                    Failure(case.name, case.kind, "TARGET", "-",
                            f"package directory {case.package!r} not found")
                )
                continue
            pkg = load_package(matches[0], checklist)
            baselines[case.package] = (pkg, _index(analyse_package(pkg, checklist)))

        base_pkg, base = baselines[case.package]

        perturbed = copy.deepcopy(base_pkg)
        case.transform(perturbed)
        after = _index(analyse_package(perturbed, checklist))

        result.total += 1
        ok, tot = result.by_kind.get(case.kind, (0, 0))
        case_failures: list[Failure] = []

        # -- target expectations -------------------------------------
        if case.target:
            got = after.get(case.target)
            if got is None:
                case_failures.append(
                    Failure(case.name, case.kind, "TARGET", case.target,
                            "no row produced for target requirement")
                )
            else:
                if case.expect_label and got.predicted_label != case.expect_label:
                    case_failures.append(
                        Failure(case.name, case.kind, "TARGET", case.target,
                                f"expected {case.expect_label}, got "
                                f"{got.predicted_label} (rule={got.rule_id})")
                    )
                if (case.expect_applicability
                        and got.applicability_decision != case.expect_applicability):
                    case_failures.append(
                        Failure(case.name, case.kind, "TARGET", case.target,
                                f"expected {case.expect_applicability}, got "
                                f"{got.applicability_decision}")
                    )
                if (case.expect_governing_contains
                        and case.expect_governing_contains not in got.governing_document):
                    case_failures.append(
                        Failure(case.name, case.kind, "GOVERN", case.target,
                                f"expected governing document containing "
                                f"{case.expect_governing_contains!r}, got "
                                f"{got.governing_document!r}")
                    )

        # -- collateral damage on everything else --------------------
        exempt = set(case.allow_drift) | ({case.target} if case.target else set())
        for req_id, before_row in base.items():
            if req_id in exempt:
                continue
            now = after.get(req_id)
            if now is None:
                case_failures.append(
                    Failure(case.name, case.kind, "DRIFT", req_id, "row disappeared")
                )
                continue
            if (now.predicted_label != before_row.predicted_label
                    or now.applicability_decision != before_row.applicability_decision
                    or now.severity != before_row.severity):
                case_failures.append(
                    Failure(
                        case.name, case.kind, "DRIFT", req_id,
                        f"{before_row.applicability_decision}/"
                        f"{before_row.predicted_label}/{before_row.severity} -> "
                        f"{now.applicability_decision}/{now.predicted_label}/"
                        f"{now.severity} (rule={now.rule_id})",
                    )
                )
            elif (case.kind == "invariance"
                  and now.applicability_decision == "APPLIES"
                  and _is_fallthrough(now.rule_id)
                  and not _is_fallthrough(before_row.rule_id)):
                # Same answer, but it stopped resting on a matched invariant.
                case_failures.append(
                    Failure(
                        case.name, case.kind, "DEGRADE", req_id,
                        f"decided by {before_row.rule_id} -> {now.rule_id}; label "
                        f"{now.predicted_label} is now a default, not a verified result",
                    )
                )

        if case_failures:
            result.failures.extend(case_failures)
            result.by_kind[case.kind] = (ok, tot + 1)
        else:
            result.passed += 1
            result.by_kind[case.kind] = (ok + 1, tot + 1)

    return result
