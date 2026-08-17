"""Submission schema conformance checks.

Encodes the challenge schema contract as assertions so it cannot regress
silently while detectors are being changed.

The four field-semantics rules come from the challenge clarification:

    document_id        = package ID (never an individual PDF)
    requirement_id     = CC requirement ID
    reference_id       = the same CC requirement ID
    reference_location = supporting standard / statute / section

with the corollary that the specific PDF, when relevant, is identified in
`governing_document` and located by `draft_location`.

Also checked: field names and order against Submission_Schema.csv, the
one-row-per-package-x-requirement grain, controlled vocabularies, the
confidence format, and the rule that a DOES_NOT_APPLY row can never carry a FLAG.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

APPLICABILITY_VOCAB = {"APPLIES", "DOES_NOT_APPLY"}
LABEL_VOCAB = {"FLAG", "NO_FLAG"}
SEVERITY_VOCAB = {"Critical", "High", "Medium", "Low", "Info"}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class ConformanceReport:
    checks: list[Check] = field(default_factory=list)
    row_count: int = 0

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def render(self) -> str:
        lines = [
            "",
            "=" * 78,
            "  SUBMISSION SCHEMA CONFORMANCE",
            "=" * 78,
        ]
        for c in self.checks:
            lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}")
            lines.append(f"         {c.detail}")
        lines += [
            "  " + "-" * 66,
            f"  rows checked: {self.row_count}",
            f"  RESULT: {'CONFORMS' if self.ok else 'NONCONFORMING'}",
            "=" * 78,
        ]
        return "\n".join(lines)


def _package_ids(data_root: Path) -> set[str]:
    ids = set()
    for meta in data_root.glob("*/*/Project_Metadata.json"):
        try:
            ids.add(json.loads(meta.read_text(encoding="utf-8"))["package_id"].strip())
        except (KeyError, json.JSONDecodeError):
            continue
    return ids


def check_submission(
    submission_paths: list[str | Path],
    data_root: str | Path,
    expect_full_grid: bool = True,
) -> ConformanceReport:
    """Validate one or more submission CSVs against the schema contract."""
    root = Path(data_root)
    report = ConformanceReport()

    schema_fields = [
        r["field"].strip()
        for r in csv.DictReader(
            open(root / "Submission" / "Submission_Schema.csv", encoding="utf-8-sig")
        )
    ]
    checklist_rows = list(
        csv.DictReader(
            open(root / "References" / "Reference_Checklist.csv", encoding="utf-8-sig")
        )
    )
    cc_ids = [r["Requirement_ID"].strip() for r in checklist_rows]
    authority = {
        r["Requirement_ID"].strip(): (
            r["Reference_Source"].strip(),
            r["Section"].strip(),
        )
        for r in checklist_rows
    }
    package_ids = _package_ids(root)

    rows: list[dict] = []
    header_problems: list[str] = []
    for path in submission_paths:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != schema_fields:
                header_problems.append(
                    f"{Path(path).name}: header {reader.fieldnames} != schema {schema_fields}"
                )
            rows.extend(reader)
    report.row_count = len(rows)

    # -- header ---------------------------------------------------------
    report.checks.append(
        Check(
            "field names and order match Submission_Schema.csv",
            not header_problems,
            "; ".join(header_problems)
            or f"all {len(schema_fields)} fields present in schema order",
        )
    )

    if not rows:
        report.checks.append(Check("rows present", False, "no rows found"))
        return report

    # -- rule 1: document_id is a package ID -----------------------------
    unknown = sorted({r["document_id"] for r in rows} - package_ids)
    pdfish = sorted({r["document_id"] for r in rows if ".pdf" in r["document_id"].lower()})
    report.checks.append(
        Check(
            "document_id = package ID, never an individual PDF",
            not unknown and not pdfish,
            (f"unknown package ids: {unknown}; pdf-like: {pdfish}"
             if (unknown or pdfish)
             else f"{len(set(r['document_id'] for r in rows))} distinct package ids, "
                  "all present in Project_Metadata.json"),
        )
    )

    # -- rule 2: requirement_id is a CC ID -------------------------------
    bad_req = sorted({r["requirement_id"] for r in rows} - set(cc_ids))
    report.checks.append(
        Check(
            "requirement_id = CC requirement ID",
            not bad_req,
            f"invalid: {bad_req}" if bad_req
            else f"all values within {cc_ids[0]}..{cc_ids[-1]}",
        )
    )

    # -- rule 3: reference_id repeats requirement_id ----------------------
    mismatched = [
        (r["document_id"], r["requirement_id"], r["reference_id"])
        for r in rows
        if r["reference_id"] != r["requirement_id"]
    ]
    report.checks.append(
        Check(
            "reference_id repeats the CC requirement ID",
            not mismatched,
            f"{len(mismatched)} mismatches, e.g. {mismatched[:3]}" if mismatched
            else "reference_id == requirement_id on every row",
        )
    )

    # -- rule 4: reference_location is the external authority -------------
    problems: list[str] = []
    for r in rows:
        loc = r["reference_location"].strip()
        if not loc:
            problems.append(f"{r['document_id']}/{r['requirement_id']}: blank")
            continue
        if ".pdf" in loc.lower():
            problems.append(f"{r['document_id']}/{r['requirement_id']}: names a package PDF")
            continue
        source, section = authority.get(r["requirement_id"], ("", ""))
        if source and source not in loc:
            problems.append(f"{r['requirement_id']}: missing source {source!r}")
        elif section and section not in loc:
            problems.append(f"{r['requirement_id']}: missing section {section!r}")
    report.checks.append(
        Check(
            "reference_location = supporting standard/statute/section",
            not problems,
            f"{len(problems)} problems, e.g. {problems[:3]}" if problems
            else "every row cites the checklist Reference_Source and Section; "
                 "no row points at a package PDF",
        )
    )

    # -- corollary: the PDF is identified in governing_document ----------
    applies = [r for r in rows if r["applicability_decision"] == "APPLIES"]
    located = [r for r in applies if ".pdf" in r["governing_document"].lower()]
    unlocated = [
        f"{r['document_id']}/{r['requirement_id']}"
        for r in applies
        if ".pdf" not in r["governing_document"].lower()
        and "not located" not in r["governing_document"].lower()
    ]
    report.checks.append(
        Check(
            "governing_document identifies the specific PDF on in-scope rows",
            not unlocated,
            f"{len(unlocated)} in-scope rows name no PDF: {unlocated[:3]}" if unlocated
            else f"{len(located)}/{len(applies)} APPLIES rows name a package PDF",
        )
    )

    # -- grain ----------------------------------------------------------
    counts = Counter((r["document_id"], r["requirement_id"]) for r in rows)
    dupes = [k for k, v in counts.items() if v > 1]
    detail = f"{len(counts)} unique package x requirement pairs"
    grain_ok = not dupes
    if expect_full_grid:
        present_packages = {r["document_id"] for r in rows}
        missing = [
            (p, cc) for p in sorted(present_packages) for cc in cc_ids
            if (p, cc) not in counts
        ]
        grain_ok = grain_ok and not missing
        detail = (
            f"{len(present_packages)} packages x {len(cc_ids)} requirements = "
            f"{len(present_packages) * len(cc_ids)} expected, {len(rows)} present; "
            f"{len(dupes)} duplicates, {len(missing)} missing"
        )
    report.checks.append(
        Check("exactly one row per package x requirement", grain_ok, detail)
    )

    # -- vocabularies and formats ---------------------------------------
    vocab_problems = []
    bad_app = {r["applicability_decision"] for r in rows} - APPLICABILITY_VOCAB
    bad_lab = {r["predicted_label"] for r in rows} - LABEL_VOCAB
    bad_sev = {r["severity"] for r in rows} - SEVERITY_VOCAB
    if bad_app:
        vocab_problems.append(f"applicability_decision: {sorted(bad_app)}")
    if bad_lab:
        vocab_problems.append(f"predicted_label: {sorted(bad_lab)}")
    if bad_sev:
        vocab_problems.append(f"severity: {sorted(bad_sev)}")
    report.checks.append(
        Check(
            "controlled vocabularies",
            not vocab_problems,
            "; ".join(vocab_problems) or
            "applicability_decision, predicted_label and severity all within the "
            "challenge taxonomy",
        )
    )

    bad_conf = []
    for r in rows:
        try:
            value = float(r["confidence"])
            if not 0.0 <= value <= 1.0:
                bad_conf.append(r["confidence"])
        except ValueError:
            bad_conf.append(r["confidence"])
    report.checks.append(
        Check(
            "confidence is a number in 0.00-1.00",
            not bad_conf,
            f"{len(bad_conf)} invalid, e.g. {bad_conf[:3]}" if bad_conf
            else "all values numeric and within range",
        )
    )

    # -- flag discipline -------------------------------------------------
    contradictions = [
        f"{r['document_id']}/{r['requirement_id']}"
        for r in rows
        if r["applicability_decision"] == "DOES_NOT_APPLY"
        and r["predicted_label"] != "NO_FLAG"
    ]
    report.checks.append(
        Check(
            "DOES_NOT_APPLY rows are always NO_FLAG",
            not contradictions,
            f"{len(contradictions)} contradictions: {contradictions[:3]}"
            if contradictions else
            f"all {sum(1 for r in rows if r['applicability_decision'] == 'DOES_NOT_APPLY')} "
            "out-of-scope rows carry NO_FLAG",
        )
    )

    # -- evidence on flags ----------------------------------------------
    missing_evidence = [
        f"{r['document_id']}/{r['requirement_id']}"
        for r in rows
        if r["predicted_label"] == "FLAG"
        and (not r["draft_location"].strip() or not r["draft_evidence"].strip())
    ]
    report.checks.append(
        Check(
            "FLAG rows carry draft_location and draft_evidence",
            not missing_evidence,
            f"{len(missing_evidence)} incomplete: {missing_evidence[:3]}"
            if missing_evidence else
            f"all {sum(1 for r in rows if r['predicted_label'] == 'FLAG')} flagged rows "
            "cite a location and evidence span",
        )
    )

    return report
