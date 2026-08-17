"""Local scorer implementing the weights in Evaluation/Evaluation_Criteria.csv.

    Applicability accuracy                20%
    Finding detection (precision+recall)  25%
    Cross-document precedence resolution  20%
    Semantic deviation discrimination     15%
    Evidence and citation correctness     15%
    Severity agreement                     5%

Two subsets are derived rather than hard-coded:

* **precedence subset** - rows where the package actually contains an Addendum
  carrying replacement text for that requirement. Derived from the parsed
  package, not from the label rationale, so the metric measures resolution
  behaviour rather than agreement with prose. These are the rows where reading
  the base document alone produces the wrong answer.
* **semantic subset** - rows whose rationale describes paraphrase, equivalent or
  reorganised wording. These are the false-positive traps.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import Finding, Package
from .precedence import is_addendum
from .reference import ReferenceChecklist

WEIGHTS = {
    "applicability_accuracy": 0.20,
    "finding_detection_f1": 0.25,
    "precedence_resolution": 0.20,
    "semantic_discrimination": 0.15,
    "evidence_correctness": 0.15,
    "severity_agreement": 0.05,
}

_SEMANTIC_RATIONALE = re.compile(
    r"paraphrase|equivalent|reorganized|reorganised|reordered|wording|capitalization"
    r"|capitalisation|numbered list|restated",
    re.IGNORECASE,
)


@dataclass
class Label:
    package_id: str
    requirement_id: str
    applicability: str
    label: str
    severity: str
    rationale: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.package_id, self.requirement_id)

    @property
    def is_semantic_case(self) -> bool:
        return bool(_SEMANTIC_RATIONALE.search(self.rationale))


def load_labels(path: str | Path) -> dict[tuple[str, str], Label]:
    labels: dict[tuple[str, str], Label] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            label = Label(
                package_id=row["Package_ID"].strip(),
                requirement_id=row["Requirement_ID"].strip(),
                applicability=row["Expected_Applicability"].strip(),
                label=row["Expected_Label"].strip(),
                severity=row["Expected_Severity"].strip(),
                rationale=row.get("Rationale", "").strip(),
            )
            labels[label.key] = label
    return labels


@dataclass
class Metric:
    name: str
    score: float
    detail: str
    weight: float

    @property
    def contribution(self) -> float:
        return self.score * self.weight


@dataclass
class Report:
    metrics: list[Metric] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    row_count: int = 0

    @property
    def weighted_score(self) -> float:
        return sum(m.contribution for m in self.metrics)

    def render(self) -> str:
        lines = [
            "",
            "=" * 78,
            "  LOCAL EVALUATION  (weights from Evaluation/Evaluation_Criteria.csv)",
            "=" * 78,
            f"  {'metric':<38} {'score':>8} {'weight':>8} {'contrib':>9}",
            "  " + "-" * 66,
        ]
        for m in self.metrics:
            lines.append(
                f"  {m.name:<38} {m.score * 100:>7.1f}% {m.weight * 100:>7.0f}% "
                f"{m.contribution * 100:>8.2f}"
            )
        lines += [
            "  " + "-" * 66,
            f"  {'WEIGHTED TOTAL':<38} {'':>8} {'':>8} "
            f"{self.weighted_score * 100:>8.2f}",
            "",
            f"  rows scored: {self.row_count}    mismatched rows: {len(self.errors)}",
        ]
        for m in self.metrics:
            lines.append(f"    - {m.name}: {m.detail}")
        if self.errors:
            lines += ["", "  MISMATCHES", "  " + "-" * 66]
            for e in self.errors:
                lines.append(
                    f"    {e['package_id']:<22} {e['requirement_id']:<7} "
                    f"expected {e['expected_applicability']}/{e['expected_label']}"
                    f"/{e['expected_severity']:<8} "
                    f"got {e['predicted_applicability']}/{e['predicted_label']}"
                    f"/{e['predicted_severity']}"
                )
                lines.append(f"      rule={e['rule_id']}  rationale={e['rationale']}")
        lines.append("=" * 78)
        return "\n".join(lines)


def _evidence_supports(finding: Finding, packages_by_id: dict[str, Package]) -> bool:
    """Evidence must be a verbatim span of the cited governing document."""
    if not finding.draft_evidence:
        return False
    package = packages_by_id.get(finding.document_id)
    if package is None:
        return False

    file_name = ""
    if "(" in finding.governing_document and ")" in finding.governing_document:
        file_name = finding.governing_document.rsplit("(", 1)[1].rstrip(")").strip()

    haystacks = [
        _norm(c.text) for c in package.clauses
        if not file_name or c.file_name == file_name
    ]
    needle = _norm(finding.draft_evidence)
    return any(needle in h for h in haystacks)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _addendum_revised_requirements(
    package: Package, checklist: ReferenceChecklist
) -> set[str]:
    """Requirement IDs for which this package contains Addendum replacement text."""
    out: set[str] = set()
    for clause in package.clauses:
        if is_addendum(clause) and clause.is_replacement and clause.revises_heading:
            req_id = checklist.resolve_heading(clause.revises_heading)
            if req_id:
                out.add(req_id)
    return out


def evaluate(
    findings: list[Finding],
    labels: dict[tuple[str, str], Label],
    packages: list[Package],
    checklist: ReferenceChecklist,
) -> Report:
    packages_by_id = {p.package_id: p for p in packages}
    report = Report()

    scored: list[tuple[Finding, Label]] = []
    for finding in findings:
        label = labels.get((finding.document_id, finding.requirement_id))
        if label is not None:
            scored.append((finding, label))
    report.row_count = len(scored)

    if not scored:
        report.metrics = [
            Metric(name, 0.0, "no labelled rows matched", weight)
            for name, weight in WEIGHTS.items()
        ]
        return report

    # 1. Applicability accuracy -------------------------------------------
    app_hits = sum(
        1 for f, l in scored if f.applicability_decision == l.applicability
    )
    report.metrics.append(
        Metric(
            "Applicability accuracy",
            app_hits / len(scored),
            f"{app_hits}/{len(scored)} applicability decisions correct",
            WEIGHTS["applicability_accuracy"],
        )
    )

    # 2. Finding detection: F1 on FLAG -------------------------------------
    tp = sum(1 for f, l in scored if f.predicted_label == "FLAG" and l.label == "FLAG")
    fp = sum(1 for f, l in scored if f.predicted_label == "FLAG" and l.label != "FLAG")
    fn = sum(1 for f, l in scored if f.predicted_label != "FLAG" and l.label == "FLAG")
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    report.metrics.append(
        Metric(
            "Finding detection (F1)",
            f1,
            f"precision {precision:.3f} ({tp} TP / {fp} FP), "
            f"recall {recall:.3f} ({fn} FN)",
            WEIGHTS["finding_detection_f1"],
        )
    )

    # 3. Cross-document precedence ----------------------------------------
    # Subset derived from the parsed packages: a requirement is a precedence
    # case when an Addendum in that package supplies replacement text for it.
    superseded_keys = {
        (pkg.package_id, req_id)
        for pkg in packages
        for req_id in _addendum_revised_requirements(pkg, checklist)
    }
    prec_rows = [(f, l) for f, l in scored if (f.document_id, f.requirement_id) in superseded_keys]
    if prec_rows:
        prec_hits = 0
        for f, l in prec_rows:
            label_ok = f.predicted_label == l.label
            cited_addendum = "addendum" in f.governing_document.lower()
            if label_ok and cited_addendum:
                prec_hits += 1
        prec_score = prec_hits / len(prec_rows)
        prec_detail = (
            f"{prec_hits}/{len(prec_rows)} Addendum-superseded rows resolved with the "
            "correct label and the Addendum cited as governing"
        )
    else:
        prec_score, prec_detail = 1.0, "no Addendum-superseded rows in this split"
    report.metrics.append(
        Metric("Cross-document precedence", prec_score, prec_detail,
               WEIGHTS["precedence_resolution"])
    )

    # 4. Semantic deviation discrimination ---------------------------------
    sem_rows = [(f, l) for f, l in scored if l.is_semantic_case]
    if sem_rows:
        sem_hits = sum(1 for f, l in sem_rows if f.predicted_label == l.label)
        sem_score = sem_hits / len(sem_rows)
        sem_detail = (
            f"{sem_hits}/{len(sem_rows)} paraphrase/equivalent-wording rows classified "
            "correctly (false-positive burden test)"
        )
    else:
        sem_score, sem_detail = 1.0, "no paraphrase rows in this split"
    report.metrics.append(
        Metric("Semantic deviation discrimination", sem_score, sem_detail,
               WEIGHTS["semantic_discrimination"])
    )

    # 5. Evidence and citation correctness ---------------------------------
    ev_rows = [
        (f, l) for f, l in scored
        if f.predicted_label == "FLAG"
        or (f.document_id, f.requirement_id) in superseded_keys
    ]
    if ev_rows:
        ev_hits = sum(1 for f, _ in ev_rows if _evidence_supports(f, packages_by_id))
        ev_score = ev_hits / len(ev_rows)
        ev_detail = (
            f"{ev_hits}/{len(ev_rows)} FLAG and precedence rows carry evidence that is a "
            "verbatim span of the cited governing document"
        )
    else:
        ev_score, ev_detail = 1.0, "no rows requiring evidence"
    report.metrics.append(
        Metric("Evidence and citation correctness", ev_score, ev_detail,
               WEIGHTS["evidence_correctness"])
    )

    # 6. Severity agreement -----------------------------------------------
    sev_hits = sum(1 for f, l in scored if f.severity == l.severity)
    report.metrics.append(
        Metric(
            "Severity agreement",
            sev_hits / len(scored),
            f"{sev_hits}/{len(scored)} severity labels match the challenge taxonomy",
            WEIGHTS["severity_agreement"],
        )
    )

    # Mismatch log ---------------------------------------------------------
    for f, l in scored:
        if (
            f.applicability_decision != l.applicability
            or f.predicted_label != l.label
            or f.severity != l.severity
        ):
            report.errors.append(
                {
                    "package_id": f.document_id,
                    "requirement_id": f.requirement_id,
                    "expected_applicability": l.applicability,
                    "predicted_applicability": f.applicability_decision,
                    "expected_label": l.label,
                    "predicted_label": f.predicted_label,
                    "expected_severity": l.severity,
                    "predicted_severity": f.severity,
                    "rule_id": f.rule_id,
                    "rationale": l.rationale,
                }
            )

    return report
