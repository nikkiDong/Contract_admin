#!/usr/bin/env python3
"""Contract Clause Risk Flagging - command line entry point.

    python run.py dev                      # analyse Development/ and self-score
    python run.py val                      # analyse Validation/ -> submission CSV
    python run.py all                      # both
    python run.py dev --llm bedrock        # enable LLM adjudication of residuals
    python run.py inspect Harbor_Crossing  # show parsed clauses for one package
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crf import applicability, precedence
from crf.evaluate import evaluate, load_labels
from crf.extract import discover_packages, load_package
from crf.pipeline import run_split, write_audit, write_submission
from crf.reference import ReferenceChecklist

DATA_ROOT = Path(__file__).parent / "Contract_Clause_Risk_Flagging"
CHECKLIST = DATA_ROOT / "References" / "Reference_Checklist.csv"
LABELS = DATA_ROOT / "Development" / "Development_Labels.csv"
OUT = Path(__file__).parent / "out"


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n  {text}\n{'=' * 78}")


def cmd_split(split: str, args) -> int:
    split_dir = DATA_ROOT / ("Development" if split == "dev" else "Validation")
    checklist = ReferenceChecklist.load(CHECKLIST)

    _banner(f"{split_dir.name.upper()}  ({len(checklist)} requirements)")

    findings, packages, adjudicator = run_split(
        split_dir, checklist, llm_provider=args.llm, llm_model=args.model
    )
    print(f"  packages analysed : {len(packages)}")
    print(f"  clauses extracted : {sum(len(p.clauses) for p in packages)}")
    print(f"  rows produced     : {len(findings)}")
    print(f"  adjudicator       : {adjudicator.stats()}")

    flags = [f for f in findings if f.predicted_label == "FLAG"]
    print(f"  flags raised      : {len(flags)} of {len(findings)} rows")
    by_sev: dict[str, int] = {}
    for f in flags:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
    for sev in ("Critical", "High", "Medium", "Low"):
        if sev in by_sev:
            print(f"      {sev:<9}: {by_sev[sev]}")

    suffix = "development" if split == "dev" else "validation"
    sub = write_submission(findings, OUT / f"submission_{suffix}.csv")
    aud = write_audit(findings, OUT / f"audit_{suffix}.csv")
    print(f"\n  submission -> {sub}")
    print(f"  audit      -> {aud}")

    if split == "dev" and LABELS.exists():
        report = evaluate(findings, load_labels(LABELS), packages, checklist)
        print(report.render())
        return 0 if not report.errors else 0
    return 0


def cmd_inspect(args) -> int:
    checklist = ReferenceChecklist.load(CHECKLIST)
    target = args.package.lower()
    matches = [
        p for split in ("Development", "Validation")
        for p in discover_packages(DATA_ROOT / split)
        if target in p.name.lower()
    ]
    if not matches:
        print(f"No package matching {args.package!r}.", file=sys.stderr)
        return 1

    for package_dir in matches:
        pkg = load_package(package_dir, checklist)
        _banner(f"{pkg.package_id} - {pkg.project_title}")
        print(f"  federal_aid={pkg.federal_aid}  baba={pkg.baba_applicable}  "
              f"subcontracting={pkg.subcontracting_planned}")
        print(f"  claim={pkg.claim_event}  delay={pkg.delay_event}  "
              f"changed_work={pkg.changed_work_event}")
        print(f"  addenda={pkg.issued_addenda or 'none'}  "
              f"value={pkg.contract_value}  fhwa_1273_attached="
              f"{pkg.has_fhwa_1273_attachment}")

        print("\n  RESOLVED CLAUSES")
        for req in checklist:
            found = precedence.candidates(pkg, req.requirement_id, checklist)
            if not found:
                continue
            governing, superseded, note = precedence.resolve(
                pkg, req.requirement_id, checklist
            )
            print(f"\n  {req.requirement_id}  {req.requirement_name}")
            for c in found:
                mark = "GOVERNS   " if c is governing else "superseded"
                print(f"    [{mark}] {c.file_name:<44} p{c.page}  {c.text[:110]}")
            if superseded:
                print(f"      -> {note}")

        print("\n  APPLICABILITY")
        for req in checklist:
            decision, reason = applicability.decide(pkg, req.requirement_id)
            print(f"    {req.requirement_id}  {decision:<15} {reason[:88]}")

        unresolved = [
            c for c in pkg.clauses if checklist.resolve_heading(c.heading) is None
        ]
        if unresolved:
            print("\n  UNRESOLVED HEADINGS (not mapped to any requirement)")
            for c in unresolved:
                print(f"    {c.file_name:<44} {c.heading}")
    return 0


def cmd_schema(args) -> int:
    """Validate generated submission CSVs against the schema contract."""
    from crf.conformance import check_submission

    paths = sorted(OUT.glob("submission_*.csv"))
    if not paths:
        print(f"No submission CSVs in {OUT}. Run `python run.py all` first.",
              file=sys.stderr)
        return 2
    print(f"checking: {', '.join(p.name for p in paths)}")
    report = check_submission(paths, DATA_ROOT)
    print(report.render())
    return 0 if report.ok else 1


def cmd_robustness(args) -> int:
    from crf.robustness import run_suite

    checklist = ReferenceChecklist.load(CHECKLIST)
    try:
        result = run_suite(DATA_ROOT, checklist, only=args.package)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.render())
    return 1 if result.failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run.py", description="Contract Clause Risk Flagging pipeline."
    )
    parser.add_argument(
        "command",
        choices=["dev", "val", "all", "inspect", "robustness", "schema"],
        help="dev = score against labels, val = produce submission, "
             "inspect = debug one package, robustness = perturbation suite, "
             "schema = validate submission field semantics",
    )
    parser.add_argument(
        "package", nargs="?",
        help="package name for `inspect`; case or kind filter for `robustness`",
    )
    parser.add_argument(
        "--llm", default="null", choices=["null", "bedrock", "anthropic"],
        help="adjudication provider for residual uncertainty (default: null = rules only)",
    )
    parser.add_argument("--model", default=None, help="override the model id")
    args = parser.parse_args()

    if args.command == "inspect":
        if not args.package:
            parser.error("inspect requires a package name")
        return cmd_inspect(args)
    if args.command == "robustness":
        return cmd_robustness(args)
    if args.command == "schema":
        return cmd_schema(args)
    if args.command == "all":
        rc = cmd_split("dev", args)
        return rc or cmd_split("val", args)
    return cmd_split(args.command, args)


if __name__ == "__main__":
    raise SystemExit(main())
