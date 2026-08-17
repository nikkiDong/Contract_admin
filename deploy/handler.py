"""Lambda entry point for the contract clause risk flagging service.

Routes (REST API, Lambda proxy integration):

    POST /analyze              {"package_id": "VAL-OAK-HOLLOW"}
    GET  /findings/{id}        -> all rows for one package
    GET  /findings/{id}?flags_only=true
    GET  /health

The pipeline itself runs unchanged from `crf/`. This module only moves data:
S3 -> /tmp -> pipeline -> DynamoDB, and DynamoDB -> JSON.

PDF text extraction uses pypdf here (no poppler binary in the Lambda runtime).
That backend is verified byte-identical to the local `pdftotext` path on the
supplied packages, so the deployed service and the CLI produce the same rows.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import boto3
# Explicit: `import boto3` alone does not guarantee this submodule is loaded.
from boto3.dynamodb.conditions import Key

# Force the pure-Python backend before crf.extract reads the setting.
os.environ.setdefault("CRF_PDF_BACKEND", "pypdf")

from crf.extract import load_package  # noqa: E402
from crf.llm import Adjudicator, build_provider  # noqa: E402
from crf.pipeline import analyse_package  # noqa: E402
from crf.reference import ReferenceChecklist  # noqa: E402

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

BUCKET = os.environ["PACKAGES_BUCKET"]
TABLE = os.environ["FINDINGS_TABLE"]
PACKAGE_PREFIX = os.environ.get("PACKAGE_PREFIX", "packages")
CHECKLIST_KEY = os.environ.get("CHECKLIST_KEY", "reference/Reference_Checklist.csv")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "null")

_s3 = boto3.client("s3")
_table = boto3.resource("dynamodb").Table(TABLE)

# Cached across warm invocations.
_checklist: ReferenceChecklist | None = None


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json",
            # Responses are data-only; no HTML is rendered from them.
            "x-content-type-options": "nosniff",
            "cache-control": "no-store",
        },
        "body": json.dumps(body, default=str),
    }


def _load_checklist() -> ReferenceChecklist:
    global _checklist
    if _checklist is None:
        local = Path(tempfile.gettempdir()) / "Reference_Checklist.csv"
        if not local.exists():
            _s3.download_file(BUCKET, CHECKLIST_KEY, str(local))
        _checklist = ReferenceChecklist.load(local)
        LOG.info("loaded checklist with %d requirements", len(_checklist))
    return _checklist


def _safe_package_id(raw: str) -> str:
    """Reject anything that could escape the intended S3 prefix or /tmp path."""
    value = (raw or "").strip()
    if not value or len(value) > 128:
        raise ValueError("package_id must be 1-128 characters")
    if not all(c.isalnum() or c in "-_" for c in value):
        raise ValueError("package_id may contain only letters, digits, '-' and '_'")
    return value


def _download_package(package_id: str, dest: Path) -> int:
    """Copy s3://BUCKET/PACKAGE_PREFIX/<package_id>/** into `dest`."""
    prefix = f"{PACKAGE_PREFIX}/{package_id}/"
    paginator = _s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            relative = key[len(prefix):]
            # Defence in depth: the prefix is already constrained, but never let a
            # crafted key write outside dest.
            target = (dest / relative).resolve()
            if not str(target).startswith(str(dest.resolve())):
                LOG.warning("skipping key outside destination: %s", key)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            _s3.download_file(BUCKET, key, str(target))
            count += 1
    return count


def _persist(findings) -> None:
    with _table.batch_writer() as batch:
        for finding in findings:
            item = finding.to_audit_row()
            item["document_id"] = finding.document_id
            item["requirement_id"] = finding.requirement_id
            batch.put_item(Item={k: ("" if v is None else str(v)) for k, v in item.items()})


def handle_analyze(body: dict) -> dict:
    package_id = _safe_package_id(body.get("package_id", ""))
    checklist = _load_checklist()

    workdir = Path(tempfile.mkdtemp(prefix="pkg-"))
    try:
        files = _download_package(package_id, workdir)
        if files == 0:
            return _response(404, {
                "error": "package not found",
                "package_id": package_id,
                "expected_prefix": f"s3://{BUCKET}/{PACKAGE_PREFIX}/{package_id}/",
            })
        if not (workdir / "Project_Metadata.json").exists():
            return _response(400, {
                "error": "package is missing Project_Metadata.json",
                "package_id": package_id,
                "files_downloaded": files,
            })

        package = load_package(workdir, checklist)
        adjudicator = Adjudicator(build_provider(LLM_PROVIDER))
        findings = analyse_package(package, checklist, adjudicator)
        _persist(findings)

        flags = [f for f in findings if f.predicted_label == "FLAG"]
        by_severity: dict[str, int] = {}
        for f in flags:
            by_severity[f.severity] = by_severity.get(f.severity, 0) + 1

        return _response(200, {
            "package_id": package.package_id,
            "project_title": package.project_title,
            "files_downloaded": files,
            "clauses_extracted": len(package.clauses),
            "rows": len(findings),
            "flags": len(flags),
            "flags_by_severity": by_severity,
            "adjudicator": adjudicator.stats(),
            "note": "Decision-support findings for human review; not a legal conclusion.",
        })
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def handle_findings(document_id: str, flags_only: bool) -> dict:
    document_id = _safe_package_id(document_id)
    condition = Key("document_id").eq(document_id)
    resp = _table.query(KeyConditionExpression=condition)
    items = resp.get("Items", [])
    while "LastEvaluatedKey" in resp:
        resp = _table.query(
            KeyConditionExpression=condition,
            ExclusiveStartKey=resp["LastEvaluatedKey"],
        )
        items.extend(resp.get("Items", []))

    if not items:
        return _response(404, {
            "error": "no findings stored for this package; POST /analyze first",
            "document_id": document_id,
        })

    items.sort(key=lambda i: i.get("requirement_id", ""))
    if flags_only:
        items = [i for i in items if i.get("predicted_label") == "FLAG"]

    return _response(200, {
        "document_id": document_id,
        "count": len(items),
        "findings": items,
        "note": "Decision-support findings for human review; not a legal conclusion.",
    })


def handler(event, context):  # noqa: ARG001
    LOG.info("event: %s", json.dumps({
        k: event.get(k) for k in ("resource", "path", "httpMethod")
    }))
    try:
        method = event.get("httpMethod", "")
        resource = event.get("resource", "")

        if resource == "/health":
            return _response(200, {"status": "ok", "table": TABLE, "bucket": BUCKET})

        if resource == "/analyze" and method == "POST":
            raw = event.get("body") or "{}"
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                return _response(400, {"error": "request body must be valid JSON"})
            if not isinstance(body, dict):
                return _response(400, {"error": "request body must be a JSON object"})
            return handle_analyze(body)

        if resource == "/findings/{document_id}" and method == "GET":
            params = event.get("pathParameters") or {}
            query = event.get("queryStringParameters") or {}
            flags_only = str(query.get("flags_only", "")).lower() in {"1", "true", "yes"}
            return handle_findings(params.get("document_id", ""), flags_only)

        return _response(404, {"error": "route not found", "resource": resource})

    except ValueError as exc:
        return _response(400, {"error": str(exc)})
    except Exception:
        LOG.exception("unhandled error")
        # Do not leak internals to the caller; details are in CloudWatch.
        return _response(500, {"error": "internal error; see CloudWatch logs"})
