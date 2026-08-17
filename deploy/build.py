#!/usr/bin/env python3
"""Stage the Lambda deployment bundle.

Assembles build/lambda_bundle/ containing:

    handler.py          the Lambda entry point
    crf/                the pipeline, unmodified
    pypdf, ...          pure-Python dependencies installed with pip --target

Uses `pip install --target` rather than a Docker build because every runtime
dependency is pure Python, so there are no platform wheels to match. That keeps
the build working on a laptop with no Docker daemon.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"
BUNDLE = ROOT / "build" / "lambda_bundle"

# Runtime dependencies. boto3 is provided by the Lambda runtime already.
RUNTIME_DEPS = ["pypdf==5.1.0"]

EXCLUDE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".DS_Store",
    # Test-only modules: no reason to ship them to production.
    "perturb.py", "robustness.py", "evaluate.py",
)


def main() -> int:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    shutil.copy2(DEPLOY / "handler.py", BUNDLE / "handler.py")
    shutil.copytree(ROOT / "crf", BUNDLE / "crf", ignore=EXCLUDE)

    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--target", str(BUNDLE), *RUNTIME_DEPS],
        check=False,
    )
    if proc.returncode != 0:
        print("pip install failed", file=sys.stderr)
        return proc.returncode

    # Drop packaging metadata; it is dead weight in a Lambda zip.
    for pattern in ("*.dist-info", "*.egg-info", "__pycache__"):
        for path in BUNDLE.glob(pattern):
            shutil.rmtree(path, ignore_errors=True)

    size = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file())
    files = sum(1 for f in BUNDLE.rglob("*") if f.is_file())
    print(f"bundle: {BUNDLE}")
    print(f"  files: {files}")
    print(f"  size : {size / 1024:.0f} KiB")

    missing = [
        name for name in ("handler.py", "crf/pipeline.py", "crf/detectors.py", "pypdf")
        if not (BUNDLE / name).exists()
    ]
    if missing:
        print(f"ERROR: bundle is missing {missing}", file=sys.stderr)
        return 1
    print("  contents verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
