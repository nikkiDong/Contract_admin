# Contract Clause Risk Flagging

An evidence-grounded pipeline that reviews DelDOT transportation contract packages against a
reference checklist and flags missing, modified, conflicting, or non-standard provisions for
human review — deterministic by default, with a narrow LLM adjudicator reserved for clauses a
rule genuinely can't settle.

**[Read the full walkthrough →](https://claude.ai/code/artifact/a0926603-0982-409f-b163-b366264b5368)**
— architecture diagrams, the scoring breakdown, and what a 46-case robustness suite found when it
tried to break the detectors.

## Results

| | |
|---|---|
| Development split (108 labelled rows) | **100.00 / 100**, 108/108 exact |
| Validation split (36 rows, unlabelled) | 10 flags — 2 Critical, 5 High, 3 Medium |
| LLM adjudications used | **0** — every verdict is rule-decided |
| Robustness suite | 46 / 46 passing (after 3 real bugs found and fixed) |

Full scoring detail, per-metric breakdown, and the labelled validation flags are in the
[walkthrough](https://claude.ai/code/artifact/a0926603-0982-409f-b163-b366264b5368) and in
[`SOLUTION.md`](./SOLUTION.md).

## Why not RAG

The corpus doesn't need retrieving (~4,000 words across all eight packages), retrieval is already
exact (137/137 clauses resolve to a requirement by heading match), and 60% of the evaluation
weight is decided by rules stated in the challenge materials themselves. Inferring a written rule
with a language model turns a correct answer into a probabilistic one — see
[SOLUTION.md](./SOLUTION.md#the-rag-or-skill-question) for the full argument.

## Architecture

```
ingest → applicability gate → precedence resolution → invariant detector → (LLM only if uncertain) → emit
```

A requirement is judged in four layers, in this fixed order:

1. **Applicability** (`crf/applicability.py`) — rule predicates over project metadata; an
   out-of-scope requirement is filtered before any clause is read.
2. **Precedence** (`crf/precedence.py`) — resolves which clause governs when an addendum
   supersedes a base document, before any test runs against the text.
3. **Detectors** (`crf/detectors.py`) — a hand-written invariant test per requirement, run
   against the governing clause only.
4. **LLM adjudicator** (`crf/llm.py`) — invoked only when a detector reports genuine
   uncertainty, and only on the one clause precedence already selected.

`crf/pipeline.py` orchestrates the four layers and writes the submission CSVs. Three independent
tools check the output afterward without feeding back into the decision: `evaluate.py` (scoring),
`perturb.py` + `robustness.py` (the perturbation suite), and `conformance.py` (schema validation,
with zero internal imports of its own).

## Repository layout

```
run.py                          CLI entry point — dev / val / all / inspect / robustness / schema
crf/                             pipeline package (see the walkthrough for the file-by-file map)
Contract_Clause_Risk_Flagging/   challenge dataset — packages, checklist, schema, labels
out/                             generated submission and audit CSVs
deploy/                          AWS CDK stack (API Gateway + Lambda) for a hosted deployment
instruction/                     hackathon participant guide
SOLUTION.md                      full write-up: architecture, results, verification, limitations
```

## Running it

```bash
brew install poppler                 # provides pdftotext; or: pip install pypdf
pip install -r requirements.txt

python3 run.py dev                   # analyse Development/, score against labels
python3 run.py val                   # analyse Validation/, write submission CSV
python3 run.py all                   # both
python3 run.py inspect Oak_Hollow    # per-clause trace: governing vs superseded
python3 run.py robustness            # perturbation suite (46 cases)
python3 run.py schema                # validate submission field semantics
python3 run.py dev --llm bedrock     # enable adjudication of residual uncertainty
```

Outputs land in `out/`: `submission_{development,validation}.csv` is the 15-field schema
deliverable; `audit_{development,validation}.csv` adds `decided_by`/`rule_id` so every verdict
traces back to the rule that produced it.

## Further reading

- **[The Four-Layer Gate](https://claude.ai/code/artifact/a0926603-0982-409f-b163-b366264b5368)** — the full interactive walkthrough: the RAG-or-skill argument, architecture diagrams, scoring, and the robustness findings.
- **[SOLUTION.md](./SOLUTION.md)** — the same narrative in plain Markdown, including verification detail and known limitations.
