# Contract Clause Risk Flagging - end-to-end solution

## The RAG-or-skill question

Neither, as usually framed. **This challenge does not need vector retrieval, and the
part that looks like it needs an LLM is mostly a rules problem.** Three properties of
the supplied data decide the architecture:

**1. The corpus does not need retrieving.** Every document in every package is one or
two pages. A full package is roughly 1,500 words; all eight packages together are
about 4,000 words. That fits in a single prompt with room to spare. A vector index
adds a chunking step, an embedding step, a similarity threshold and a top-k cutoff,
and every one of those is a new place to lose a clause. There is no recall problem to
solve here, so retrieval infrastructure is pure downside.

**2. Retrieval is already exact.** Package documents use the checklist
`Requirement_Name` verbatim as their section headings, and addenda use
`Revision to <Requirement_Name>` followed by `REPLACEMENT TEXT:`. A normalised
string match resolves every clause to its requirement. Measured on this dataset:
**137 of 137 clauses resolved, 0 unresolved.** Embeddings would be strictly worse
here, because the requirements are lexically adjacent by design. "Contract changes
must follow written process" (CC-11) and "Notification of contract changes" (CC-12)
are near-neighbours in any embedding space and have different invariants and
different correct answers in the same package.

**3. The scored behaviours are mostly deterministic.** Mapping the evaluation weights
onto what actually decides each one:

| Evaluation metric | Weight | What decides it |
|---|---|---|
| Applicability accuracy | 20% | `Project_Metadata.json` booleans. Pure rules. |
| Cross-document precedence | 20% | Addendum supersession + DelDOT 105.6 ladder. Pure rules. |
| Severity agreement | 5% | Checklist `Severity_Guidance`, verbatim. Pure lookup. |
| Evidence and citation correctness | 15% | Falls out of heading-anchored extraction. |
| Finding detection | 25% | Numeric/modal invariants from `Challenge_Reference_Rule`. |
| Semantic deviation discrimination | 15% | The only genuinely semantic judgement. |

So 60% of the weight is decided by rules that are *written down in the challenge
materials*. Inferring a stated rule with a language model converts a correct answer
into a probabilistic one. The remaining 40% is invariant-checking, which is also
largely mechanical: the checklist states each invariant as a number or a modal
obligation ("10% of total bid price", "within 7 calendar days", "three years",
"no less than 50%", "written consent required").

**What this solution does instead:** a deterministic pipeline shaped like a skill,
with a *narrow* LLM adjudicator reserved for clauses whose invariant cannot be
decided mechanically. The model is never asked to find findings, choose a governing
document, or decide applicability. It answers one bounded question: is this clause
materially deviant or an equivalent restatement?

On the supplied data that adjudicator is invoked **zero times** - every clause is
settled by an invariant test that positively matched evidence. It exists for
documents that do not look like these, and it fails closed.

## Results

Development split, scored with the weights in `Evaluation/Evaluation_Criteria.csv`:

```
metric                                    score   weight   contrib
------------------------------------------------------------------
Applicability accuracy                   100.0%      20%    20.00
Finding detection (F1)                   100.0%      25%    25.00
Cross-document precedence                100.0%      20%    20.00
Semantic deviation discrimination        100.0%      15%    15.00
Evidence and citation correctness        100.0%      15%    15.00
Severity agreement                       100.0%       5%     5.00
------------------------------------------------------------------
WEIGHTED TOTAL                                             100.00

rows scored: 108    mismatched rows: 0
  precision 1.000 (28 TP / 0 FP), recall 1.000 (0 FN)
  10/10 paraphrase rows classified correctly (false-positive burden)
  7/7 Addendum-superseded rows resolved with the Addendum cited as governing
  35/35 evidence spans verbatim from the cited governing document
```

Read this honestly: **108/108 on 108 labelled rows is a small sample, and the rules
were written after reading the development documents.** The score demonstrates the
architecture is sufficient, not that it generalises. See *Limitations*.

Validation split (unlabelled) produces 36 rows, 10 flags: 2 Critical, 5 High,
3 Medium. I derived the expected answers by hand from the PDFs before running the
pipeline, and all 36 rows agree, including the three addendum-superseded cases
(Mill Creek CC-17 → Addendum B, Oak Hollow CC-04 → Addendum C, CC-12 → Addendum B).

## Architecture

```
Contract package (PDFs + Document_Index.csv + Project_Metadata.json)
   │
   ├─ 0. INGEST            extract.py
   │     pdftotext -layout, split on known headings, not on token windows.
   │     -> Clause{file, doc_type, heading, text, page, is_replacement}
   │
   ├─ 1. APPLICABILITY     applicability.py          [20% of score]
   │     Metadata predicate per requirement. Runs BEFORE any clause is read,
   │     so an out-of-scope requirement cannot produce a flag.
   │     -> APPLIES / DOES_NOT_APPLY + reason
   │
   ├─ 2. PRECEDENCE        precedence.py             [20% of score]
   │     (a) Addendum carrying "Revision to X" + REPLACEMENT TEXT supersedes X;
   │         latest ordinal wins (Addendum_A -> 1, _B -> 2, _C -> 3).
   │     (b) Otherwise DelDOT 105.6 ladder over base documents.
   │     -> governing clause + superseded list + resolution note
   │
   ├─ 3. INVARIANT TEST    detectors.py              [25% + 15%]
   │     One detector per requirement, testing the number or modal obligation
   │     named in Challenge_Reference_Rule. Returns the verbatim sentence it
   │     fired on. Unmatched text returns uncertain=True rather than guessing.
   │
   ├─ 4. ADJUDICATION      llm.py                    [residual only]
   │     Invoked only when step 3 reports uncertainty. Bedrock Converse /
   │     Anthropic API / null. Rejects evidence not present in the clause.
   │     Any failure degrades to the step-3 verdict.
   │
   └─ 5. EMIT              pipeline.py + evaluate.py
         15-field submission CSV, an audit CSV adding decided_by/rule_id,
         and the weighted local scorer.
```

### Why the layer order is the design

Both hard traps in this dataset are ordering problems, not modelling problems.

**Precedence before detection.** Harbor Crossing's General Conditions set bond
coverage at 75%; Addendum B replaces it with 100%. Stone Creek's General Conditions
invert the precedence hierarchy; Addendum A replaces it. Riverbend's General
Conditions make oral direction binding; Addendum A replaces it. Any system that
tests clause text before resolving supersession flags all three and is wrong all
three times. Resolving first makes them correct without a special case.

**Applicability before retrieval.** Pine Grove is not federal-aid, so CC-01 and CC-09
cannot be findings regardless of what the documents say. Gating first removes the
largest available source of over-flagging at zero cost.

### Why detectors flag on invariants, not on difference

The dataset deliberately punishes similarity-based flagging. These pairs are all
**equivalent** and must not be flagged:

- `ten percent (10%)` / `one-tenth (10%)` (CC-02)
- `three (3) years` / `36 months` (CC-13)
- `Conflicts shall be resolved using the order of precedence supplied...` /
  `Read all contract documents together; when a conflict exists, apply the same
  priority sequence...` (CC-10)
- `NON-COLLUSION CERTIFICATION - The required certification is to be completed...` /
  `The required certification shall be completed and submitted...` (CC-03)

Each detector therefore extracts the invariant and compares it, rather than comparing
strings. CC-02 flags when the percentage is not 10, not when the wording differs.
CC-13 normalises months to years before comparing. That is what produces 10/10 on
semantic discrimination and 0 false positives, with no model call.

A related rule: clauses that **defer** to the reference ("within the reference
period", "as stated in the applicable contract documents", "the governing
contract/reference schedule") preserve the requirement and are not findings. A
keyword approach reads the absence of "20 calendar days" as an omission and flags it.

CC-11 is worth calling out as the trap that caught my first implementation. The
compliant and violating forms share nearly all their vocabulary:

- preserved: `oral direction alone does not modify the contract`
- violated:  `oral direction ... immediately modifies scope, price, or time`

A keyword match on `oral direction` + `modif` fires on both. The detector tests the
negated form first, and a misplaced negative lookahead in the first version inverted
four rows. Ordering the polarity tests explicitly is the fix.

## Running it

```bash
brew install poppler                 # provides pdftotext; or: pip install pypdf

python3 run.py dev                   # analyse Development/, score against labels
python3 run.py val                   # analyse Validation/, write submission CSV
python3 run.py all                   # both
python3 run.py inspect Oak_Hollow    # per-clause trace: governing vs superseded
python3 run.py robustness            # perturbation suite (46 cases)
python3 run.py robustness invariance  # filter by kind or case name
python3 run.py schema                # validate submission field semantics
python3 run.py dev --llm bedrock     # enable adjudication of residual uncertainty
```

## Submission field semantics

Per the challenge clarification, the schema is interpreted as:

| field | meaning |
|---|---|
| `document_id` | the **package** ID, never an individual PDF |
| `requirement_id` | the CC requirement ID |
| `reference_id` | the **same** CC requirement ID |
| `reference_location` | the supporting external standard / statute / section |

The specific PDF, where relevant, is identified in `governing_document` and
located by `draft_location`. One row per contract package x CC requirement.

```
document_id         DEV-HARBOR-CROSSING
requirement_id      CC-01
reference_id        CC-01
reference_location  FHWA-1273 + federal-aid proposal - FHWA-1273 I.1
governing_document  Proposal and General Notices (Proposal_and_General_Notices.pdf)
draft_location      Proposal_and_General_Notices.pdf > Federal requirements (page 1)
```

`crf/conformance.py` enforces this as 11 machine checks (`python3 run.py schema`)
so the contract cannot drift while detectors are being changed. Each check was
verified to actually fail when its rule is broken - 13 deliberate violations were
injected and all 13 were caught, including `document_id` set to a PDF filename,
`reference_location` pointing at a package document, and a `DOES_NOT_APPLY` row
carrying a FLAG.

Outputs land in `out/`:

- `submission_{development,validation}.csv` - the 15 schema fields, in schema order
- `audit_{development,validation}.csv` - adds `decided_by` and `rule_id` per row, so
  every decision is traceable to the specific rule that produced it

`inspect` is the debugging surface. It prints, per requirement, every clause found,
which one governs, which are superseded and why, plus the applicability decision and
any heading that failed to resolve.

## Robustness: measuring the overfitting

100% on development is a statement about sufficiency, not generalisation, because
the detectors were written with those documents open. There are no held-out labels
to fix that - but labels are not actually required. `crf/perturb.py` and
`crf/robustness.py` manufacture inputs whose correct answer is known *by
construction*, then assert it.

46 cases in four kinds:

| kind | cases | assertion |
|---|---|---|
| invariance | 22 | meaning preserved, so **no** decision may move |
| directional | 16 | meaning changed, so **one named** requirement must move to a stated label and nothing else may |
| applicability | 5 | flipping a metadata gate moves that requirement in/out of scope |
| precedence | 3 | supersession chains resolve to the correct governing document |

The "and nothing else may move" half of the directional assertion is what catches
over-broad patterns: a detector reading a neighbouring requirement's text shows up
as collateral damage even when its own row is right.

Four failure modes are reported. Three are obvious (`TARGET`, `DRIFT`, `GOVERN`).
The fourth, `DEGRADE`, exists because **label equality is not evidence of health**.
When an invariant test stops matching, the detector falls through to an `unclear`
path and returns `NO_FLAG` by default. On a compliant clause that is the right
answer for the wrong reason - and it becomes the *wrong* answer the moment the same
clause carries a violation. Invariance cases therefore also require the deciding
rule to stay off the fall-through paths. Adding that check took the reported failure
count from 4 to 16, which is the more honest number.

### What it found, and what changed

Three root causes, all real, all now fixed:

**Word-number blindness (6 failures).** `strip_parenthetical_digits` renders
`ten percent (10%)` as `ten percent`, which is how documents that do not carry the
parenthetical digit form would read. Every numeric extractor read digits only, so
they returned nothing. Riverbend CC-12 turned into a **false negative** - a real
30-day violation stopped being detected. The other five were `DEGRADE`: the label
stayed correct while the reasoning evaporated. Fixed with a spelled-number parser
(`parse_spelled`) wired in as a fallback to `percents`, `days` and
`retention_years`. It handles `seventy-five` → 75, `one hundred` → 100,
`one-tenth` → 10%, and `thirty-six months` → 3.0 years.

*This is the exact limitation the first version of this document predicted. The
harness turned a guess into a measurement and then into a fix.*

**Whitespace sensitivity (7 failures).** Detector patterns span several words, so
one irregular space defeats them. Three were outright `DRIFT`: Northfield's
Critical CC-09 flag, plus CC-16 and CC-17, silently disappeared. The extractor
already normalises whitespace, so this could not fire through the PDF path - but it
would fire immediately for a caller feeding clauses from Textract or a JSON API,
which is precisely the planned deployment shape. Fixed by normalising at the
detector dispatch point instead of trusting an upstream step.

**Modal and synonym coupling (2 failures).** CC-03 matched the literal string
`shall be completed`, so `must be completed` fell through. Fixed by replacing
literal modals with shared `MODAL` / `SUPPLIED` / `OBLIGATORY` alternations.
`shall`, `must` and `is to` carry the same force here and real drafts mix them.

After the fixes: **46/46 pass, development still 100.00 / 108 rows exact, and no
validation decision changed** (only the CC-03 explanation string, which now
mentions modal choice). The suite is the regression guard for any further detector
work.

## Verification performed

- Development: 108/108 rows match `Development_Labels.csv` on applicability, label
  and severity simultaneously. 0 mismatches.
- Heading resolution: 137/137 clauses across all 8 packages resolved to a
  requirement, 0 unresolved.
- No fall-through decisions: 0 rows across both splits were decided by an
  `unclear` / `missing` / `no_value` path, so every verdict rests on a positive
  evidence match.
- Submission schema: 11/11 conformance checks pass across all 144 rows
  (`run.py schema`). Field names and order match `Submission_Schema.csv` exactly
  (15/15). Grain is 8 packages x 18 requirements = 144, zero duplicates, zero
  missing combinations. All FLAG rows carry non-empty `draft_location` and
  `draft_evidence`. Blank evidence occurs only on DOES_NOT_APPLY rows, which the
  schema permits.
- Conformance checks are themselves tested: 13 injected schema violations were all
  detected, so the checks are not vacuous.
- Evidence grounding: 35/35 flagged and precedence rows carry `draft_evidence` that
  is a verbatim substring of the document named in `governing_document`. Checked by
  substring match against the parsed clause, not by eye.
- Determinism: two consecutive full runs produce bit-identical CSVs (md5 equal).
- Vocabulary: `severity` ⊆ {Critical, High, Medium, Info}; `applicability_decision`
  ⊆ {APPLIES, DOES_NOT_APPLY}; `predicted_label` ⊆ {FLAG, NO_FLAG}; every
  DOES_NOT_APPLY row is NO_FLAG.
- Adjudicator robustness: exercised with a stub provider against valid JSON,
  code-fenced JSON, hallucinated evidence, an invalid label, non-JSON prose and an
  out-of-range confidence. Hallucinated evidence is replaced with the real clause
  span; malformed responses fall back to the rule verdict and increment a failure
  counter.

**Not verified:** the Bedrock adjudication path against a live endpoint, for two
independent reasons found while testing:

1. In `us-east-1`, the workshop role `WSParticipantRole/Participant`
   (account 451786966743) is blocked from `bedrock:InvokeModel` by an explicit deny
   in the `ws-dont-modify-policy-0` identity policy, for every Claude model in the
   region. `bedrock:ListFoundationModels` succeeds, so the models are visible but
   not invocable.
2. Calls in other regions (`us-west-2`, `us-east-2`) fail earlier, during credential
   refresh: the `default` profile authenticates via a `login_session` scoped to
   `us-east-1`, so cross-region calls return
   `CreateOAuth2Token ... authorization grant is invalid`.

The participant guide states the event Region is typically `us-west-2` and that AWS
account credentials come from the Workshop Studio dashboard. Configuring a profile
from that dashboard for the event Region is the thing to try before concluding
Bedrock is unavailable. Until then the provider code is exercised only via a stub.

Validation-split accuracy is also unverified against ground truth, since those labels
are not supplied - the agreement claimed above is against my own manual reading.

None of this blocks the solution: the pipeline scores 100% on development with the
adjudicator disabled, and `--llm null` is the default.

## Limitations

- **The rules were written with the development documents in hand.** 100% on
  development is a consistency check, not a generalisation estimate. The robustness
  suite above is the closest thing to a generalisation measurement here, and it only
  covers perturbations I thought to write. The honest signal remains the validation
  split, which is unlabelled.
- **The robustness suite tests everything downstream of PDF extraction, not
  extraction itself.** It mutates `Package` objects in memory. Heading resolution is
  covered separately (137/137), but a document whose text layer differs
  structurally - two-column layout, tables, scanned images needing OCR - is not
  exercised at all.
- **Heading extraction assumes headings match the checklist.** Real solicitations use
  section numbers (`102.8`) and prose headings instead. The path forward is to widen
  `HEADING_ALIASES` and add a section-number index; the aliasing indirection already
  exists for exactly this reason, but it has not been tested against real documents.
- **Addendum ordinals come from filename letters.** `Addendum_A` → 1 works here
  because the packages are named sequentially. Real addenda are dated, and dates
  should drive the ordering.
- **DelDOT 105.6 mapping is approximate.** `General Conditions` has no exact slot in
  the statutory ladder; it is mapped to the `Standard Specifications` rank. In these
  packages each requirement appears in one base document plus optional addenda, so
  the ladder almost never binds and this choice is untested by the data.
- These are decision-support findings for human review, not legal conclusions. Every
  row carries a `recommended_human_action` and none asserts a legal outcome.

## Where an LLM genuinely earns its place

Not on this dataset, but on real ones:

1. **Heading-free documents.** When section boundaries are not marked, a model is a
   good segmenter. Keep the segmentation step separate from the judgement step so
   evidence stays traceable.
2. **Novel clause language.** An invariant test only catches deviations someone
   anticipated. A clause that defeats a requirement through unanticipated
   construction is exactly the residual-uncertainty case step 4 exists for.
3. **Explanation quality.** The current explanations are templated. A model writing
   the reviewer-facing narrative *after* the label and evidence are already fixed
   improves readability with no risk to correctness.

The pattern in all three: let the model handle language, keep stated rules in code.
Applicability, precedence and severity are written down in the challenge materials.
Executing them is free and exact; inferring them is neither.
