# Phase 1 — start here

Budget: about four hours. Setup and structural audit helpers are provided. You
implement the scorer and reason through validation; there is no trained solution.

## 1. Open and check (15 minutes)

Run `bash start_notebook.sh` from the project root. Open the localhost link printed
by Jupyter. Choose the Amazon ML (local) kernel. Run the first notebook cells.
Confirm the displayed interpreter belongs to `.venv` and all seven inputs exist.
Read `reports/environment.json`. Available RAM is more useful than installed RAM.

## 2. Inspect schemas (20 minutes)

Run the bounded preview cells. Source files must have four columns: entity_id,
business_name, business_address, country. Ground truth has source1_entity_id and
matched_entity_ids. All supplied helpers read UTF-8 TSV and preserve blank strings.
Read the structural audit and explain what it checks and what it does not check.

## 3. Understand truth (30 minutes)

Implement parsing in the notebook: an empty cell represents zero matches; a
nonempty comma-separated cell represents several IDs. Detect repeated IDs BEFORE
converting to a set. Explain why two S2 matches are allowed and why a missing target
address is not evidence that it belongs to another business.

Full-file reference counts from the provided data:
- 2,206,821 reference/truth rows.
- 123,247 empty truth lists.
- 7,638,365 positive links.
- Match counts range from 0 to 11 (observed, not a prediction limit).

## 4. Implement and verify your scorer (45 minutes)

Implement `score_entity` in the notebook. Compute TP/FP/FN using set operations.
Use F0.5 = 1.25 TP / (1.25 TP + FP + 0.25 FN), with both-empty scoring 1.
Enable the supplied six-example check cell only after implementation. Then build
your macro scorer over every required reference. Missing prediction rows should
be detected as a pipeline error; intentional no-match predictions are empty sets.

Check the three-entity example described in the notebook: its macro score is 1/3.
Predicting empty for every training entity must score 123247 / 2206821 (~0.05585).
Do not substitute pooled pair F0.5 or class-level macro averaging.

## 5. Inspect real labeled groups (40 minutes)

Choose 20–30 references spanning both training countries, singletons, varied match
counts, and missing addresses. Collect their labeled target IDs, then stream S2/S3
to fetch just those records. Do not load ten million records into a Python dict.
Record what changed and what evidence of identity survived. Preserve raw text.
Do not use test records as labeled examples or look up businesses on the internet.

## 6. Design the split (45 minutes)

Write your plan before coding the split. Start with 70/15/15 train/development/
locked-holdout BUSINESS groups and preserve country and match-count buckets.
These are proposed proportions, not a generated split. Save a seed and manifest.

Audit whether a target is linked to multiple references. Connected labeled groups
must remain together. Keep every reference's true variants with it. Assign
unmatched target records separately as distractors. Ensure held-out records do
not leak into supervised training negatives. Later stress-test retrieval against
large distractor pools. France has no training labels: country-transfer checks
are stress tests, not measured France performance.

## 7. Bring back your results (5 minutes)

Share your parsing/scoring implementation, six test results, empty-prediction
baseline, five interesting labeled examples, and split plan. Then we review before
Phase 2. The audit script is infrastructure, not a substitute for these exercises.

## Common issues

| Symptom | Likely cause | Resolution |
|---|---|---|
| Only one column | Wrong separator | Use tabs explicitly |
| Kernel disappears | Memory exhausted | Restart, shrink preview/batch, avoid full-file dataframes |
| `nan` becomes a token | Missing values converted to strings | Preserve empty fields deliberately |
| Hindi text lost | ASCII-only normalization | Keep Unicode and the raw version |
| Module not found | Wrong kernel | Select this project's environment; do not install globally |
| File not found after transfer | Partial folder copy | Preserve directory structure or edit resource_dir |
| Metric divides by zero | Both sets empty | Apply the official score of 1 |
| Excellent pair accuracy | Mostly easy nonmatches | Use entity-level macro F0.5 and realistic candidates |
| Validation unexpectedly easy | Shared identities / tiny distractor pool | Audit group isolation and retrieval conditions |

The notebook intentionally skips unfinished exercise checks during Run All. Turn
them on after implementing the functions. Skipped exercises are not passing tests.
