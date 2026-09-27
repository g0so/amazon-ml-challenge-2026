# Revised execution plan — 27 September, 16:45 IST

Deadline confirmed by the user: **23:59 IST today**. Full training data is already on the RTX 5060 laptop. This plan supersedes the execution limits in the earlier RTX handoff and the proposed sequence in `docs/recovery/START_HERE.md`. Preserve those documents as history.

## Decision

Run larger GPU training and retrieval repair in parallel. Keep the current submission as a fallback. Do not spend the remaining time on another 5,000-query experiment or on serial approval exchanges. The objective is the strongest validated submission achievable before the deadline; 0.98 remains the target, not an established capability.

The user wants Luna to execute on the RTX laptop, Gemini to supply substantial implementation work for local execution, and Codex to inspect compact evidence and make decisions. Neither external AI has been contacted by this task. Copy the accompanying prompts to them.

## What has actually been established

| Measurement | Value | Interpretation |
|---|---:|---|
| Public leaderboard | 0.764 | User-reported submission 1; portal upload identity not independently checked |
| Secondary development macro F0.5 | 0.916894 | 2,000 previously inspected queries; small target pool |
| Tuning macro F0.5 | 0.916289 | Selected threshold 0.362 on these queries |
| Training queries / candidate pairs | 5,000 / 406,065 | Only 0.324% of the 1,544,769 TRAIN queries |
| Recorded GPU fit time | 11.03 seconds | Evidence that this particular fit was cheap, not a forecast for larger fits |
| Current model on its TRAIN queries | 0.839310 | Newly recomputed; in-sample, not a generalization estimate |
| TRAIN candidate oracle | 0.913352 | Perfect rejection/acceptance of existing candidates still cannot reach 0.98 |
| TRAIN missed links | 3,673 | 2,691 retrieval misses and 982 retrieved-but-rejected positives |
| TRAIN singleton false-positive rate | 22.70% | Versus 9.92% on tuning; substantial precision/calibration concern |
| TRAIN retrieval misses with identical normalized name | 343 | Confirmed gap in the existing independent exact-name coverage |
| TRAIN retrieval misses with identical normalized address | 65 | Another measurable repair opportunity; none of these 2,691 misses match both fields exactly |

The independent audit is `reports/recovery/2026-09-27_strategy_audit.json`. Six metric edge cases passed. The 12 sampled feature rows per partition exactly match fresh feature extraction; this is a sample check, not exhaustive parity. The local matching TSV has the expected saved SHA-256. No claim is made that the portal uploaded that exact file.

Target pools differ: TRAIN 7,223,790; DEV 1,548,174; TEST 9,969,589. These observations strongly justify a controlled pool-size experiment, but do not establish that pool size explains the entire leaderboard gap. France is unlabeled and unseen in training. No country-specific test accuracy is known.

The previous full test run scored **138,530,079 pairs**. Summed batch processing time is **8,491 seconds (2 h 22 min)**, excluding some indexing, writes, validation and interruption overhead. The status file's 3,218 seconds describes only its last process. Do not use it as the full-run forecast. Cached CatBoost rescore plus validation took about 303 seconds.

## Two coordinated workstreams

### Luna / RTX laptop: scale the matcher now

Use `docs/LUNA_RTX_EXECUTION_PROMPT.txt`. Begin with the existing 22 features and model-compatible retrieval so Ubuntu can cheaply rescore its saved features if this helps. Expand to **50,000 queries immediately**, then **250,000**, then **500,000 or all 1.54 million TRAIN queries** if measured extraction time, RAM and improvement justify it. These are checkpoints, not arbitrary ceilings. Use the full eligible target pool for retrieval, not a tiny sampled negative universe.

All candidates at 250,000 queries could be around 20 million pairs; measured counts determine the real size. Twenty million rows x 22 float32 features occupy 1.76 GB before labels, model pools, copies, indexes or training overhead. Stream feature shards and release retrieval indexes before GPU fitting. Full dataset processing does not require all records, strings and pair matrices to coexist in RAM/VRAM.

Use both all-candidate fitting at the smaller checkpoint and representative hard-negative selection at larger checkpoints. Retain retrieved positives and strong negatives; sample the long easy-negative tail with recorded sampling probabilities. Calibration must use complete candidate sets. Preserve singleton queries and unretrieved positives in evaluation denominators. Training on more data alone is not an explanation for recovered retrieval misses.

Create fresh disjoint tune and comparison query sets from DEV, excluding previously used tune/secondary IDs. Prefer 5,000–10,000 queries each after an initial 2,000-query checkpoint. Use a common TRAIN+DEV target corpus (8,771,964 targets), including distractors and every DEV truth target. Keep HOLDOUT text/labels outside model selection. Do not recreate assignments independently on the two machines.

Deliver a usable native model with schema, fixed threshold, full query-level validation predictions, timings and CPU parity. Continue automatically through the next checkpoint; do not stop merely because one fit finished.

### Gemini / Ubuntu: repair candidate retrieval

Use `docs/GEMINI_RETRIEVAL_PROMPT.txt`. Gemini supplies code and commands; the user runs them on Ubuntu. The first useful output is an executable diagnostic/repair script, not another strategy essay.

On the SAME fresh DEV queries, compare the existing DEV pool with TRAIN+DEV. First run 1,000–2,000 stratified queries, then expand to 5,000 or more. Freeze the current classifier and threshold for the diagnostic. Count true links actually absent from the corpus separately from retrieval failures; absence of a known DEV truth target is a corpus construction error.

Prioritize complementary candidate routes:

1. Independent exact-name and exact-address buckets, with deterministic cost controls for large buckets. Current exact matching requires both fields simultaneously. Identical-name candidates must not be automatically accepted as matches.
2. Combined name/address evidence, including candidates that rank moderately on both and lose in separate top-50 lists. Measure S2/S3 tie starvation and source-balanced quotas.
3. Test a wider DF cutoff and top-K as separate ablations. The existing absolute cutoff 5,000 can discard all useful tokens in common business names. Expanding only K cannot fix excluded postings.
4. If remaining misses show spelling/diacritic damage, add a bounded character-based or accent-folded route. Keep original text and original 22-feature semantics. Deeper multilingual modeling is conditional on cross-script misses being a major remaining loss and a measured end-to-end runtime that fits today.

Union repair candidates with the baseline so old coverage cannot disappear accidentally. Report oracle improvement, extra pairs, latency and memory. Send the selected portable retrieval implementation and its configuration to Luna; the same retrieval must generate training/calibration and production candidates. Avoid two incompatible independently modified implementations.

## How the results choose the next step

* Oracle below 0.98: reaching 0.98 on that evaluation set is impossible with those candidates. Improve retrieval; a model-only submission may still be a worthwhile incremental improvement.
* Oracle at least 0.99 but classifier well below it: invest in larger representative training, ambiguity-aware features and precision calibration. An oracle barely above 0.98 leaves almost no error budget.
* Many exact-name false matches or singleton errors: inspect common-name ambiguity and candidate score competition. Do not globally lower the threshold to chase recall.
* 50k → 250k learning curve flat: use error categories to choose features/retrieval changes before spending time on 1.5m indistinguishable examples.
* Cross-script misses dominate after lexical repair: an RTX-trained multilingual reranker/retriever becomes a justified next experiment, but first benchmark both indexing all targets and inference on 1.73m queries. A fast training run alone does not establish feasibility.

A fresh comparison set supports choosing between candidates but becomes development evidence once inspected repeatedly. Reserve one 10k–20k HOLDOUT query evaluation for the frozen finalist; build its eligible corpus with all supplied training targets, keep settings fixed, and report that corpus-size change. Do not tune after seeing this final check. Report paired query bootstrap uncertainty for close model comparisons; small score deltas are not automatically real improvements.

## Time budget and execution cutoffs

These are targets; use the actual clock and benchmark, not blind adherence to a stale schedule.

| IST | Expected result |
|---|---|
| Now–17:20 | Both workers active; corpus/split checks; retrieval diagnosis; first larger training shards |
| 17:20–18:15 | First larger GPU model; selected retrieval ablation; production throughput estimate |
| 18:15–19:00 | Repaired-candidate training/calibration and comparison; freeze production configuration as soon as it wins |
| Approximately 19:00 | Start changed-candidate full inference if forecast supports completion; start earlier if slower |
| 22:15 | Target completion of full inference |
| 22:15–23:00 | Complete validation, inspect result, portal upload by user, capture score/file hash |
| 23:00–23:40 | Complete reproducibility package; allow submission/packaging buffer |

Compute the latest production start as **22:15 minus 1.4 x measured projected total runtime**, including index construction, feature extraction, writing, assembly and validation. If this time arrives earlier than 19:00, start earlier or choose the measured faster plan. The old feature pipeline already consumed over 2h20 of batch work. Doubling all candidate counts may consume the remaining window.

Keep two production paths:

* Existing 22 features / unchanged candidates: validate the larger model, then use a NEW run of cached rescoring. Preserve the original validated output.
* Additive retrieval repair: reuse existing feature rows by explicit `(reference_id, target_id)` identity, compute only new pair rows, reconstruct offsets and output ordering, and verify cached-vs-fresh parity on a sample. If safe merging cannot be finished quickly, use a complete new resumable run with a measured ETA.

More features change the schema and require computing them on all scored test pairs. No silent substitution of similarity functions in the old 22 columns. Freeze expensive schema changes only when the end-to-end budget supports them.

## Required handoffs and validation

Luna returns `model.cbm`, ordered feature schema, threshold and tune protocol, candidate config/hash, data/split hashes, model parameters, query-level predictions, metric breakdown, memory/timing report and CPU parity fixture. Gemini returns portable code, exact run commands, same-query small/large-pool report, repair ablations and estimated production runtime.

Verify both TSVs cover all 1,732,544 test references exactly once; targets exist, lists have no duplicates, final matches are subsets of the exact candidates actually scored. Use official checks plus memory-safe complete streaming checks if the all-in-memory validator is too large. France must remain included. Complete the required final ZIP with code, dependencies, reproducibility instructions and methodology document.

No external business lookup/enrichment; only supplied data. Follow the official model license/size rules. Preserve prior runs and caches. Do not upload to the portal automatically. A score claim requires an actual portal result.

Official implementation references: https://catboost.ai/docs/en/features/training-on-gpu and https://catboost.ai/docs/en/concepts/python-reference_pool . GPU training can be nondeterministic; save the exact model and predictions rather than promising bitwise retraining reproducibility.
