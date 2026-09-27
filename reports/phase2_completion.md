# Phase 2 completed: verified validation and honest baselines

## Review findings and corrections

1. **Removed answer leakage.** The original baseline intersected proposed matches with the true target set, silently removing false positives. Predictions now depend only on country, business name, and address. Ground truth is attached after prediction generation.
2. **Restored the complete development target pool.** The original implementation omitted unmatched distractors. The replacement searches all dev-assigned S2/S3 records, including other references’ variants and unmatched targets.
3. **Preserved and independently verified your split.** Your saved JSON assignments were imported, not reshuffled. Checks compare actual audited IDs, coverage, true ownership, inherited partitions, country, and match counts. A self-reported manifest length alone is insufficient.
4. **Fixed the undefined reference-set variable and costly lookups.** A disk-backed indexed implementation replaces the broken baseline path. Source TSVs are streamed, rather than loaded as full dataframes.
5. **Made normalization conservative.** Unicode NFC, case folding, and whitespace collapse are applied to matching copies; raw strings are preserved exactly. Ambiguous address expansions were removed. Country participates in both rules. Empty name/address/country keys cannot match.
6. **Made evaluation explicit and reusable.** The metric handles both-empty predictions correctly, includes every development reference, and reports false positives rather than filtering them. Complete dev prediction files include empty rows.
7. **Finished the notebook and reproducibility records.** Saved reports, code/input hashes, split identity, runtime, memory, error examples, and experiment rows now accompany the results. Original modules are backed up locally in `data/processed/phase2_originals/`.

## Saved split

Seed: 2026. Provenance: `preserved_existing_json`. Split ID: `8f33fb5819bd9acf5bb963c278707919ecc4f6b205242bf5ef84f2826caf741e`.

| Partition | References | All targets | Unmatched distractors |
|---|---:|---:|---:|
| train | 1,544,769 | 7,223,790 | 1,876,813 |
| dev | 331,018 | 1,548,174 | 402,537 |
| holdout | 331,034 | 1,548,255 | 402,504 |

References are stratified by country and true-match-count buckets. Every labeled target follows its owner. Holdout assignments were checked structurally; **holdout performance was not scored**.

## Measured development results

Competition metric: mean per-reference F0.5, **not classification accuracy**. Both-empty sets receive 1.

| Rule | Macro F0.5 | Micro precision | Micro recall | TP | FP | FN | Candidate oracle F0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_exact | 5.5850% | 100.0000% | 0.0002% | 2 | 0 | 1,145,635 | 5.5850% |
| normalized_exact | 7.3524% | 100.0000% | 0.8428% | 9,655 | 0 | 1,135,982 | 7.3524% |

Predicting no matches for everyone scores **5.5846%**. Normalization changes macro F0.5 by **+1.7675 percentage points**.

| Rule | Country | Macro F0.5 | Micro recall |
|---|---|---:|---:|
| raw_exact | India | 5.5883% | 0.0004% |
| raw_exact | US | 5.5827% | 0.0000% |
| normalized_exact | India | 6.5754% | 0.4711% |
| normalized_exact | US | 7.8709% | 1.0912% |

Normalized matching falsely matches 0 of 18,486 singleton references (0.0000%).

The oracle keeps only true IDs from each exact-key candidate set. It is a diagnostic upper bound for this candidate route, not an implementable prediction rule. A classifier restricted to these candidates cannot exceed that ceiling. The difference between the achieved score and the oracle measures avoidable candidate false positives; a low oracle means retrieval must find more true matches.


### What the numbers mean here

The normalized rule recovers **9,655 of 1,145,637 true links (0.8428%)** and misses **1,135,982**. No false positives were observed in this dev partition. That is high observed precision with extremely low recall, not a strong overall matcher. Only 9,148 of 331,018 references receive any predicted target.

The normalized oracle equals the achieved **7.3524% macro F0.5**: there are no false positives for a better classifier to remove from these exact candidates. Improving the classifier alone cannot rescue this candidate set. The 18,486 true singletons explain most of the 5.5846% predict-nothing score.

Concrete missed examples in the supplied labels:

- `Memorial Ministries` versus `Mem0rial Ministries Co`: a character substitution and extra suffix; address also changes `IN` to `Indiana`.
- `Concordia's Complete Apparel` versus `Concordia's Apparel Complete`: word order changes, with address punctuation differences.
- `Interstate School of Medicine` versus `interstateschoolmedicine.com`: a compressed domain-style name.
- `Great Agro Private Limited` versus `ग्रेट एग्रो प्राइवेट लिमिटेड`, or simply `GA`: cross-script text and abbreviation. One supplied true target also has a changed house number, so a rigid number-equality filter could lose true matches.

These examples motivate candidate routes; their frequency has not been estimated. Normalized US macro F0.5 is 7.8709%, versus 6.5754% for India. Do not infer France performance from either.

## Validation and resource use

- All six automated tests passed: audit integrity, shared targets/singletons/distractors, conservative normalization, all 64 pairs of subsets over three IDs for the metric, adversarial label-blind matching, and saved-split corruption detection.
- The adversarial baseline test creates a singleton with a text-identical distractor. That false positive survives prediction generation and lowers the score. It also checks country isolation, multiple matching targets, blank keys, and agreement between SQL and the independent Python metric.
- Complete split verification/import plus baseline run: **392.89 seconds**; peak process resident memory **973.11 MiB** on this Ubuntu machine. These are observed values, not an enforced memory cap or a promise for future retrieval.
- Both baselines were evaluated on 331,018 references and 1,548,174 targets, including 402,537 unmatched distractors.
- Both exported TSVs were independently rescored using the Python metric, with exact row coverage and TP/FP/FN agreement against the SQL evaluation. Results are in `phase2_output_verification.json`.
- All nine notebook code cells executed successfully with rebuilds disabled.
- The legacy JSON import still temporarily loads assignment dictionaries; indexes and record text live on disk. Keep disk space for both old and replacement databases during reruns.

## Remaining limitations

- The dev target pool is the complete assigned dev partition, approximately 15% of training targets. Production retrieval has many more competitors. Benchmark larger/full target pools before treating local precision or runtime as representative of test inference.
- Grouping prevents shared labeled IDs across partitions, not all semantic leakage. Related businesses or content aliases may remain across distinct reference IDs.
- France has no labeled training examples. US/India scores do not estimate French performance.
- Exact equality misses spelling changes, word order, abbreviations, transliteration, and missing addresses. Preserving punctuation is intentionally conservative, not a claim that punctuation never should be normalized.
- False-positive examples may reflect genuinely distinct entities or label limitations. Do not overwrite ground truth based solely on visual similarity.
- The exported examples are the first ten per error type by ID, not a representative random sample or quantified error taxonomy.
- No holdout tuning, leaderboard submission, learned model, threshold optimization, or final test prediction has been performed.

## Phase 3 handoff

1. Read the normalized false-negative examples in the notebook and describe the transformations preventing equality.
2. Build complementary candidate routes using names and addresses separately, with country-aware blocking and explicit handling of missing addresses. Retain the exact rule as one route.
3. Benchmark a bounded dev batch first, using the full dev target pool. Measure the fraction of true links retrieved (candidate recall), oracle macro F0.5, candidates per reference, runtime, and RAM.
4. Scale the useful routes and inspect which positive links are still absent. Also stress-test the increased distractor density of larger target pools.
5. Only then learn pairwise similarity decisions with logistic regression, trained on train references and realistic retrieved negatives. Tune on dev; keep holdout closed until the confirmation checkpoint.

**Phase 2 is complete.** Its objective was a trustworthy reproducible baseline, not a competitive final matcher. The next bottleneck to address is candidate coverage, measured rather than guessed. A 99% result remains an aspiration, not a guarantee.
