# Phase 2 relationship audit

2026-09-26T13:53:35.465267+00:00

**Scope:** Full training ground truth and all three training source ID columns

| Measure | Result |
|---|---:|
| reference_rows | 2,206,821 |
| positive_link_rows | 7,638,365 |
| singleton_rows | 123,247 |
| distinct_labeled_targets | 7,638,365 |
| unmatched_target_record_rows | 2,681,854 |
| truth_rows_with_duplicate_targets | 0 |
| duplicate_reference_ids | 0 |
| duplicate_source_ids | 0 |
| targets_with_multiple_references | 0 |
| truth_references_missing_from_source1 | 0 |
| source1_records_missing_truth | 0 |
| truth_targets_missing_from_sources | 0 |

## Interpretation

Each labeled reference and its target variants forms a separate labeled group: no targets are shared across references.
A connected-component algorithm is unnecessary for the supplied labeled edges. Preserve each complete reference group when splitting.

Unmatched target rows are potential distractors, not records to discard or attach to invented references.

## Next step

Create and verify a saved group split manifest. It has not been created by this audit.

## Limits

- Does not establish semantic identity correctness or completeness of labels.
- Distinct IDs can still have identical or related business text; content leakage is not checked.
- No split manifest, normalization, matcher, or model was created.

Elapsed: 211.28 seconds.
The JSON report and SQLite audit_metadata table include input SHA-256 checksums.

## Resource use of the verified full run

- Wall time: 211.32 seconds (about 3.5 minutes).
- Peak resident process RAM: 94,352 KiB (about 92.1 MiB).
- Completed SQLite database: 1,473.0 MiB.
- All notebook result cells executed successfully with the full scan disabled.

These measurements apply to this audit, not future training or retrieval.
