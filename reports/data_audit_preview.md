# Data audit (preview)

2026-09-25T17:00:54.922931+00:00

Streaming counts; no full source table was loaded into memory.

| File | Rows scanned | Blank addresses | Country counts |
|---|---:|---:|---|
| train_source1.tsv | 1,000 | 0 | {'US': 595, 'India': 405} |
| train_source2.tsv | 1,000 | 35 | {'India': 396, 'US': 604} |
| train_source3.tsv | 1,000 | 37 | {'US': 593, 'India': 407} |
| test_source1.tsv | 1,000 | 0 | {'US': 390, 'France': 128, 'India': 482} |
| test_source2.tsv | 1,000 | 24 | {'India': 469, 'France': 140, 'US': 391} |
| test_source3.tsv | 1,000 | 26 | {'India': 473, 'France': 159, 'US': 368} |

## Ground truth

```json
{
  "file": "6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv",
  "rows_scanned": 1000,
  "blank_fields": {
    "matched_entity_ids": 47
  },
  "match_count_distribution": {
    "0": 47,
    "1": 55,
    "2": 178,
    "3": 248,
    "4": 222,
    "5": 143,
    "6": 68,
    "7": 35,
    "8": 3,
    "9": 1
  },
  "links_by_source": {
    "S2": 1657,
    "S3": 1787
  },
  "duplicate_ids_within_truth_rows": 0,
  "invalid_target_prefixes": 0,
  "empty_prediction_baseline": 0.047
}
```

## Limits

- No global ID uniqueness or cross-file referential-integrity checks.
- Preview rows are file heads, not a random or representative sample.
- Counts do not establish label correctness or a modeling baseline.
