# Data audit (full)

2026-09-25T17:02:53.191615+00:00

Streaming counts; no full source table was loaded into memory.

| File | Rows scanned | Blank addresses | Country counts |
|---|---:|---:|---|
| train_source1.tsv | 2,206,821 | 0 | {'US': 1323633, 'India': 883188} |
| train_source2.tsv | 5,034,616 | 168,967 | {'India': 2017799, 'US': 3016817} |
| train_source3.tsv | 5,285,603 | 175,916 | {'US': 3170056, 'India': 2115547} |
| test_source1.tsv | 1,732,544 | 0 | {'US': 663106, 'France': 259452, 'India': 809986} |
| test_source2.tsv | 4,887,273 | 129,408 | {'India': 2312565, 'France': 703378, 'US': 1871330} |
| test_source3.tsv | 5,082,316 | 136,098 | {'India': 2405000, 'France': 731615, 'US': 1945701} |

## Ground truth

```json
{
  "file": "6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv",
  "rows_scanned": 2206821,
  "blank_fields": {
    "matched_entity_ids": 123247
  },
  "match_count_distribution": {
    "0": 123247,
    "1": 119157,
    "2": 375212,
    "3": 530841,
    "4": 484115,
    "5": 321957,
    "6": 164868,
    "7": 63968,
    "8": 18680,
    "9": 4205,
    "10": 534,
    "11": 37
  },
  "links_by_source": {
    "S2": 3693619,
    "S3": 3944746
  },
  "duplicate_ids_within_truth_rows": 0,
  "invalid_target_prefixes": 0,
  "empty_prediction_baseline": 0.05584820880352326
}
```

## Limits

- No global ID uniqueness or cross-file referential-integrity checks.
- Preview rows are file heads, not a random or representative sample.
- Counts do not establish label correctness or a modeling baseline.
