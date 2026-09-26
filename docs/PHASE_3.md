# Phase 3 - retrieval and validation

The repository has no Phase 2 retrieval implementation yet, so the Phase 3
harness is the first reusable candidate-generation path. It uses the existing
streaming TSV reader and the official entity-level precision-weighted F0.5
definition from the Phase 1 notebook.

## Bounded experiment

Command:

```text
python scripts/run_phase3.py --sample-size 120
```

The sample is deterministic with seed 2026. It selects 20 references from each
of six match-count strata (0, 1, 2, 3, 4, and 5-or-more) from the training truth,
then joins those IDs to Source 1 by entity ID. The run contained 70 US and 50
India references. It streams Source 2 and Source 3 once, retaining candidates
only for those 120 references. It does not create a full target index, all-pairs
matrix, submission file, or full-dataset candidate cache.

Results are saved to `reports/phase3_retrieval.json`.

| Route | Candidate recall | Oracle macro F0.5 | Mean candidates | Min | Max |
|---|---:|---:|---:|---:|---:|
| normalized name | 0.0600 | 0.333953 | 9.04 | 0 | 162 |
| normalized address | 0.0100 | 0.235641 | 0.20 | 0 | 2 |
| country + name prefix | 0.3700 | 0.148643 | 2,693.32 | 0 | 20,168 |
| union | 0.4100 | 0.153703 | 2,693.43 | 0 | 20,168 |

The union scan took 135.02 seconds and peaked at 0.053 GiB process RSS on the
Windows development machine. The memory measurement uses psutil when available,
the Windows process-memory API otherwise, and a Unix RSS fallback.

The prefix route is the complementary high-recall route, while exact normalized
name/address routes provide much smaller candidate sets. The union is therefore
not yet suitable as a final model input without a second-stage filter, but its
streaming shape can scale by processing reference batches and partitioning or
disk-backing larger indexes rather than loading all targets at once.
