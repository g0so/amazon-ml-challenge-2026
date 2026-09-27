# Spartans — Amazon ML Challenge 2026

This repository is the final public snapshot for the Spartans team in the Amazon ML Challenge 2026. It documents the final selected solution and includes the final archived artifacts for the best evaluated public submission.

## Final result

**Best evaluated public score: 0.789**

This score was reported by the team after the competition portal evaluation. The repository preserves the final evidence, model archive, and submission record for that result.

## Dataset and scope

The project was built on the Amazon ML Challenge 2026 dataset and challenge task. The raw organizer dataset, large processed feature stores, and expensive local caches are not included in this GitHub snapshot because they were not publicly released and are too large to distribute in a clean public repo.

This repository is therefore a clean, reviewable source archive for the solution, not a fully self-contained raw-data reproduction bundle.

## Quick start

- [Final archive overview](archive/README.md)
- [Final selection record](archive/final/selection.json)
- [Final methodology write-up](archive/final/Documentation_template.md)
- [Retrospective and lessons learned](docs/FINAL_RETROSPECTIVE.md)
- [GitHub handoff and storage notes](docs/GITHUB_ARCHIVE.md)

## Final submission summary

| Submission | Evaluated public score |
|---|---:|
| Original model A | 0.764 |
| Repaired model B on legacy candidates | 0.787 |
| India A / US and France B hybrid | 0.785 |
| Final equal-probability blend for India/US, B for France | **0.789** |

Scores above are user-reported portal results. Development scores correspond to different query/target distributions and are not leaderboard predictions.

## Final policy

- India and US: predict a link when `(p_A + p_B) / 2 >= 0.6`.
- France: retain model B predictions at `p_B >= 0.665`.
- Retrieval used a legacy candidate filter with a cap of 5,000 records, up to five tokens, separate top-50 name/address routes, and exact-both matching.
- Coverage: 1,732,544 references and 138,530,079 candidate pairs.

## Reproduction note

Use Python 3.13 on Linux and install `requirements-final.txt` in a virtual environment. The original organizer data must be obtained separately and placed under the path referenced by `project_config.json`.

```bash
python scripts/restore_final_archive.py
python scripts/reproduce_final.py --test-dir /absolute/path/dataset/test --policy blend --output /absolute/path/new_run --workers 2 --batch-size 1000
```

The restore step verifies archived checksums and restores the frozen model files and final matching TSV without overwriting different existing files. The inference entry point can rebuild from raw test data or reuse verified cache outputs with `--cache-root`.

Training packets, canonical split assignments, and the trainer source are archived for lineage. See [archive instructions](archive/README.md).

## Repository structure

| Location | Contents |
|---|---|
| `archive/` | Final models, compressed outputs, training packets, checksums, and local inventory |
| `src/` | Feature extraction, retrieval, and processing implementations |
| `scripts/` | Final replay, evaluation, validation, diagnostic, and historical experiment code |
| `models/` | Small model artifacts and metadata retained in the repo |
| `reports/` | Selected evaluation and validation records |
| `docs/` | Final retrospective, archive notes, and project handoff documents |
| `notebooks/` | Early exploratory work |
| `data/` | Local data README and project notes |

## Team and contributors

Spartans — Amazon ML Challenge 2026

Teammates and collaborators:
- Nandini Thirumalaraju
- Vedaanga Varma

## Notes for future work

This repo is meant to be a clean, usable foundation for future improvement and extension. It keeps the final solution package, documentation, and archived evidence in a form that is easy to review on GitHub while leaving the nonpublic raw data and expensive local caches out of the public repository.

The goal is to make the work understandable, reusable, and extendable without exposing competition data that was never meant to be public.
