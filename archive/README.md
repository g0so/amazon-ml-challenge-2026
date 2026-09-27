# Curated final archive

Final public leaderboard score: **0.789**, user-reported. The original complete ZIP is retained without modification after evaluation.

| Directory/file | Purpose |
|---|---|
| `final/selection.json` | Selected output/package paths, hashes and public score |
| `final/matching_results.tsv.gz` | Losslessly compressed final leaderboard upload |
| `final/policy.json` | Frozen model mixture, thresholds and evaluation evidence |
| `final/validation.json` | Full TEST coverage, alignment and subset validation |
| `final/Documentation_template.md`, `Methodology.pdf` | Approach written at package creation, before portal evaluation |
| `models/A`, `models/B` | Frozen model weights and their original metadata |
| `training/original_packet.zip` | Original model's portable labeled feature packet |
| `training/repaired_packet.zip` | Repaired retrieval packet: train, tune and confirmation |
| `training/split_manifest.json.gz` | Compressed canonical split assignments |
| `evaluation/` | Final tuning/confirmation IDs, selections and metric reports |
| `SHA256_MANIFEST.json` | Integrity hashes for archived assets |
| `local_inventory.json` | Paths/sizes of preserved local data; not a backup of those files |

## Restore frozen inference

From the repository root, run `python scripts/restore_final_archive.py`. This uses only the Python standard library, verifies the archive, restores A/B under `artifacts/`, and restores predictions under `reports/final_blend_submission/`. Different existing files cause an error instead of being overwritten.

For full inference, follow the root README. To retrain, extract the desired training ZIP to a separate directory and invoke `scripts/train_original_packet.py` or `scripts/train_repaired_packet.py --help`. The model B trainer accepts its packet ZIP directly. Original thresholds and hashes are in the archived metadata. Retraining requires CatBoost and a working GPU backend; supplied frozen weights are necessary for exact submission reproduction.

The packet data derives only from organizer inputs. Raw organizer datasets, full feature caches and the 862 MiB final ZIP remain local and are excluded from the clean GitHub export. See `docs/GITHUB_ARCHIVE.md` for optional release storage. Canonical split assignments can be restored with Python's `gzip` module when auditing the raw-data split.
