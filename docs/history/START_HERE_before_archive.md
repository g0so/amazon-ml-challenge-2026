# Amazon ML hackathon — current workspace

Shared model/split assets and final verification outputs are available in Git as compressed bundles. After pulling, run:

```sh
python scripts/prepare_rtx_handoff.py --extract
python scripts/restore_verification_artifacts.py --extract
```

See [verification outputs](output/verification/README.md) for contents and original paths. Raw organizer datasets and rebuildable SQLite/feature caches remain local.

**Current execution plan (27 September, 16:45 IST):** [larger GPU training plus retrieval repair](docs/RECOVERY_PLAN_2026-09-27.md). The user confirmed full data is on the RTX laptop and the deadline is 23:59 IST. This replaces the old one-experiment limit. Ready-to-copy prompts: [Luna](docs/LUNA_RTX_EXECUTION_PROMPT.txt) and [Gemini](docs/GEMINI_RETRIEVAL_PROMPT.txt). Independent audit: `reports/recovery/2026-09-27_strategy_audit.json`.

Start with [the recovery handoff](docs/recovery/START_HERE.md).

- Actual leaderboard score: **0.764** (submission1, September27 16:29 IST; user reported).
- Current model: `artifacts/gpu_challenger/model.cbm`, threshold0.362.
- Submitted output: `data/processed/inference_runs/catboost_first_submission/output/matching_results.tsv`.
- Candidate output and validation are in the same run directory.
- Expensive original features and fallback: `data/processed/inference_runs/v3_first_submission/`.
- GPU training packet: `data/processed/gpu_feature_packet_v1/` and its ZIP.
- Python: `.venv/bin/python`; local CatBoost dependency: `.gpu_runtime/`.
- Recovery evidence: `reports/recovery/2026-09-27_submission_001/`.
- Consolidation audit: `reports/recovery/consolidation_manifest.json`.

Do not delete processed data as temporary: it includes expensive reusable feature caches, split manifests, and test databases. Do not retrain blindly; investigate the small-dev-pool versus full-test-pool gap first. See the recovery handoff for the diagnostic plan.

Older reports contain their original absolute paths for provenance. Their corresponding files are now also present under this project. Historical handoff documents are superseded by `docs/recovery/START_HERE.md`.
