# Recovery handoff — submission 001 scored 0.764

**Historical snapshot. Active execution instructions now live in [the 27 September scale-up plan](../RECOVERY_PLAN_2026-09-27.md).** The user has confirmed full data on the RTX laptop and authorized larger training plus parallel retrieval repair. Read this file for provenance; its earlier pause/one-experiment limits are superseded.

Read this first in a new chat. User requests preservation and organization before planning further experiments. Do not launch more training or change thresholds blindly. No score near 0.98 has been demonstrated.

## Current result and limits

User reports leaderboard submission #1, 27 September 2026, 04:29 PM IST: **macro F0.5 = 0.764**. This is user-reported portal evidence, not independently fetched. Confirm the uploaded file matches the checksum below before attributing the score conclusively to this model.

CatBoost secondary development score: **0.9168941213537718**, TP6024/FP328/FN922, 2,000 seed-2027 refs, 189,461 candidate pairs. India .88993884, US .93486431. This development sample was inspected repeatedly across experiments; it is not a fresh holdout. Tuning score .9162885, threshold .362 selected on seed-2029, disjoint from secondary refs. Model trained on only **5,000 reference queries / 406,065 candidate pairs**, not all labeled data. No France training or development labels.

V3 development score .87876284. CatBoost improved this same-sample comparison but that did not prove leaderboard generalization. Holdout labels have not been used to fit or select the model in this workflow.

## Paths: two roots matter

Canonical project root for ALL current code, data, models and outputs: `/home/a/Projects/Amazon ML hackathon`
Former Codex worktree: `/home/a/.codex/worktrees/dc00/Amazon ML hackathon` (historical duplicate; no longer needed to run the project).
ORIGINAL below now means the same canonical project root.
Python: `/home/a/Projects/Amazon ML hackathon/.venv/bin/python`
Worktree-local CatBoost installation: `.gpu_runtime` (ignored by Git; scripts prepend it).

All relative paths below now resolve inside the canonical main project. Old JSON report paths are preserved as historical provenance; use the corresponding main-project relative paths.

- Validated CatBoost outputs: `data/processed/inference_runs/catboost_first_submission/output/`
- Matching TSV SHA256: `aaf4659d58b131c2f0746e65a4329d750bdb292d510323cb1ebd13eecab5b9c1`
- Candidate TSV SHA256: `9fca2c3685648dc3a46480d1ce7530b69842cac0021088a58ba26d493fd7694e`
- Model: `artifacts/gpu_challenger/model.cbm`
- Model SHA256: `1ea59adbbe61b158501508c032f6b4173f2b1da67ccfe84c01070a3e01170f3d`
- Model settings, parity fixtures, tuning predictions: `artifacts/gpu_challenger/`
- Feature packet: `data/processed/gpu_feature_packet_v1/` and `.zip`
- Secondary evaluation: `reports/gpu_challenger_secondary_eval.json`
- Original uploaded challenger: ORIGINAL `gpu_challenger.zip`
- Full expensive feature cache and V3 fallback: ORIGINAL `data/processed/inference_runs/v3_first_submission/`
- Correct test text DB: ORIGINAL `data/processed/test_inference_verified.sqlite`
- DO NOT USE old `phase6_test_targets.sqlite` for normalized exact matching: its normalization columns were incorrect.
- Training text DB: ORIGINAL `data/processed/phase4_train_text.sqlite`
- Dev text DB: ORIGINAL `data/processed/phase2_baseline_dev.sqlite`
- Split and relationship DBs: ORIGINAL `data/processed/phase2_split.sqlite`, `training_relationships.sqlite`
- Exact secondary candidates/truth/sample: ORIGINAL `data/processed/phase3_comparison/our_phase3_candidates.zip` and `phase3_fair_comparison.zip`
- Official rules: `6ab10eb3b23ba_student_resource/student_resource/README.md`
- Official validator: same resource directory `utils/validate_submission.py`

Do not relocate/delete large files or overwrite completed runs. Small evidence/code copies and hash manifest live in `reports/recovery/2026-09-27_submission_001/`. That snapshot is portable context, not a complete data/model backup. Git excludes artifacts and processed data; committing alone does not transfer them.

## Established validation

1,732,544 test queries: France259452, India809986, US663106. Test target pool9,969,589. Both CatBoost and V3 complete outputs passed unchanged official per-file checks applied batchwise plus full streaming coverage, target existence, uniqueness and subset checks. Entire official validator was not run as one in-memory job because it retains all candidate sets.

CatBoost rescore used the same cached 22-feature rows and candidate ordering as V3. CPU/GPU fixture probability difference ~1e-19, no threshold differences. Model/packet/schema verification passed. Windows path ordering explains sender packet digest; verifier uses PureWindowsPath ordering. Three submission tests passed. Rescore+validation took ~303 seconds. This validates mechanics, not statistical performance.

## Highest-priority hypotheses (unproven)

1. **Candidate-pool mismatch:** dev has1,548,174 targets, training partition7,223,790, test9,969,589. Saved train packet oracle macro .91335245 versus tuning .98405507. Different samples mean these are warning evidence, not a controlled causal comparison.
2. **Scale-sensitive retrieval:** per-country token DF cutoff min(5000, .05*N), top5 query tokens, top50 name + top50 address union plus exact route. At larger N the absolute5000 cutoff can eliminate useful tokens; top-K competition and token selection also change. Audit rates, missed-positive ranks and candidate oracle at realistic pool size before raising K or removing the cutoff.
3. **Country/domain shift:** France absent from training; US/India test mix differs from dev. Test country counts alone cannot identify France's score. Do not blame France without evidence.
4. **Calibration/classifier shift:** threshold .362 was selected with a much smaller target pool; harder negatives may change precision and singleton false matches. More training alone will not recover excluded true candidates.
5. **Implementation or upload mismatch:** verify actual uploaded file hash, metric edge cases, normalization, query-to-candidate offsets, and training/inference consistency. Existing passing checks reduce but do not eliminate these risks.

## Proposed next decision sequence (plan, not executed)

A. Establish provenance: confirm submitted file; preserve .764 record. Independently check official macro metric (including zero-truth and zero-prediction cases) on saved predictions. Summarize test candidate/match counts by country and compare equivalent dev summaries; unlabeled test summaries are diagnostics, not accuracy.
B. Build a bounded, reproducible realistic-pool diagnostic on DEV queries disjoint from fitting. Keep query truth complete; include a much larger eligible target pool as distractors, without using labels to select candidates. Never fit on diagnostic labels. Preserve closed holdout for one final check; explicitly define allowed corpus/split policy before including other partition text. Retrieve the SAME dev queries on small and large pools to isolate pool-size effects. Measure link recall, oracle macro, actual macro, FP/singletons, token exclusion and runtime by country. Start 200-500 stratified dev refs, expand only if needed; label this diagnostic, not final proof.
C. If oracle collapses, prioritize retrieval: test ONE controlled change (DF policy/top-token selection or K) with frozen classifier; quantify gained positives and candidate/runtime cost. If oracle stays strong but actual score collapses, prioritize realistic-pool hard negatives and calibration using separate fit/tune refs, then evaluate on a separate diagnostic set.
D. Only then choose a bounded CatBoost retrain or retrieval modification. RTX5060 trains features quickly; CPU retrieval/string features are the expensive part. Current cached feature rescoring is minutes. Changed candidates/features require recomputation—do not promise the same runtime.
E. Submit only a validated, evidence-backed improvement, record exact hashes and portal result. Do not sweep thresholds on the leaderboard. Reserve time for complete inference and validation before the deadline; ask current deadline/time when resuming.

## Reusable scripts

`scripts/evaluate_gpu_challenger.py`: frozen-threshold secondary evaluation.
`scripts/rescore_gpu_challenger.py`: resumable cached-feature rescore; frozen identity/checksums. Existing output name is tied to current identity—use a NEW run for changes.
`scripts/run_submission.py`: country-wise retrieval/features and atomic caches.
`src/business_entity_resolution/inference_features.py`: exact training-compatible22 features.
`scripts/export_gpu_feature_packet.py`, `scripts/verify_gpu_feature_packet.py`: CPU-to-GPU packet.
`docs/RTX5060_AI_HANDOFF.txt`, `docs/TWO_MACHINE_TRANSFER.md`: prior GPU instructions; this recovery handoff supersedes claims of production readiness.

## Collaboration constraints

User is low on Codex credits. Use concise batched inspection, no agents or large experiments unless justified. User has16GB Ubuntu i5 plus16GB RTX5060 laptop; GPU8GB. Prefer GPU for fitting and CPU for retrieval/features. Never claim .98 guaranteed. Do not use more leaderboard submissions as a substitute for sound validation.
