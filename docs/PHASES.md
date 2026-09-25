# Revised plan: two 16 GB machines, 30 hours plus 10 optional

## Hardware assignments

- **Ubuntu i5-13420H / 16 GB:** primary project, notebooks, streaming audits,
  feature development, baseline training, and initial retrieval benchmarks.
- **Ryzen laptop / 16 GB / RTX 5060:** optional separate long-running experiment
  or later GPU workload. Confirm actual GPU VRAM and drivers before choosing a
  neural model. No CUDA setup is needed for Phase 1.
- **AWS / currently 4 GB:** optional small checks only. Do not wait for GPU quota
  approval or build the critical path around cloud access. Credits are a billing
  resource, not a guarantee of launch quota.

The Ubuntu inspection on 2026-09-25 showed about 15 GiB physical RAM, 6.8 GiB
available, and 215 GiB free disk. Available RAM changes as apps run. Close unused
applications yourself when needed; the setup does not stop other applications.
Start with a process-memory target near 4 GiB and two threads. This is not an
enforced memory cap or a promise that full-scale retrieval will fit.

## Phases

| Phase | Hours | Your work | Checkpoint |
|---|---:|---|---|
| 1. Foundations | 4 | Understand supplied audits, implement metric, inspect labels, design grouped splits | Six metric cases pass; empty-list baseline agrees; split plan recorded |
| 2. Baseline | 3 | Conservative normalization and exact matching | First reproducible validation score |
| 3. Retrieval | 7 | Complementary candidate routes; batch/index design | Candidate recall, oracle macro F0.5, peak RAM, and full-run time estimate |
| 4. Logistic regression | 5 | Similarity features, realistic negative pairs, training | Beats baseline on unseen business groups |
| 5. Error analysis | 4 | Fix largest error categories; tune threshold on development set | Improvement survives separate confirmation |
| 6. Boosted trees | 3 | One controlled comparison on the same candidates and splits | Keep only if gain justifies complexity |
| 7. Finish | 4 | Full inference, validation, reproducibility, documentation | Both TSVs and complete final package verified |

These are hands-on allocations, not runtime guarantees. The deadline remains
27 September 2026, 23:59 IST. Benchmark full-scale throughput during Phase 3 and
start final inference early enough to leave time for one rerun. If time runs short,
drop Phase 6 before sacrificing validation or final checks.

Use the optional 10 hours for the measured bottleneck, not automatically for
deep learning. Do not rebuild identical caches on both machines. Transfer source,
split manifests, configuration, and selected artifacts; rebuild virtual environments.

## Scaling checkpoints

1. Read five rows safely and stream all-file counts.
2. Build a development sample of complete business groups; do not mix it with the
   raw file-head previews. Audit shared target IDs before group assignment.
3. Benchmark candidate retrieval with increasing distractor pools. Small pools
   can artificially inflate precision and retrieval performance.
4. Increase batch sizes gradually while recording peak process RAM and time.
5. Cache compact derived features, keep sparse representations sparse, and
   consider disk-backed indexes/partitions when an index exceeds available RAM.
6. Choose training sample size based on validation and resource use, rather than
   assuming all possible pairs must be materialized.

No neural model, ensemble, or 99% score is required to complete the learning
milestones. The actual target is entity-level macro F0.5; France is unseen in
training, and public leaderboard results do not guarantee private performance.
