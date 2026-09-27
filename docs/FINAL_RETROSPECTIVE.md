# Final retrospective

Team: Spartans. Event: Amazon ML Challenge 2026. Final best reported public score: **0.789**.

## What worked

Country blocking and lexical retrieval made millions of targets tractable. Two 5,000-query CatBoost training packets produced reusable small models on the RTX 5060. The final model B fit took about 15 seconds. Preserving the original feature cache made full TEST rescoring possible in minutes: the final blended run took about 199 seconds, covered 1,732,544 queries and checked all 138,530,079 candidate pairs.

## What held the score back

The old 5,000 document-frequency cap excluded useful common tokens in large target pools. Repaired retrieval improved development candidate coverage, but complete production retrieval/features could not finish within the remaining deadline. The final submission therefore retained legacy candidates. A classifier cannot recover links that never reach its candidate set.

Small-development-pool results overstated transfer to the complete target corpus. Repaired-candidate validation did not establish accuracy on legacy candidates. A country-specific hybrid improved an inspected development comparison but fell from 0.787 to 0.785 on the public leaderboard. More GPU training alone would not solve those retrieval and distribution problems.

## Final bounded experiment

Six thousand DEV queries excluded known earlier samples and were split into 3,000 tuning and 3,000 confirmation queries. Retrieval used the full TRAIN+DEV target pool. An initial selection requiring nonnegative tuning gains in both countries retained model B. A secondary aggregate-only tuning selection chose equal A/B probabilities and threshold 0.6. France remained B at 0.665.

Confirmation macro F0.5 was 0.823371 versus 0.821846 for B. The difference was +0.001524, with paired-bootstrap 95% interval [-0.001717, +0.004867]. This was uncertain development evidence, not a proven accuracy improvement. One exploratory portal submission subsequently scored **0.789**, compared with 0.787 for B. No private score is known.

## Engineering lessons

- Measure indexing, candidate retrieval, features and prediction separately. GPU availability does not accelerate Python string comparisons automatically.
- Use bounded batches and disk-backed or memory-mapped caches. A global Python dictionary of all cached pairs caused memory failures.
- Freeze a portable packet contract and verify feature order, hashes, offsets, truth and split membership before transfer.
- Never edit worker-imported code while Windows multiprocessing jobs are running.
- Benchmark representative end-to-end work early and preserve a valid fallback throughout.
- Keep model hashes distinct from transfer-ZIP hashes, and candidate recall distinct from oracle macro F0.5.

## If the project is revisited

First build a representative full-corpus evaluation with a genuinely untouched final holdout. Then repair candidate retrieval, benchmark total production cost, and only afterward scale training. That is future work, not a claim about this submission.
