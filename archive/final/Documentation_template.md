# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Spartans

**Submission Date:** 27 September 2026

**Team Members:**

- Jaya Venkata Saketh Busarla — Gandhi Institute of Technology and Management (GITAM), Visakhapatnam, Andhra Pradesh
- Nandini Tirumalaraju — Andhra University College of Engineering (AUCE), Visakhapatnam, Andhra Pradesh
- Vedaanga Varma Gullala — Andhra University (AU), Andhra Pradesh

**Final policy:** India and US: mean of A and B probabilities >= 0.6; France: B >= 0.665

## 1. Executive summary and problem analysis

We resolve each deduplicated Source 1 business to zero or more Source 2/3 records using country blocking, lexical candidate retrieval and a CatBoost pair classifier. Names and addresses contain missing fields, spelling variation, abbreviations and multilingual text. Only the supplied competition data is used; no external business lookup or enrichment is performed.

## 2. Candidate generation

Final inference uses the original country-specific candidate pipeline: retain tokens with document frequency at most min(5000, 0.05 x country target count), select up to five query tokens by inverse document frequency, and union the top 50 name and top 50 address results plus exact normalized-name AND normalized-address matches. Ties use target ID order. All 1,732,544 test references are covered, including 259,452 France references. The final candidate set contains 138,530,079 pairs.

An experimental DF20,000 / K100 joint-and-exact retriever improved development recall and supplied Model B training pairs, but it was NOT deployed for the final full test run. Candidate recall is imperfect; we do not claim that all true matches were retained.

## 3. Matching models and decision policy

The unchanged 22 features encode name/address token Jaccard, normalized exact agreement, missingness, character sequence similarity, containment, numeric agreement/conflict, address-number relationships and name-similarity interactions with missing addresses. Two depth-6 CatBoost models were trained on GPU from 5,000 labeled TRAIN queries each: A used 406,065 original candidate pairs; B used 517,417 repaired candidate pairs. CatBoost uses an Apache-2.0 licensed implementation; these are trained tree ensembles, not pretrained foundation models.

Final policy: India and US: mean of A and B probabilities >= 0.6; France: B >= 0.665. Thresholds were selected on tuning data. Final inference reused cached legacy features by exact pair identity; models and feature order were frozen.

## 4. Experiments, results and error analysis

Public leaderboard: original A 0.764; all-country B 0.787; country hybrid 0.785. The 0.787 submission remains the best confirmed result. This package is a final experimental blend; its public score was unavailable at packaging.

A fresh 6,000-query DEV sample excluded known earlier evaluation samples and was split into 3,000 tuning and 3,000 confirmation queries. All candidates came from legacy retrieval against the full TRAIN+DEV target pool. The initial selection requiring tuning gains in both countries retained B/.665. A secondary aggregate-only tuning selection chose equal A/B weights and threshold 0.6, with France unchanged.

On confirmation, the blend scored macro F0.5 0.823371 versus B 0.821846 (difference +0.001524; 95% paired-bootstrap interval [-0.001717, +0.004867]). India: 0.768947 versus 0.768531; US: 0.857011 versus 0.854802. These results do NOT establish a statistically reliable improvement. No further threshold changes were made after this comparison. France has no labeled validation data.

## 5. Reproducibility, validation and conclusion

The ZIP contains both output TSVs, both frozen model files, training packets/manifests, source for experiments/training/inference, pinned runtime dependencies and exact raw-data reproduction commands. The final production entry point is code/business_entity_resolution/src/project_snapshot/scripts/reproduce_final.py. It can reconstruct both outputs from raw test TSVs or verify/reuse the original feature caches. GPU retraining is nondeterministic; supplied weights define the submitted model.

Independent checks verified exact reference coverage, duplicate-free IDs, target existence, identical candidate mappings, match subsets and country-policy assignment. A small raw-input three-country smoke run passed the final reproduction entry point and official batchwise checks. Structural validity does not prove accuracy. Cache reuse made deployment feasible, while retrieval exclusions and domain shift remain the principal limitations.

## Exact final policy

For India and US, predict a link when (p_A + p_B)/2 >= 0.6. For France, retain p_B >= 0.665. Run reproduce_final.py with --policy blend. The raw-input three-country blend smoke test passed. Full raw-data reproduction was not repeated; final production reused hash-verified cached features.
