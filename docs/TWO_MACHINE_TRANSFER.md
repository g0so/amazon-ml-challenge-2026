# Two-machine transfer contract

Read `RTX5060_AI_HANDOFF.txt` on the RTX laptop and `UBUNTU_AI_HANDOFF.txt` on the Ubuntu machine. They are ready-to-use AI prompts. The feature exporter is implemented in `scripts/export_gpu_feature_packet.py`, with independent verification in `scripts/verify_gpu_feature_packet.py`. The GPU trainer and challenger test rescorer still require implementation. Do not assume the feature ZIP exists until its export and verification reports confirm completion.

## Git versus private transfer

Commit reviewed source, these documents, tests, dependency specifications and selected small result reports. Avoid committing live logs, scratch checkpoints, raw competition datasets, environments, SQLite databases, feature matrices or generated test outputs. `data/processed/`, raw `dataset/`, `.venv/` and most outputs are already ignored. A Git clone therefore does **not** give the second laptop the data needed for training.

Small saved models in `models/` are currently untracked unless you add them. They can be shared privately or committed deliberately if consistent with team rules. Never commit credentials. The user will commit/push; no push was performed by this handoff task.

The live Ubuntu run is in `/home/a/Projects/Amazon ML hackathon`, whereas this editing worktree may be elsewhere. Commit the intended checkout; do not reset, clean, or switch the active inference checkout mid-run.

## Compact GPU feature packet v1

Write to ignored `data/processed/gpu_feature_packet_v1/`, then ZIP for private transfer. Use stable deterministic ordering. Suggested contents:

```text
metadata.json
README.txt
train/
  X.npy
  y.npy
  reference_ids.json
  countries.json
  offsets.npy
  candidates.jsonl
  truth.jsonl
tune/
  X.npy
  y.npy
  reference_ids.json
  countries.json
  offsets.npy
  candidates.jsonl
  truth.jsonl
  v3_probabilities.npy
```

Schema:

- `X.npy`: shape `(number_of_candidate_pairs, 22)`, finite float64, exact V3 feature order. Do not fit a scaler globally or alter feature definitions.
- `y.npy`: matching length, 0/1 label for each retrieved pair. Derived after retrieval. Train and tune labels remain separate.
- `reference_ids.json`: ordered unique reference IDs, including those with no candidates.
- `countries.json`: same-length ordered country strings, for evaluation only.
- `offsets.npy`: integer array of length `number_of_references + 1`; starts at zero, nondecreasing, ends at the pair count. Reference i owns rows `[offsets[i], offsets[i+1])`; equal offsets represent no candidates.
- `candidates.jsonl`: one row per reference, in manifest order: `{"reference_id":"S1-...","candidate_ids":["S2-...",...]}`. Candidate order must match X/y rows within that reference. No duplicates.
- `truth.jsonl`: one explicit row per reference: `{"reference_id":"S1-...","true_ids":[...]}`. Includes true targets missed by retrieval and empty singleton lists. Never construct this file only from positive candidate pairs.
- `v3_probabilities.npy`: V3 class-1 probabilities in exact tuning pair order. Compare with the new model using the same per-reference scorer, not different samples.

`metadata.json` must record format version, feature names and code hash, dtypes/shapes, source input and split hashes, sample hashes/seeds, retrieval settings, V3 model hash/threshold, versions, class counts, and SHA-256 for every payload file. Validate partition membership, explicit ground-truth coverage, label/pair consistency and finite arrays. Reject altered or incomplete packets.

Keep the seed-2027 secondary-dev feature packet on Ubuntu initially. Send only train/tune to the trainer. The chosen challenger returns to Ubuntu for its locked-threshold secondary check. Do not expose the internal holdout to tuning.

## Reply package from the GPU laptop

The native model, metadata, threshold search, tuning metrics/predictions, minimal environment lock, training source and numeric CPU/GPU parity fixture go under `artifacts/gpu_challenger/`. Return a ZIP privately. The model must consume the same 22 features to reuse the test feature cache. Verify CPU inference on Ubuntu before rescoring complete test batches.

## Start messages

RTX laptop AI: “Read `docs/RTX5060_AI_HANDOFF.txt` and follow it. First inspect this checkout and your GPU environment. Do not assume the feature packet is present.”

Ubuntu AI: “Read `docs/UBUNTU_AI_HANDOFF.txt` and follow it. First inspect the live original project's status and paths. Preserve its active inference process. Implement only new exporter/rescorer files; do not modify frozen code.”

Budget setup to 30–45 minutes. If GPU setup is blocked, use the second laptop's CPU for the bounded model comparison; keep the baseline submission independent.

## Exporter and verifier

On Ubuntu, run the exporter from the editing checkout, explicitly pointing at the live project data:

```bash
"/home/a/Projects/Amazon ML hackathon/.venv/bin/python" scripts/export_gpu_feature_packet.py --data-root "/home/a/Projects/Amazon ML hackathon"
```

Do not launch a duplicate exporter while one is running. The exporter reads the original project, writes only into this checkout, and retains atomic partial batches for identical-input resumption. It checks a 2 GiB available-memory floor during index construction. Completed output is `data/processed/gpu_feature_packet_v1.zip`, with a SHA-256 sidecar.

After extraction on either laptop:

```bash
python scripts/verify_gpu_feature_packet.py path/to/extracted/gpu_feature_packet_v1
```

The ZIP has `metadata.json`, `train/`, and `tune/` at its root; pass that root folder to the verifier. It requires only NumPy and the standard library. Verification checks file hashes, row alignment, label consistency, sample separation, and baseline rescoring.
