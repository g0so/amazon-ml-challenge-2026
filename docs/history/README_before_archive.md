# Amazon ML Challenge — learning workspace

**For teammates:** read the [24-page team handbook](output/pdf/team_handbook.pdf)
for private GitHub onboarding, Ubuntu/Windows setup, the full ML roadmap,
validation, team workflow, and final submission instructions.

Start with **`notebooks/01_phase1_foundations.ipynb`**. This project supports your
implementation work: it contains environment setup, structural audits, verified
validation splits, and two evaluated exact-matching baselines. Phase 2 results
and explanations are in **`notebooks/02_baseline.ipynb`** and
**`reports/phase2_completion.md`**. No test submission has been generated.

## Open the notebook on this Ubuntu machine

From this project directory:

```bash
bash start_notebook.sh
```

Open the localhost URL printed in the terminal. Select **Amazon ML (local)** as
the kernel. Keep the server local; do not disable token authentication. Stop the
server with Ctrl+C when finished. If the kernel appears missing, select the Python
kernel from this project's `.venv`.

## Project map

| Location | Purpose |
|---|---|
| `notebooks/01_phase1_foundations.ipynb` | Your first notebook: working previews, metric exercises, and validation planning |
| `docs/PHASES.md` | Revised 30-hour plan for your actual hardware |
| `docs/PHASE_1.md` | Step-by-step work and troubleshooting |
| `docs/AWS_QUOTAS.md` | Verified explanation of your AWS quota error |
| `project_config.json` | Data location, seed, and conservative starting settings |
| `scripts/check_environment.py` | Check installed packages, RAM, disk, and input files |
| `scripts/audit_data.py` | Stream preview/full file counts without loading entire tables |
| `notebooks/02_baseline.ipynb` | Complete Phase 2 audit, split, baseline results, and error review |
| `scripts/audit_relationships.py` | Full training-ID integrity audit using a disk-backed SQLite database |
| `src/business_entity_resolution/` | Readers, conservative normalization, strict metric, verified splits, disk-backed baseline |
| `data/processed/` | Local databases, preserved split assignments, development predictions (excluded from Git) |
| `reports/` | Audits, environment report, and experiment log |
| `artifacts/` | Future model files and caches |
| `output/` | Future final TSV outputs; `pdf/team_handbook.pdf` is the team guide |

Original PDFs and organizer resource directory remain in their original locations.
The raw dataset has not been moved, copied, or edited.

## Useful checks

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python scripts/audit_data.py
# Optional: streams all seven files, writes full counts, takes longer.
.venv/bin/python scripts/audit_data.py --full
```

The Phase 2 relationship audit has a separate command:

```bash
.venv/bin/python scripts/audit_relationships.py
```

It scans all training labels and source IDs, writes
`data/processed/training_relationships.sqlite`, and saves results to
`reports/phase2_relationship_audit.md` and `.json`. It checks shared targets,
duplicate IDs, source membership, and ground-truth coverage. Its database is
excluded from Git. Read the saved report in `02_baseline.ipynb` without rerunning
the full audit. Each rebuild uses a new temporary database, so it cannot append
duplicate rows to a prior run. Allow free disk space for both databases during a
rebuild. The audit checks labeled relationships, not semantic business identity.

The default `audit_data.py` preview reads only the first 1,000 rows per file;
those rows are NOT a modeling dataset. The separate relationship audit checks
global training ID integrity but cannot prove semantic label correctness.

## Recreate the environment on another machine

Copy the project and original data, but exclude `.venv` and rebuild it. Use the
same Python major/minor version recorded in `reports/setup_verification.md` where
possible. On Ubuntu or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name amazon-ml --display-name 'Amazon ML (local)'
bash start_notebook.sh
```

On Windows PowerShell, use `py -m venv .venv`, replace `.venv/bin/python` with
`.venv\Scripts\python.exe`, and launch JupyterLab using that executable with
`-m jupyterlab notebooks/01_phase1_foundations.ipynb` (the Bash launcher is for Linux).
The pinned file records the verified Ubuntu environment; platform-specific wheel
availability on another OS must be checked rather than assuming compatibility.

`requirements.in` lists the small set of direct dependencies. `requirements.txt`
pins the installed environment after setup. Modeling/GPU packages are deferred.

## Memory rules

- Stream audits and load small previews. Never call `list()` on a full TSV reader.
- Preserve original strings and blank fields. Do not turn missing values into "nan".
- Start with one notebook kernel and at most two compute threads.
- The 4 GiB working-memory target in config is guidance, not an enforced limit.
- Two 16 GB laptops do not form one 32 GB machine; GPU memory is separate from RAM.
- A future complete training sample must retain all labeled variants of each
  selected business. Do not independently sample source heads.
- Full retrieval must use batches/indexes or disk-backed partitions, not an
  all-pairs comparison or dense similarity matrix. Benchmark before scaling.

The existing organizer validator remains in the original `utils/` directory. It
checks formatting, not model quality; enable `--check-ids` when feasible and ensure
both final output files are actually present. Do not submit the learning notebook
as a finished solution package.

## Phase 2: reproduce or read the saved results

Read `notebooks/02_baseline.ipynb` first; its rebuild flags default to False.
To deliberately rerun the split verification and both full development baselines:

```bash
.venv/bin/python scripts/run_phase2.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_phase2_outputs.py
```

Run from this project directory with the original training files and relationship
DB present. On a fresh machine, run the relationship audit first. Transfer the
existing `split_manifest.json` privately to preserve the exact team assignments;
the pipeline imports it into `phase2_split.sqlite` and checks IDs against the audit.
Without that JSON, the original seeded recipe is reconstructed; compare the
reported split ID before comparing experiments across machines.

SQLite keeps indexes on disk. The one-time legacy JSON import and development ID
sets still use RAM; measured process memory is in `reports/phase2_run.json`.
Keep several GB of free disk for databases, indexes, and temporary rebuilds.
The original three Phase 2 modules were backed up under
`data/processed/phase2_originals/` before corrections. Existing scratch scripts
remain available, but the supported end-to-end entry point is `run_phase2.py`.

Reports record input hashes and split identity. Both development prediction TSVs
include every development reference, including empty predictions, and are saved
under `data/processed/phase2/`. They are evaluation artifacts, not final test
submissions. Holdout scores have not been opened. See the completion report for
limitations before starting Phase 3 retrieval.
