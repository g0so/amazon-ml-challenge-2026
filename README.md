# Amazon ML Challenge — learning workspace

**For teammates:** read the [24-page team handbook](output/pdf/team_handbook.pdf)
for private GitHub onboarding, Ubuntu/Windows setup, the full ML roadmap,
validation, team workflow, and final submission instructions.

Start with **`notebooks/01_phase1_foundations.ipynb`**. This project supports your
implementation work: it contains environment setup, bounded previews, structural
audits, and guided exercises. No matcher, model, or submission has been generated.

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
| `src/business_entity_resolution/` | Small path/reader helpers; your reusable ML code goes here later |
| `data/processed/` | Future derived data and saved validation splits |
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

The default audit reads only the first 1,000 rows per file. Those rows are an
inspection preview, NOT a modeling dataset. Neither audit proves global ID
uniqueness, label consistency, or referential integrity.

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
