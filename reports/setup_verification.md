# Setup verification

Date: 2026-09-25. Ubuntu local workspace, Python 3.13.12.

Completed:
- Created a project-local `.venv`; installed notebook/data-exploration dependencies.
- Recorded all installed package versions in `requirements.txt`.
- Registered the `Amazon ML (local)` kernel inside the project environment.
- Dependency consistency check passed.
- All seven expected raw files exist.
- Notebook format validation passed.
- Executed every notebook code cell sequentially in the project Python interpreter:
  setup and previews passed; unfinished learner checks remained explicitly skipped.
- Reader smoke checks passed for Unicode, empty fields, preview limits, and wrong-delimiter rejection.
- Notebook launcher shell syntax checked.
- Preview structural audit completed; see `data_audit_preview.md`.

Limits:
- These checks do not establish model quality or complete dataset integrity.
- The scoring exercises intentionally remain unimplemented.
- No model training, prediction files, cloud resources, or AWS account changes.
- Notebook code was checked in-process; an interactive browser session was not launched.
- Rebuild `.venv` if moving the project; do not copy an existing virtual environment.

## Full streaming audit

Completed all seven files in 111.75 seconds with peak resident process RAM of
17,552 KiB (about 17.1 MiB), measured by `/usr/bin/time`. Counts agree with the
earlier inspection: 2,206,821 truth rows, 123,247 singletons, and 7,638,365 links.
No duplicate target IDs within individual truth lists or invalid target prefixes
were found. Cross-row ID uniqueness and cross-file references were not checked.
This low memory use applies to this streaming audit, not future retrieval/modeling.
See `data_audit_full.md` and `data_audit_full.json`.
