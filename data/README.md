# Data locations

Original organizer files remain under:
`6ab10eb3b23ba_student_resource/student_resource/dataset/`.

`project_config.json` points to that resource directory. No raw data was moved or
copied. `src/business_entity_resolution/workspace.py` resolves paths relative to
the project, so copying the entire project to another machine preserves them.

Use `data/processed/` for future reproducible derived data and split manifests.
Do not overwrite raw TSVs. Do not put independently sampled source heads into a
training dataset: doing so loses true matches. Previews are inspection only.

Keep row data out of the experiment log. Do not fetch outside business data.
