# GitHub archive handoff

## Recommended upload

Use the clean snapshot `output/github/Spartans_0.789_source_archive.zip`. Extract its contents into a new repository checkout and upload using Git. It contains the source, documentation, frozen models, compressed final predictions, portable training packets, canonical splits and selected reports. The archive itself is a transport file; commit the extracted files so GitHub can display the source.

The snapshot has no Git history, raw organizer dataset, Python environments, test databases, feature caches or complete submission ZIPs. It is intended to avoid carrying earlier bulky transfer bundles into a new archive repository. Check the competition's data-sharing terms before making organizer-derived training packets or predictions public; a private repository preserves the current sharing scope.

Existing local Git history and previously tracked transfer bundles have been left intact. Ignore rules do not remove already tracked files. The clean snapshot is the simplest way to leave that legacy history out without rewriting or deleting it.

## Complete deliverable as a release asset

For the full final competition package, attach this file to a GitHub Release in the archive repository:

`output/final/blend_experimental/Spartans_submission.zip`

SHA-256: `421ffd8bf9033cdaa871ba42c8c5bd905b18055cde298f3c23103127f4bab209`

The ZIP contains both TSV outputs, code, models and methodology. It was created before evaluation; the accompanying final-selection record documents the later 0.789 result. Upload `archive/final/selection.json` alongside it if helpful. The large candidate TSV is inside this complete package, not the source snapshot.

## Local preservation

No raw data or expensive caches were removed. `archive/local_inventory.json` records their paths and sizes. An inventory is not a backup: retain the local project or a separate disk backup if you want the databases and 27 GB feature arrays later. The original `.git` history is also local and is not part of the clean snapshot.

## Integrity and restoration

Run `python scripts/restore_final_archive.py` from the extracted source tree to verify the archived assets and restore frozen model/prediction paths. `archive/SHA256_MANIFEST.json` records content hashes. The clean source ZIP also has a `.sha256` sidecar.

No repository was created, no commit was made, and nothing was pushed during organization. You control the destination and visibility.
