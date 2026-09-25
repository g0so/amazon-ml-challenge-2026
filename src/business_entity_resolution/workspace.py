"""Portable paths and streaming TSV readers; no matching/modeling logic."""

import csv
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "project_config.json").read_text(encoding="utf-8"))
RESOURCE_DIR = ROOT / CONFIG["resource_dir"]
DATASET_DIR = RESOURCE_DIR / "dataset"
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def dataset_files():
    """Return known source files and ground truth in a stable order."""
    paths = [DATASET_DIR / split / f"{split}_source{source}.tsv"
             for split in ("train", "test") for source in (1, 2, 3)]
    return paths + [DATASET_DIR / "train" / "train_ground_truth.tsv"]


def iter_tsv(path, limit=None):
    """Yield UTF-8 TSV rows as strings; preserve empty fields and check structure.

    Only the current row is held in memory. A limit is for previews, never an
    independently sampled modeling dataset.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative or None")
    path = Path(path)
    expected = TRUTH_COLUMNS if path.name.endswith("ground_truth.tsv") else SOURCE_COLUMNS
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected:
            raise ValueError(f"{path.name}: expected columns {expected}, got {reader.fieldnames}")
        rows = reader if limit is None else itertools.islice(reader, limit)
        for row_number, row in enumerate(rows, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}: malformed TSV record near row {row_number}")
            yield row


def preview(path, limit=5):
    """Materialize only a small preview, never a whole source file."""
    return list(iter_tsv(path, limit=limit))
