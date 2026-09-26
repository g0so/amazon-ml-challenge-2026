"""Run the bounded Phase 3 retrieval and validation benchmark."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.phase3_retrieval import run_benchmark
from business_entity_resolution.workspace import CONFIG, DATASET_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "phase3_retrieval.json")
    args = parser.parse_args()
    train = DATASET_DIR / "train"
    result = run_benchmark(
        train / "train_source1.tsv",
        train / "train_ground_truth.tsv",
        [train / "train_source2.tsv", train / "train_source3.tsv"],
        args.sample_size,
    )
    result["seed"] = CONFIG["seed"]
    result["design"] = "stream targets once per route; no full target index or all-pairs matrix"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()