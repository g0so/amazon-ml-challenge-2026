"""Streaming structural audit; default is a 1,000-row preview per file.

Use --full for complete counts. Does not train, normalize, sample training groups,
check global ID uniqueness, or resolve entities.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from business_entity_resolution.workspace import CONFIG, dataset_files, iter_tsv


def audit(path, limit):
    truth = path.name.endswith("ground_truth.tsv")
    countries, blanks, cardinality, links = Counter(), Counter(), Counter(), Counter()
    duplicates = invalid = rows = 0
    for row in iter_tsv(path, limit):
        rows += 1
        blanks.update(key for key, value in row.items() if not value.strip())
        if truth:
            raw = row["matched_entity_ids"]
            ids = [value.strip() for value in raw.split(",")] if raw.strip() else []
            duplicates += len(ids) != len(set(ids))
            invalid += sum(not value.startswith(("S2-", "S3-")) for value in ids)
            cardinality[len(ids)] += 1
            links.update(value.split("-", 1)[0] for value in ids)
        else:
            countries[row["country"]] += 1
    result = {"file": str(path.relative_to(ROOT)), "rows_scanned": rows,
              "blank_fields": dict(blanks)}
    if truth:
        result.update(match_count_distribution=dict(sorted(cardinality.items())),
                      links_by_source=dict(links), duplicate_ids_within_truth_rows=duplicates,
                      invalid_target_prefixes=invalid,
                      empty_prediction_baseline=cardinality[0] / rows if rows else None)
    else:
        result["countries"] = dict(countries)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Stream all rows; bounded memory, more time")
    args = parser.parse_args()
    limit = None if args.full else CONFIG["audit_sample_rows_per_file"]
    results = []
    for path in dataset_files():
        print(f"Reading {path.name}...", flush=True)
        results.append(audit(path, limit))
    scope = "full" if args.full else "preview"
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": scope,
              "row_limit_per_file": limit,
              "limitations": ["No global ID uniqueness or cross-file referential-integrity checks.",
                              "Preview rows are file heads, not a random or representative sample.",
                              "Counts do not establish label correctness or a modeling baseline."],
              "files": results}
    output = ROOT / "reports" / f"data_audit_{scope}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# Data audit ({scope})", "", report["created_at_utc"], "",
             "Streaming counts; no full source table was loaded into memory.", "",
             "| File | Rows scanned | Blank addresses | Country counts |",
             "|---|---:|---:|---|"]
    for result in results:
        if "countries" in result:
            lines.append(f"| {Path(result['file']).name} | {result['rows_scanned']:,} | "
                         f"{result['blank_fields'].get('business_address', 0):,} | "
                         f"{result['countries']} |")
    lines += ["", "## Ground truth", "", "```json", json.dumps(results[-1], indent=2),
              "```", "", "## Limits", ""] + [f"- {item}" for item in report["limitations"]]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved reports/data_audit_{scope}.json and .md")


if __name__ == "__main__":
    main()
