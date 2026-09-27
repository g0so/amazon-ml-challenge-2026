"""Check local resources and input files. Does not connect to AWS."""
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from business_entity_resolution.workspace import CONFIG, dataset_files


def main():
    packages = {}
    for name in ("jupyterlab", "ipykernel", "pandas", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    memory = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        memory = {"total_gib": round(vm.total / 2**30, 2),
                  "available_gib": round(vm.available / 2**30, 2)}
    except ImportError:
        memory["note"] = "Install project dependencies for a portable RAM check."
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "memory": memory,
        "disk_free_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2),
        "working_memory_target_gib": CONFIG["working_memory_target_gib"],
        "packages": packages,
        "files": [{"file": str(p.relative_to(ROOT)), "exists": p.is_file(),
                   "size_mib": round(p.stat().st_size / 2**20, 2) if p.is_file() else None}
                  for p in dataset_files()],
    }
    out = ROOT / "reports" / "environment.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nThis report checks availability, not model feasibility or data integrity.")
    return 0 if all(f["exists"] for f in report["files"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
