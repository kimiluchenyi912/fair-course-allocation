from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.validation import validate_configuration

from .models import GenerationConfigError
from .student_generator import generate_synthetic_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic student requests.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config-dir", default="data/config")
    parser.add_argument("--templates-dir", default="data/templates")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    validation_report = validate_configuration(args.config_dir, args.templates_dir)
    print(validation_report.to_text())
    if validation_report.errors:
        return 1

    try:
        result = generate_synthetic_dataset(
            args.config_dir,
            args.scenario,
            args.seed,
            templates_dir=args.templates_dir,
        )
    except (GenerationConfigError, ValueError) as exc:
        print(f"Generation FAIL: {exc}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.students.to_csv(output_dir / "students.csv", index=False)
    result.requests.to_csv(output_dir / "requests.csv", index=False)
    result.summary.to_csv(output_dir / "generation_summary.csv", index=False)
    metadata = dict(result.metadata)
    metadata["output_file_hashes"] = {
        "students.csv": _file_hash(output_dir / "students.csv"),
        "requests.csv": _file_hash(output_dir / "requests.csv"),
    }
    with (output_dir / "generation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        "Generation PASS: "
        f"{len(result.students)} students, "
        f"{(result.requests['request_type'] == 'primary').sum()} primary request rows, "
        f"{(result.requests['request_type'] == 'alternate').sum()} alternate request rows"
    )
    print("Students by grade:")
    print(result.students.groupby("grade").size().sort_index().to_string())
    print("Target loads:")
    print(result.students.groupby(["grade", "target_course_count"]).size().sort_index().to_string())
    return 0


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
