from __future__ import annotations

import argparse

from .models import SectionPlanningError
from .runner import plan_sections_from_files, write_result_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan synthetic section counts and period layout.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config-dir", default="data/config")
    parser.add_argument("--templates-dir", default="data/templates")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    try:
        result = plan_sections_from_files(
            args.input_dir,
            args.config_dir,
            args.scenario,
            args.seed,
            args.templates_dir,
        )
        write_result_atomic(result, args.output_dir)
    except SectionPlanningError as exc:
        print(f"Section planning FAIL: {exc}")
        return 1

    print(
        "Section planning PASS: "
        f"{result.metadata['total_logical_sections']} logical sections, "
        f"{result.metadata['total_section_rows']} section rows, "
        f"{result.metadata['total_planned_seats']} planned seats, "
        f"{result.metadata['total_remaining_waitlist']} remaining waitlist"
    )
    print("Sections by period:")
    print(result.period_layout_summary[["period", "logical_section_count", "section_row_count", "planned_seats"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
