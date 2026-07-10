from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before this import)
import numpy as np
import pandas as pd


SCHEMA_VERSION = "visualization_artifacts_v1"
RUNNER_NAME = "benchmark_visualizations_v1"
DEFAULT_TOP_N_COURSES = 15
FIGURE_DPI = 150
STANDARD_FIGURE_SIZE = (9.0, 6.0)
GROUPED_FIGURE_SIZE = (10.0, 6.0)
TOP_COURSES_FIGURE_SIZE = (9.0, 8.0)

# Maps the short --algorithms CLI key (as recorded in benchmark_manifest.json
# "algorithms_run") to the canonical algorithm_name used inside the CSV
# artifacts. This mirrors src/benchmark_runner.py's algorithm key set.
CLI_KEY_TO_ALGORITHM_NAME = {
    "random": "seeded_random_greedy",
    "fcfs": "first_come_first_served_greedy",
    "grade_priority": "grade_priority_greedy",
    "constrained": "constrained_first_greedy",
    "cp_sat": "fair_cp_sat_solver_v1_2",
}

ALGORITHM_DISPLAY_NAMES = {
    "seeded_random_greedy": "Seeded Random",
    "first_come_first_served_greedy": "FCFS",
    "grade_priority_greedy": "Grade Priority",
    "constrained_first_greedy": "Constrained First",
    "fair_cp_sat_solver_v1_2": "CP-SAT",
}

GRADE_ORDER = (9, 10, 11, 12)

GRADE_LEVEL_UNMET_RATE_DEFINITION = (
    "For each (algorithm, grade) pair: the share of students in that grade, under that "
    "algorithm, with primary_unmet_count > 0 in student_outcomes.csv. Denominator is the "
    "count of student_outcomes.csv rows for that algorithm and grade; numerator is the "
    "count of those rows with primary_unmet_count > 0."
)

KNOWN_LIMITATIONS = (
    "CP-SAT was not run as part of this benchmark; every result shown is a greedy baseline "
    "(no repair, backtracking, or infeasibility proof).",
    "None of these results represent a global optimum, and no FEASIBLE result is described as OPTIMAL.",
    "This benchmark run contains zero priority_protected students in student_outcomes.csv for "
    "every algorithm evaluated, so protected-student policy behavior cannot be validated from "
    "this run even though protected_violations reads 0 for all algorithms.",
    "request_outcomes.csv (when present) records only candidate_attempts_count, not a per-attempt "
    "rejection reason, so period-conflict/capacity/duplicate-course failure breakdowns cannot be "
    "computed or plotted from these artifacts.",
)

REQUIRED_ALGORITHM_SUMMARY_COLUMNS = (
    "algorithm_name",
    "status",
    "students",
    "primary_assigned",
    "primary_unmet",
    "primary_satisfaction_rate",
    "total_alternates_assigned",
    "fully_scheduled_students",
    "ordinary_violations",
    "protected_violations",
)
REQUIRED_COURSE_UNMET_COLUMNS = (
    "algorithm_name",
    "candidate_key",
    "primary_demand",
    "primary_assigned",
    "primary_unmet",
    "primary_unmet_rate",
)
REQUIRED_STUDENT_OUTCOME_COLUMNS = (
    "algorithm_name",
    "student_id",
    "grade",
    "primary_unmet_count",
    "fully_scheduled",
    "priority_protected",
)


class VisualizationInputError(ValueError):
    """Raised when benchmark artifacts are missing, empty, or fail validation."""


@dataclass(frozen=True)
class BenchmarkArtifacts:
    artifact_dir: Path
    manifest: dict[str, Any]
    algorithm_summary: pd.DataFrame
    course_unmet_summary: pd.DataFrame
    student_outcomes: pd.DataFrame
    algorithm_order: tuple[str, ...]
    source_files: dict[str, str]


def load_benchmark_artifacts(artifact_dir: Path) -> BenchmarkArtifacts:
    """Load and validate the benchmark artifacts needed for visualization.

    Raises VisualizationInputError with a message naming the offending file
    and field for any missing directory, missing file, empty file, missing
    column, or invalid (non-numeric, negative, out-of-range, NaN/inf) value.
    """
    if not artifact_dir.is_dir():
        raise VisualizationInputError(f"Artifact directory does not exist or is not a directory: {artifact_dir}")

    manifest_path = artifact_dir / "benchmark_manifest.json"
    manifest = _load_manifest(manifest_path)

    algorithm_summary_path = artifact_dir / "algorithm_summary.csv"
    algorithm_summary = _load_csv(algorithm_summary_path, REQUIRED_ALGORITHM_SUMMARY_COLUMNS)
    _validate_algorithm_summary(algorithm_summary, algorithm_summary_path.name)

    course_unmet_path = artifact_dir / "course_unmet_summary.csv"
    course_unmet_summary = _load_csv(course_unmet_path, REQUIRED_COURSE_UNMET_COLUMNS, allow_empty=True)
    if not course_unmet_summary.empty:
        _validate_course_unmet_summary(course_unmet_summary, course_unmet_path.name)

    student_outcomes_path = artifact_dir / "student_outcomes.csv"
    student_outcomes = _load_csv(student_outcomes_path, REQUIRED_STUDENT_OUTCOME_COLUMNS)
    _validate_student_outcomes(student_outcomes, student_outcomes_path.name)

    known_algorithms = set(algorithm_summary["algorithm_name"])
    _check_algorithm_alignment(known_algorithms, course_unmet_summary, student_outcomes)

    algorithm_order = _resolve_algorithm_order(manifest, algorithm_summary)

    source_files = {
        "benchmark_manifest": manifest_path.name,
        "algorithm_summary": algorithm_summary_path.name,
        "course_unmet_summary": course_unmet_path.name,
        "student_outcomes": student_outcomes_path.name,
    }

    return BenchmarkArtifacts(
        artifact_dir=artifact_dir,
        manifest=manifest,
        algorithm_summary=algorithm_summary,
        course_unmet_summary=course_unmet_summary,
        student_outcomes=student_outcomes,
        algorithm_order=algorithm_order,
        source_files=source_files,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VisualizationInputError(f"Required file is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VisualizationInputError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise VisualizationInputError(f"{path.name} must contain a JSON object.")
    return data


def _load_csv(path: Path, required_columns: Iterable[str], *, allow_empty: bool = False) -> pd.DataFrame:
    if not path.is_file():
        raise VisualizationInputError(f"Required file is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise VisualizationInputError(f"{path.name} is empty (no header row).") from exc
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise VisualizationInputError(f"{path.name} is missing required column(s): {', '.join(missing)}")
    if frame.empty and not allow_empty:
        raise VisualizationInputError(f"{path.name} has a header but no data rows.")
    return frame


def _numeric_column(frame: pd.DataFrame, column: str, source_label: str) -> pd.Series:
    series = pd.to_numeric(frame[column], errors="coerce")
    if series.isna().any():
        raise VisualizationInputError(
            f"{source_label} column {column!r} contains non-numeric or missing values."
        )
    if np.isinf(series.to_numpy(dtype=float)).any():
        raise VisualizationInputError(f"{source_label} column {column!r} contains non-finite (inf) values.")
    return series


def _non_negative_count_column(frame: pd.DataFrame, column: str, source_label: str) -> pd.Series:
    series = _numeric_column(frame, column, source_label)
    if (series < 0).any():
        raise VisualizationInputError(
            f"{source_label} column {column!r} contains negative value(s), which is invalid for a count field."
        )
    return series


def _rate_column(frame: pd.DataFrame, column: str, source_label: str, *, low: float = 0.0, high: float = 1.0) -> pd.Series:
    series = _numeric_column(frame, column, source_label)
    if ((series < low) | (series > high)).any():
        raise VisualizationInputError(
            f"{source_label} column {column!r} contains value(s) outside the valid [{low}, {high}] range."
        )
    return series


def _validate_algorithm_summary(frame: pd.DataFrame, source_label: str) -> None:
    for column in ("primary_assigned", "primary_unmet", "total_alternates_assigned", "fully_scheduled_students", "ordinary_violations", "protected_violations", "students"):
        _non_negative_count_column(frame, column, source_label)
    _rate_column(frame, "primary_satisfaction_rate", source_label)
    if frame["algorithm_name"].isna().any() or (frame["algorithm_name"].astype(str).str.strip() == "").any():
        raise VisualizationInputError(f"{source_label} contains blank algorithm_name value(s).")
    if frame["algorithm_name"].duplicated().any():
        duplicates = sorted(frame.loc[frame["algorithm_name"].duplicated(), "algorithm_name"].unique())
        raise VisualizationInputError(f"{source_label} contains duplicate algorithm_name row(s): {duplicates}")
    zero_student_rows = frame.loc[frame["students"] == 0, "algorithm_name"].tolist()
    if zero_student_rows:
        raise VisualizationInputError(
            f"{source_label} has students == 0 for algorithm(s) {zero_student_rows}; cannot compute a fully-scheduled rate."
        )


def _validate_course_unmet_summary(frame: pd.DataFrame, source_label: str) -> None:
    for column in ("primary_demand", "primary_assigned", "primary_unmet"):
        _non_negative_count_column(frame, column, source_label)
    _rate_column(frame, "primary_unmet_rate", source_label)
    if frame["candidate_key"].isna().any() or (frame["candidate_key"].astype(str).str.strip() == "").any():
        raise VisualizationInputError(f"{source_label} contains blank candidate_key value(s).")


def _validate_student_outcomes(frame: pd.DataFrame, source_label: str) -> None:
    _non_negative_count_column(frame, "primary_unmet_count", source_label)
    grade_series = pd.to_numeric(frame["grade"], errors="coerce")
    if grade_series.isna().any():
        raise VisualizationInputError(f"{source_label} column 'grade' contains non-numeric or missing values.")
    unexpected_grades = sorted(set(grade_series.astype(int).unique()) - set(GRADE_ORDER))
    if unexpected_grades:
        raise VisualizationInputError(
            f"{source_label} column 'grade' contains value(s) outside the expected {GRADE_ORDER}: {unexpected_grades}"
        )
    for column in ("fully_scheduled", "priority_protected"):
        series = frame[column]
        if series.dtype != bool:
            coerced = series.astype(str).str.strip().str.lower()
            if not coerced.isin({"true", "false"}).all():
                raise VisualizationInputError(
                    f"{source_label} column {column!r} must be boolean (True/False); found other value(s)."
                )
    if frame["student_id"].isna().any() or (frame["student_id"].astype(str).str.strip() == "").any():
        raise VisualizationInputError(f"{source_label} contains blank student_id value(s).")


def _check_algorithm_alignment(
    known_algorithms: set[str],
    course_unmet_summary: pd.DataFrame,
    student_outcomes: pd.DataFrame,
) -> None:
    if not course_unmet_summary.empty:
        unexpected_course = sorted(set(course_unmet_summary["algorithm_name"]) - known_algorithms)
        if unexpected_course:
            raise VisualizationInputError(
                "course_unmet_summary.csv references algorithm_name(s) not present in "
                f"algorithm_summary.csv: {unexpected_course}"
            )
    student_algorithms = set(student_outcomes["algorithm_name"])
    unexpected_student = sorted(student_algorithms - known_algorithms)
    if unexpected_student:
        raise VisualizationInputError(
            "student_outcomes.csv references algorithm_name(s) not present in "
            f"algorithm_summary.csv: {unexpected_student}"
        )
    missing_student = sorted(known_algorithms - student_algorithms)
    if missing_student:
        raise VisualizationInputError(
            "student_outcomes.csv is missing row(s) for algorithm_name(s) present in "
            f"algorithm_summary.csv: {missing_student}; the grade-level chart requires "
            "per-student data for every evaluated algorithm."
        )


def _resolve_algorithm_order(manifest: dict[str, Any], algorithm_summary: pd.DataFrame) -> tuple[str, ...]:
    """Determine a stable algorithm display order.

    Prefers benchmark_manifest.json's "algorithms_run" (the CLI key order the
    benchmark was actually executed in), mapped to canonical algorithm_name
    values. Falls back to algorithm_summary.csv row order (which is itself
    execution order, per src/benchmark_runner.py). Never invents an
    algorithm that is not an actual row in algorithm_summary.csv.
    """
    known_algorithms = tuple(algorithm_summary["algorithm_name"])
    known_set = set(known_algorithms)
    algorithms_run = manifest.get("algorithms_run")
    if isinstance(algorithms_run, list) and algorithms_run:
        ordered: list[str] = []
        for cli_key in algorithms_run:
            algorithm_name = CLI_KEY_TO_ALGORITHM_NAME.get(cli_key, cli_key)
            if algorithm_name in known_set and algorithm_name not in ordered:
                ordered.append(algorithm_name)
        remaining = [name for name in known_algorithms if name not in ordered]
        if ordered:
            return tuple(ordered) + tuple(remaining)
    return known_algorithms


def _display_name(algorithm_name: str) -> str:
    return ALGORITHM_DISPLAY_NAMES.get(algorithm_name, algorithm_name)


def _ordered_summary(artifacts: BenchmarkArtifacts) -> pd.DataFrame:
    indexed = artifacts.algorithm_summary.set_index("algorithm_name", drop=False)
    return indexed.loc[list(artifacts.algorithm_order)].reset_index(drop=True)


def _new_axes(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    return fig, ax


def _save_and_close(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def render_algorithm_primary_outcomes(artifacts: BenchmarkArtifacts, output_path: Path) -> None:
    summary = _ordered_summary(artifacts)
    labels = [_display_name(name) for name in summary["algorithm_name"]]
    assigned = summary["primary_assigned"].to_numpy()
    unmet = summary["primary_unmet"].to_numpy()

    fig, ax = _new_axes(GROUPED_FIGURE_SIZE)
    x = np.arange(len(labels))
    ax.bar(x, assigned, label="Assigned", color="#2c7fb8")
    ax.bar(x, unmet, bottom=assigned, label="Unmet", color="#d95f0e")
    totals = assigned + unmet
    for index, (assigned_value, unmet_value, total_value) in enumerate(zip(assigned, unmet, totals)):
        ax.text(index, assigned_value / 2, str(int(assigned_value)), ha="center", va="center", color="white", fontsize=9)
        if unmet_value > 0:
            # Placed just above the bar (not centered in the unmet segment)
            # so thin unmet slivers never overlap or clip the label text.
            ax.text(index, total_value, f"unmet: {int(unmet_value)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, float(totals.max()) * 1.1)
    ax.set_ylabel("Number of primary requests")
    ax.set_title("Primary Request Outcomes by Algorithm")
    ax.legend(loc="lower right")
    _save_and_close(fig, output_path)


def render_algorithm_primary_satisfaction(artifacts: BenchmarkArtifacts, output_path: Path) -> None:
    summary = _ordered_summary(artifacts)
    labels = [_display_name(name) for name in summary["algorithm_name"]]
    rates = summary["primary_satisfaction_rate"].to_numpy()

    fig, ax = _new_axes(STANDARD_FIGURE_SIZE)
    x = np.arange(len(labels))
    ax.bar(x, rates, color="#31a354")
    for index, rate in enumerate(rates):
        label_y = max(rate - 0.04, 0.02)
        ax.text(index, label_y, f"{rate * 100:.1f}%", ha="center", va="top", color="white", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Primary satisfaction rate")
    ax.set_title("Primary Satisfaction Rate by Algorithm")
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value * 100:.0f}%")
    _save_and_close(fig, output_path)


def render_algorithm_fully_scheduled(artifacts: BenchmarkArtifacts, output_path: Path) -> None:
    summary = _ordered_summary(artifacts)
    labels = [_display_name(name) for name in summary["algorithm_name"]]
    counts = summary["fully_scheduled_students"].to_numpy()
    totals = summary["students"].to_numpy()
    rates = counts / totals

    fig, ax = _new_axes(STANDARD_FIGURE_SIZE)
    x = np.arange(len(labels))
    ax.bar(x, counts, color="#756bb1")
    for index, (count, rate) in enumerate(zip(counts, rates)):
        ax.text(index, count, f"{int(count)} ({rate * 100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(totals) * 1.12)
    ax.set_ylabel("Number of fully scheduled students")
    ax.set_title("Fully Scheduled Students by Algorithm")
    _save_and_close(fig, output_path)


def render_algorithm_policy_violations(artifacts: BenchmarkArtifacts, output_path: Path) -> None:
    summary = _ordered_summary(artifacts)
    labels = [_display_name(name) for name in summary["algorithm_name"]]
    ordinary = summary["ordinary_violations"].to_numpy()
    protected = summary["protected_violations"].to_numpy()

    fig, ax = _new_axes(GROUPED_FIGURE_SIZE)
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(
        x - width / 2,
        ordinary,
        width,
        label="Ordinary students exceeding the allowed primary-unmet limit",
        color="#e6550d",
    )
    ax.bar(
        x + width / 2,
        protected,
        width,
        label="Protected students with unmet primary requests",
        color="#3182bd",
    )
    for index, (ordinary_value, protected_value) in enumerate(zip(ordinary, protected)):
        ax.text(index - width / 2, ordinary_value, str(int(ordinary_value)), ha="center", va="bottom", fontsize=9)
        ax.text(index + width / 2, protected_value, str(int(protected_value)), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of students")
    ax.set_title("Policy Violations by Algorithm")
    ax.legend(loc="upper right", fontsize=8)
    _save_and_close(fig, output_path)


def compute_grade_unmet_rates(artifacts: BenchmarkArtifacts) -> pd.DataFrame:
    """Compute the grade-level unmet rate table used by the grade chart.

    See GRADE_LEVEL_UNMET_RATE_DEFINITION for the exact definition.
    """
    student_outcomes = artifacts.student_outcomes.copy()
    student_outcomes["grade"] = pd.to_numeric(student_outcomes["grade"], errors="coerce").astype(int)
    student_outcomes["has_unmet_primary"] = student_outcomes["primary_unmet_count"] > 0
    grouped = (
        student_outcomes.groupby(["algorithm_name", "grade"])["has_unmet_primary"]
        .agg(unmet_rate="mean", student_count="count")
        .reset_index()
    )
    return grouped


def render_grade_primary_unmet_rate(artifacts: BenchmarkArtifacts, output_path: Path) -> None:
    rates = compute_grade_unmet_rates(artifacts)
    fig, ax = _new_axes(GROUPED_FIGURE_SIZE)
    algorithm_count = len(artifacts.algorithm_order)
    width = 0.8 / max(algorithm_count, 1)
    x = np.arange(len(GRADE_ORDER))
    max_rate = 0.0
    for offset_index, algorithm_name in enumerate(artifacts.algorithm_order):
        subset = rates[rates["algorithm_name"] == algorithm_name].set_index("grade")
        values = np.array([subset["unmet_rate"].get(grade, 0.0) for grade in GRADE_ORDER])
        max_rate = max(max_rate, float(values.max()) if len(values) else 0.0)
        offset = (offset_index - (algorithm_count - 1) / 2) * width
        ax.bar(x + offset, values, width, label=_display_name(algorithm_name))
    ax.set_xticks(x)
    ax.set_xticklabels([f"Grade {grade}" for grade in GRADE_ORDER])
    ax.set_ylim(0.0, min(1.0, max(max_rate * 1.2, 0.1)))
    ax.set_ylabel("Share of students with a primary unmet request")
    ax.set_title("Grade-Level Primary Unmet Rate by Algorithm")
    ax.legend(fontsize=8)
    _save_and_close(fig, output_path)


def _top_unmet_courses(course_unmet_summary: pd.DataFrame, algorithm_name: str, top_n: int) -> pd.DataFrame:
    subset = course_unmet_summary[course_unmet_summary["algorithm_name"] == algorithm_name]
    subset = subset[subset["primary_unmet"] > 0]
    subset = subset.sort_values(by=["primary_unmet", "candidate_key"], ascending=[False, True])
    return subset.head(top_n)


def render_top_unmet_courses(
    artifacts: BenchmarkArtifacts,
    algorithm_name: str,
    output_path: Path,
    top_n: int,
) -> None:
    top_courses = _top_unmet_courses(artifacts.course_unmet_summary, algorithm_name, top_n)
    fig, ax = _new_axes(TOP_COURSES_FIGURE_SIZE)
    if top_courses.empty:
        ax.text(
            0.5,
            0.5,
            "No unmet primary requests",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        courses = top_courses["candidate_key"].tolist()[::-1]
        unmet = top_courses["primary_unmet"].tolist()[::-1]
        y = np.arange(len(courses))
        ax.barh(y, unmet, color="#c51b8a")
        for index, value in enumerate(unmet):
            ax.text(value, index, f" {int(value)}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(courses, fontsize=8)
        ax.set_xlabel("Primary unmet count")
    ax.set_title(f"Top Unmet Primary Courses — {_display_name(algorithm_name)}")
    _save_and_close(fig, output_path)


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_metrics_table(artifacts: BenchmarkArtifacts) -> pd.DataFrame:
    summary = _ordered_summary(artifacts)
    table = summary[
        [
            "algorithm_name",
            "primary_assigned",
            "primary_unmet",
            "primary_satisfaction_rate",
            "total_alternates_assigned",
            "fully_scheduled_students",
            "ordinary_violations",
            "protected_violations",
        ]
    ].copy()
    table["students"] = summary["students"]
    table["fully_scheduled_rate"] = table["fully_scheduled_students"] / table["students"]
    table["display_name"] = table["algorithm_name"].map(_display_name)
    return table


def render_markdown_summary(artifacts: BenchmarkArtifacts, top_n_courses: int) -> str:
    manifest_data = artifacts.manifest.get("manifest", {})
    table = build_metrics_table(artifacts)

    best_row = table.loc[table["primary_unmet"].idxmin()]
    worst_row = table.loc[table["primary_unmet"].idxmax()]

    total_protected = int(artifacts.student_outcomes["priority_protected"].sum())
    total_protected_violations = int(artifacts.algorithm_summary["protected_violations"].sum())

    lines: list[str] = []
    lines.append("# Benchmark Visualization Summary")
    lines.append("")
    lines.append("## 1. Manifest / fingerprint")
    lines.append("")
    lines.append(f"- Scenario: `{manifest_data.get('scenario_id', 'unknown')}`")
    lines.append(
        f"- Seeds: data_generation={manifest_data.get('data_generation_seed', 'unknown')}, "
        f"section_planning={manifest_data.get('section_planning_seed', 'unknown')}, "
        f"solver={manifest_data.get('solver_seed', 'unknown')}"
    )
    lines.append(f"- Students: {manifest_data.get('students', 'unknown')}")
    lines.append(f"- Logical requests: {manifest_data.get('logical_requests', 'unknown')} (primaries: {manifest_data.get('logical_primaries', 'unknown')}, alternates: {manifest_data.get('alternates', 'unknown')})")
    lines.append(f"- Logical sections: {manifest_data.get('logical_sections', 'unknown')} (section rows: {manifest_data.get('section_rows', 'unknown')})")
    lines.append(f"- Canonical input hash: `{manifest_data.get('canonical_input_hash', 'unknown')}`")
    lines.append("")
    lines.append("## 2. Algorithms detected")
    lines.append("")
    for algorithm_name in artifacts.algorithm_order:
        lines.append(f"- `{algorithm_name}` ({_display_name(algorithm_name)})")
    lines.append("")
    lines.append("## 3. Key metrics")
    lines.append("")
    lines.append("| Algorithm | Primary assigned | Primary unmet | Satisfaction | Alternates assigned | Fully scheduled | Ordinary violations | Protected violations |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in table.iterrows():
        lines.append(
            f"| {row['display_name']} | {int(row['primary_assigned'])} | {int(row['primary_unmet'])} | "
            f"{_fmt_pct(row['primary_satisfaction_rate'])} | {int(row['total_alternates_assigned'])} | "
            f"{int(row['fully_scheduled_students'])} ({_fmt_pct(row['fully_scheduled_rate'])}) | "
            f"{int(row['ordinary_violations'])} | {int(row['protected_violations'])} |"
        )
    lines.append("")
    lines.append("## 4. Observations")
    lines.append("")
    lines.append(
        f"- Among the {len(table)} evaluated greedy baselines, **{best_row['display_name']}** has the lowest "
        f"primary_unmet ({int(best_row['primary_unmet'])}) and the highest primary satisfaction "
        f"({_fmt_pct(best_row['primary_satisfaction_rate'])}) — the best-performing baseline in this run, "
        "not a proven global optimum."
    )
    if worst_row["algorithm_name"] != best_row["algorithm_name"]:
        lines.append(
            f"- **{worst_row['display_name']}** has the highest primary_unmet "
            f"({int(worst_row['primary_unmet'])}) and the lowest primary satisfaction "
            f"({_fmt_pct(worst_row['primary_satisfaction_rate'])}) among the evaluated baselines in this run."
        )
    if total_protected == 0:
        lines.append(
            f"- Protected policy violations are {total_protected_violations} across all evaluated algorithms, "
            "but this benchmark run contains **zero** priority_protected students in student_outcomes.csv — "
            "this metric cannot be used to validate protected-student policy behavior from this run."
        )
    else:
        lines.append(
            f"- Protected policy violations are {total_protected_violations} across all evaluated algorithms, "
            f"evaluated against {total_protected} priority_protected student row(s) in student_outcomes.csv."
        )
    lines.append(f"- Top-unmet-course charts show up to {top_n_courses} courses per algorithm, sorted by primary_unmet descending with candidate_key ascending as a tie-break.")
    lines.append("")
    lines.append("## 5. Limitations")
    lines.append("")
    for limitation in KNOWN_LIMITATIONS:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_visualization_manifest(
    artifacts: BenchmarkArtifacts,
    generated_files: tuple[str, ...],
    chart_data_sources: dict[str, str],
    top_n_courses: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_by": RUNNER_NAME,
        "source_artifact_dir": str(artifacts.artifact_dir),
        "source_files": artifacts.source_files,
        "source_benchmark_manifest": artifacts.manifest.get("manifest", {}),
        "algorithms_detected": list(artifacts.algorithm_order),
        "generated_files": list(generated_files),
        "chart_data_sources": chart_data_sources,
        "grade_level_unmet_rate_definition": GRADE_LEVEL_UNMET_RATE_DEFINITION,
        "grade_order": list(GRADE_ORDER),
        "top_unmet_courses_default_n": top_n_courses,
        "known_limitations": list(KNOWN_LIMITATIONS),
    }


def render_visualization_artifacts(artifacts: BenchmarkArtifacts, output_dir: Path, *, top_n_courses: int = DEFAULT_TOP_N_COURSES) -> tuple[str, ...]:
    if top_n_courses <= 0:
        raise VisualizationInputError(f"top_n_courses must be a positive integer, got {top_n_courses}.")

    output_dir = Path(output_dir)
    _validate_output_directory_relationship(artifacts.artifact_dir, output_dir)
    _check_output_directory_ownership(output_dir)

    def _populate(staging_dir: Path) -> tuple[str, ...]:
        generated: list[str] = []
        chart_data_sources: dict[str, str] = {}

        outcomes_path = staging_dir / "algorithm_primary_outcomes.png"
        render_algorithm_primary_outcomes(artifacts, outcomes_path)
        generated.append(outcomes_path.name)
        chart_data_sources[outcomes_path.name] = "algorithm_summary.csv"

        satisfaction_path = staging_dir / "algorithm_primary_satisfaction.png"
        render_algorithm_primary_satisfaction(artifacts, satisfaction_path)
        generated.append(satisfaction_path.name)
        chart_data_sources[satisfaction_path.name] = "algorithm_summary.csv"

        fully_scheduled_path = staging_dir / "algorithm_fully_scheduled.png"
        render_algorithm_fully_scheduled(artifacts, fully_scheduled_path)
        generated.append(fully_scheduled_path.name)
        chart_data_sources[fully_scheduled_path.name] = "algorithm_summary.csv"

        violations_path = staging_dir / "algorithm_policy_violations.png"
        render_algorithm_policy_violations(artifacts, violations_path)
        generated.append(violations_path.name)
        chart_data_sources[violations_path.name] = "algorithm_summary.csv"

        grade_path = staging_dir / "grade_primary_unmet_rate.png"
        render_grade_primary_unmet_rate(artifacts, grade_path)
        generated.append(grade_path.name)
        chart_data_sources[grade_path.name] = "student_outcomes.csv"

        for algorithm_name in artifacts.algorithm_order:
            course_path = staging_dir / f"top_unmet_courses_{algorithm_name}.png"
            render_top_unmet_courses(artifacts, algorithm_name, course_path, top_n_courses)
            generated.append(course_path.name)
            chart_data_sources[course_path.name] = "course_unmet_summary.csv"

        summary_path = staging_dir / "visualization_summary.md"
        summary_path.write_text(render_markdown_summary(artifacts, top_n_courses), encoding="utf-8")
        generated.append(summary_path.name)

        # visualization_manifest.json lists itself, matching the existing
        # benchmark_manifest.json convention (DEFAULT_ARTIFACT_FILES includes
        # its own filename), so append the name before building the payload.
        manifest_path = staging_dir / "visualization_manifest.json"
        generated.append(manifest_path.name)
        manifest_payload = build_visualization_manifest(artifacts, tuple(generated), chart_data_sources, top_n_courses)
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        return tuple(generated)

    return _stage_and_replace_output_directory(output_dir, _populate)


def _validate_output_directory_relationship(artifact_dir: Path, output_dir: Path) -> None:
    """Refuse an --output-dir that is, or contains, the benchmark input directory.

    Comparing resolved absolute paths so that a full-directory replacement of
    output_dir can never delete artifact_dir (or output_dir itself when the
    two flags happen to point at the same place).
    """
    artifact_resolved = artifact_dir.resolve()
    output_resolved = output_dir.resolve()
    if artifact_resolved == output_resolved or artifact_resolved.is_relative_to(output_resolved):
        raise VisualizationInputError(
            f"Refusing to use --output-dir {output_dir} because it is the same as, or an ancestor "
            f"of, --artifact-dir {artifact_dir}. Replacing this output directory would delete "
            "benchmark input artifacts. Choose an --output-dir that is not artifact_dir itself and "
            "does not contain artifact_dir."
        )


def _check_output_directory_ownership(output_dir: Path) -> None:
    """Refuse to reuse an existing, non-empty output_dir this tool did not create.

    - Missing output_dir: fine, it will be created.
    - Existing, empty output_dir: fine, safe to populate.
    - Existing, non-empty output_dir with a valid visualization_manifest.json
      whose schema_version matches SCHEMA_VERSION: recognized as a prior run
      of this tool, safe to fully replace.
    - Anything else non-empty (no manifest, unreadable/corrupt manifest, or a
      schema_version mismatch): rejected without deleting anything, since we
      cannot tell whether it holds files the caller cares about.
    """
    if not output_dir.exists():
        return
    if not any(output_dir.iterdir()):
        return

    manifest_path = output_dir / "visualization_manifest.json"
    if not manifest_path.is_file():
        raise VisualizationInputError(
            f"--output-dir {output_dir} already exists, is non-empty, and does not contain a "
            "visualization_manifest.json: existing non-empty directory is not recognized as a "
            "Visualization Artifacts v1 output directory. No files were removed."
        )
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualizationInputError(
            f"--output-dir {output_dir} already exists and is non-empty, but its "
            f"visualization_manifest.json could not be read as valid JSON ({exc}): existing "
            "non-empty directory is not recognized as a Visualization Artifacts v1 output "
            "directory. No files were removed."
        ) from exc
    if not isinstance(manifest_data, dict) or manifest_data.get("schema_version") != SCHEMA_VERSION:
        raise VisualizationInputError(
            f"--output-dir {output_dir} already exists and is non-empty, but its "
            f"visualization_manifest.json does not have schema_version == {SCHEMA_VERSION!r}: "
            "existing non-empty directory is not recognized as a Visualization Artifacts v1 output "
            "directory. No files were removed."
        )


def _stage_and_replace_output_directory(output_dir: Path, populate) -> tuple[str, ...]:
    """Generate all outputs in a staging directory, then replace output_dir with it.

    This is a staged full replacement, not a strictly atomic directory swap:
    building happens in a sibling staging directory (same parent, and so the
    same filesystem, avoiding a cross-filesystem EXDEV on rename), and only
    after that generation succeeds is any existing output_dir removed and the
    staging directory renamed into place. If generation or validation fails,
    the staging directory is cleaned up and output_dir is left untouched.
    Callers must call _validate_output_directory_relationship and
    _check_output_directory_ownership before this function to ensure the
    directory about to be replaced is safe to remove.
    """
    output_dir = Path(output_dir)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".benchmark_visualizations_staging_", dir=parent))
    try:
        generated_files = populate(staging_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.rename(output_dir)
        return generated_files
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render static visualization artifacts from benchmark_runner output.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n-courses", type=int, default=DEFAULT_TOP_N_COURSES)
    args = parser.parse_args(tuple(argv) if argv is not None else None)

    try:
        artifacts = load_benchmark_artifacts(Path(args.artifact_dir))
        generated_files = render_visualization_artifacts(
            artifacts,
            Path(args.output_dir),
            top_n_courses=args.top_n_courses,
        )
    except VisualizationInputError as exc:
        print(f"Visualization input error: {exc}")
        return 1

    print(
        "Visualization PASS: "
        f"{len(artifacts.algorithm_order)} algorithm(s), {len(generated_files)} file(s) written to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
