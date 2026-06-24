from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import ValidationReport


def load_tables(
    directory: Path,
    expected_columns: dict[str, list[str]],
    report: ValidationReport,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for filename in expected_columns:
        path = directory / filename
        if not path.exists():
            report.add_error("MISSING_FILE", str(path), "Required CSV file is missing.")
            continue
        try:
            tables[filename] = pd.read_csv(path, keep_default_na=False)
        except Exception as exc:  # pragma: no cover - pandas exception type varies.
            report.add_error("CSV_READ_ERROR", str(path), f"Could not read CSV: {exc}")
    return tables


def validate_generic_tables(
    tables: dict[str, pd.DataFrame],
    expected_columns: dict[str, list[str]],
    report: ValidationReport,
) -> None:
    for filename, expected in expected_columns.items():
        df = tables.get(filename)
        if df is None:
            continue
        actual = list(df.columns)
        index_columns = [col for col in actual if col.startswith("Unnamed") or col == "index"]
        for col in index_columns:
            report.add_error(
                "EXTRA_INDEX_COLUMN",
                filename,
                f"Unexpected index-like column '{col}'.",
            )
        for col in [col for col in expected if col not in actual]:
            report.add_error("MISSING_COLUMN", filename, f"Missing required column '{col}'.")
