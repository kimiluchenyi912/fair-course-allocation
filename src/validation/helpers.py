from __future__ import annotations

from typing import Iterable

import pandas as pd

from .models import ValidationReport


def has_columns(df: pd.DataFrame | None, columns: Iterable[str]) -> bool:
    return df is not None and all(col in df.columns for col in columns)


def line_number(index: int) -> int:
    return int(index) + 2


def text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def parse_int(value: object) -> int | None:
    value_text = text(value)
    if not value_text:
        return None
    try:
        parsed = float(value_text)
    except ValueError:
        return None
    if not parsed.is_integer():
        return None
    return int(parsed)


def parse_optional_int(value: object) -> int | None:
    return None if text(value) == "" else parse_int(value)


def parse_float(value: object) -> float | None:
    value_text = text(value)
    if not value_text:
        return None
    try:
        return float(value_text)
    except ValueError:
        return None


def parse_bool(value: object) -> bool | None:
    value_text = text(value).lower()
    if value_text == "true":
        return True
    if value_text == "false":
        return False
    return None


def require_nonempty_unique(
    df: pd.DataFrame,
    filename: str,
    column: str,
    duplicate_code: str,
    report: ValidationReport,
) -> None:
    values = df[column].map(text)
    for idx, value in values.items():
        if not value:
            report.add_error("EMPTY_ID", filename, f"{column} cannot be blank.", line_number(idx))

    duplicate_indexes = values[(values != "") & values.duplicated(keep=False)].index
    for idx in duplicate_indexes:
        report.add_error(
            duplicate_code,
            filename,
            f"{column} must be unique.",
            line_number(idx),
            values.loc[idx],
        )


def parse_semicolon_list(
    value: object,
    filename: str,
    column: str,
    line: int,
    report: ValidationReport,
    allow_blank: bool = True,
) -> list[str]:
    value_text = text(value)
    if not value_text:
        if not allow_blank:
            report.add_error("EMPTY_SEMICOLON_LIST", filename, f"{column} cannot be blank.", line)
        return []

    parts = value_text.split(";")
    malformed = (
        any(part == "" for part in parts)
        or any(part.strip() != part for part in parts)
        or len(parts) != len(set(parts))
    )
    if malformed:
        report.add_error(
            "MALFORMED_SEMICOLON_LIST",
            filename,
            f"{column} must use nonempty, trimmed, unique semicolon-separated values.",
            line,
            value_text,
        )
    return parts
