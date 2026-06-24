from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    file: str
    message: str
    row: int | None = None
    identifier: str | None = None

    def format(self) -> str:
        parts = [self.code, self.file]
        if self.row is not None:
            parts.append(f"row {self.row}")
        if self.identifier:
            parts.append(str(self.identifier))
        return f"{' | '.join(parts)}: {self.message}"


@dataclass
class ValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add_error(
        self,
        code: str,
        file: str,
        message: str,
        row: int | None = None,
        identifier: str | None = None,
    ) -> None:
        self.errors.append(ValidationIssue(code, file, message, row, identifier))

    def add_warning(
        self,
        code: str,
        file: str,
        message: str,
        row: int | None = None,
        identifier: str | None = None,
    ) -> None:
        self.warnings.append(ValidationIssue(code, file, message, row, identifier))

    def add_policy_issue(
        self,
        code: str,
        file: str,
        message: str,
        strict_policy: bool,
        row: int | None = None,
        identifier: str | None = None,
    ) -> None:
        if strict_policy:
            self.add_error(code, file, message, row, identifier)
        else:
            self.add_warning(code, file, message, row, identifier)

    def summary(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"Validation {status}: {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s)"
        )

    def to_text(self) -> str:
        lines = [self.summary()]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"- {issue.format()}" for issue in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {issue.format()}" for issue in self.warnings)
        return "\n".join(lines)
