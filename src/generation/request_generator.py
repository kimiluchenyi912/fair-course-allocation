from __future__ import annotations

from random import Random

import pandas as pd

from .models import GenerationConfig, GenerationConfigError
from .rules import (
    catalog_by_id,
    choose_weighted,
    course_weight,
    eligible_courses,
    is_excluded_from_v1,
    is_gov_econ,
    parse_weight_map,
    period_units,
    rule_value,
)


class RequestBuilder:
    def __init__(self, config: GenerationConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = Random(seed)
        self.fixed_rng = Random(f"{seed}:fixed_targets")
        self.catalog = catalog_by_id(config.catalog)
        self.fixed_targets = self._assign_fixed_targets()
        self.fixed_target_courses_by_grade = self._fixed_target_courses_by_grade()
        self.elective_candidates = self._build_elective_candidates()

    def generate_requests(self, students: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for student in students.itertuples(index=False):
            rows.extend(self._primary_requests_for_student(student))
        rows.extend(self._alternate_requests(students, rows))
        return pd.DataFrame(
            rows,
            columns=[
                "student_id",
                "course_id",
                "request_type",
                "request_rank",
                "request_group",
                "must_share_block_id",
            ],
        )

    def _primary_requests_for_student(self, student) -> list[dict]:
        rows: list[dict] = []
        state = _StudentRequestState(student.student_id, int(student.grade), int(student.target_course_count))
        if state.grade == 9:
            self._grade9(state, rows)
        elif state.grade == 10:
            self._grade10(state, rows)
        elif state.grade == 11:
            self._grade11(state, rows)
        elif state.grade == 12:
            self._grade12(state, rows)
        self._fill_remaining_with_electives(state, rows)
        if state.units != state.target_units:
            raise GenerationConfigError(
                f"Could not fill target load for {state.student_id}: "
                f"{state.units}/{state.target_units} units."
            )
        return rows

    def _grade9(self, state, rows: list[dict]) -> None:
        self._add_weighted_rule_course(state, rows, "english_weights")
        self._add_weighted_rule_course(state, rows, "math_weights")
        self._add_weighted_rule_course(state, rows, "science_weights")
        self._add_weighted_rule_course(state, rows, "pe_course_weights", fallback={"PE": 1.0})
        if state.remaining_units > 0 and self._roll_rule(state.grade, "language_probability"):
            self._add_weighted_rule_course(state, rows, "language_start_weights")

    def _grade10(self, state, rows: list[dict]) -> None:
        self._add_weighted_rule_course(state, rows, "english_weights")
        self._add_weighted_rule_course(state, rows, "social_studies_weights")
        self._add_weighted_rule_course(state, rows, "science_weights")
        fixed_math = self.fixed_targets.get(state.student_id)
        if fixed_math:
            self._add_course(state, rows, fixed_math)
        else:
            self._add_weighted_rule_course(state, rows, "math_weights")
        if state.remaining_units > 0 and self._roll_rule(state.grade, "pe_probability"):
            self._add_weighted_rule_course(state, rows, "pe_course_weights", fallback={"PE": 1.0})
        if state.remaining_units > 0 and self._roll_rule(state.grade, "language_probability"):
            self._add_weighted_rule_course(state, rows, "language_weights")

    def _grade11(self, state, rows: list[dict]) -> None:
        self._add_weighted_rule_course(state, rows, "english_weights")
        self._add_weighted_rule_course(state, rows, "social_studies_weights")
        fixed_math = self.fixed_targets.get(state.student_id)
        if fixed_math:
            self._add_course(state, rows, fixed_math)
        elif state.remaining_units > 0 and self._roll_rule(state.grade, "math_probability"):
            self._add_weighted_rule_course(state, rows, "math_weights")
        for _ in range(self._science_unit_count(state.grade)):
            if state.remaining_units > 0:
                self._add_weighted_rule_course(state, rows, "science_weights")
        if state.remaining_units > 0 and self._roll_rule(state.grade, "language_probability"):
            self._add_weighted_rule_course(state, rows, "language_weights")

    def _grade12(self, state, rows: list[dict]) -> None:
        self._add_weighted_rule_course(state, rows, "english_weights")
        gov_econ = choose_weighted(self.rng, parse_weight_map(rule_value(self.config, 12, "gov_econ_weights")))
        self._add_gov_econ_block(state, rows, gov_econ)
        fixed_math = self.fixed_targets.get(state.student_id)
        if fixed_math:
            self._add_course(state, rows, fixed_math)
        elif state.remaining_units > 0 and self._roll_rule(state.grade, "math_probability"):
            self._add_weighted_rule_course(state, rows, "math_weights")
        for _ in range(self._science_unit_count(state.grade)):
            if state.remaining_units > 0:
                self._add_weighted_rule_course(state, rows, "science_weights")
        if state.remaining_units > 0 and self._roll_rule(state.grade, "language_probability"):
            self._add_weighted_rule_course(state, rows, "language_weights")

    def _fill_remaining_with_electives(self, state, rows: list[dict]) -> None:
        while state.units < state.target_units:
            course_id = self._choose_elective(
                state.grade,
                state.target_units - state.units,
                state.primary_course_ids,
            )
            self._add_course(state, rows, course_id)

    def _alternate_requests(self, students: pd.DataFrame, primary_rows: list[dict]) -> list[dict]:
        primary_by_student: dict[str, set[str]] = {}
        for row in primary_rows:
            if row["request_type"] == "primary":
                primary_by_student.setdefault(row["student_id"], set()).add(row["course_id"])

        rows: list[dict] = []
        for student in students.itertuples(index=False):
            chosen: set[str] = set()
            excluded = set(primary_by_student.get(student.student_id, set()))
            for rank in range(1, 4):
                course_id = self._choose_elective(int(student.grade), 2, excluded | chosen)
                chosen.add(course_id)
                rows.append(
                    {
                        "student_id": student.student_id,
                        "course_id": course_id,
                        "request_type": "alternate",
                        "request_rank": rank,
                        "request_group": "alternate",
                        "must_share_block_id": "",
                    }
                )
        return rows

    def _add_weighted_rule_course(
        self,
        state,
        rows: list[dict],
        rule_key: str,
        fallback: dict[str, float] | None = None,
    ) -> None:
        weights = fallback or parse_weight_map(rule_value(self.config, state.grade, rule_key))
        for _ in range(50):
            course_id = choose_weighted(self.rng, weights)
            if course_id in self.fixed_target_courses_by_grade.get(state.grade, set()):
                continue
            course = self.catalog[course_id]
            if course_id not in state.primary_course_ids and period_units(course) <= state.remaining_units:
                self._add_course(state, rows, course_id)
                return
        raise GenerationConfigError(f"Could not select unique course for {state.student_id}: {rule_key}")

    def _add_course(self, state, rows: list[dict], course_id: str) -> None:
        course = self.catalog[course_id]
        units = period_units(course)
        if units > state.target_units - state.units:
            raise GenerationConfigError(f"Course {course_id} would overfill {state.student_id}.")
        block_id = course_id if str(course["schedule_structure"]) == "semester_block" else ""
        rows.append(_request_row(state.student_id, course_id, "primary", "", "", block_id))
        state.primary_course_ids.add(course_id)
        state.units += units

    def _add_gov_econ_block(self, state, rows: list[dict], course_id: str) -> None:
        if course_id in state.primary_course_ids:
            raise GenerationConfigError(f"Duplicate Government/Economics block for {state.student_id}.")
        if state.units + 1 > state.target_units:
            raise GenerationConfigError(f"Government/Economics would overfill {state.student_id}.")
        rows.append(_request_row(state.student_id, course_id, "primary", "", "gov_econ_block", course_id))
        rows.append(_request_row(state.student_id, course_id, "primary", "", "gov_econ_block", course_id))
        state.primary_course_ids.add(course_id)
        state.units += 1

    def _choose_elective(self, grade: int, max_units: int, excluded: set[str]) -> str:
        candidates: dict[str, float] = {}
        for course_id, units, weight in self.elective_candidates.get(grade, []):
            if course_id in excluded or units > max_units:
                continue
            candidates[course_id] = weight
        if not candidates:
            raise GenerationConfigError(f"No elective candidates for grade {grade} with {max_units} units.")
        return choose_weighted(self.rng, candidates)

    def _build_elective_candidates(self) -> dict[int, list[tuple[str, int, float]]]:
        candidates_by_grade: dict[int, list[tuple[str, int, float]]] = {}
        for grade in (9, 10, 11, 12):
            candidates: list[tuple[str, int, float]] = []
            for _, course in eligible_courses(self.config.catalog, grade).iterrows():
                course_id = str(course["course_id"])
                if is_excluded_from_v1(course) or is_gov_econ(course_id):
                    continue
                if str(course["protected_core"]).lower() == "true" or str(course["demand_tier"]) == "core":
                    continue
                if str(course["department"]) == "Mathematics":
                    continue
                if str(course["department"]) == "Science" and course_id not in {
                    "AP_ENVIRONMENTAL_SCIENCE",
                    "MARINE_BIOLOGY",
                    "ANATOMY_PHYSIOLOGY",
                    "INTRO_BIOTECH",
                }:
                    continue
                weight = course_weight(self.config, course, self.seed, grade)
                if weight > 0:
                    candidates.append((course_id, period_units(course), weight))
            candidates_by_grade[grade] = candidates
        return candidates_by_grade

    def _roll_rule(self, grade: int, rule_key: str) -> bool:
        return self.rng.random() < float(rule_value(self.config, grade, rule_key))

    def _science_unit_count(self, grade: int) -> int:
        weights = parse_weight_map(rule_value(self.config, grade, "science_units_weights"))
        return int(choose_weighted(self.rng, weights))

    def _assign_fixed_targets(self) -> dict[str, str]:
        assignments: dict[str, str] = {}
        scenario_id = str(self.config.scenario["scenario_id"])
        pools: dict[int, list[str]] = {}
        for row in self.config.fixed_targets.itertuples(index=False):
            if str(row.scenario_id) != scenario_id:
                continue
            grade = int(row.grade)
            if grade not in pools:
                profile = self.config.grade_profiles[self.config.grade_profiles["grade"].astype(int) == grade]
                if profile.empty:
                    raise GenerationConfigError(f"No grade profile for fixed target grade {grade}.")
                count = int(profile.iloc[0]["student_count"])
                pools[grade] = [f"G{grade:02d}_{index:04d}" for index in range(1, count + 1)]
                self.fixed_rng.shuffle(pools[grade])
            pool = pools[grade]
            target = int(row.target_count)
            if len(pool) < target:
                raise GenerationConfigError(f"Not enough students for fixed target {row.course_id}.")
            selected = pool[:target]
            pools[grade] = pool[target:]
            for student_id in selected:
                assignments[student_id] = str(row.course_id)
        return assignments

    def _fixed_target_courses_by_grade(self) -> dict[int, set[str]]:
        scenario_id = str(self.config.scenario["scenario_id"])
        courses_by_grade: dict[int, set[str]] = {}
        for row in self.config.fixed_targets.itertuples(index=False):
            if str(row.scenario_id) == scenario_id:
                courses_by_grade.setdefault(int(row.grade), set()).add(str(row.course_id))
        return courses_by_grade


class _StudentRequestState:
    def __init__(self, student_id: str, grade: int, target_units: int):
        self.student_id = student_id
        self.grade = grade
        self.target_units = target_units
        self.units = 0
        self.primary_course_ids: set[str] = set()

    @property
    def remaining_units(self) -> int:
        return self.target_units - self.units


def _request_row(
    student_id: str,
    course_id: str,
    request_type: str,
    request_rank,
    request_group: str,
    must_share_block_id: str,
) -> dict:
    return {
        "student_id": student_id,
        "course_id": course_id,
        "request_type": request_type,
        "request_rank": request_rank,
        "request_group": request_group,
        "must_share_block_id": must_share_block_id,
    }
