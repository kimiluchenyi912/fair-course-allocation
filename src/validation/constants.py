from __future__ import annotations

CONFIG_COLUMNS = {
    "grade_profiles.csv": [
        "grade",
        "student_count",
        "share_5_classes",
        "share_6_classes",
        "share_7_classes",
        "allowed_free_periods",
        "source_type",
        "notes",
    ],
    "capacity_rules.csv": [
        "rule_id",
        "course_category",
        "default_capacity",
        "capacity_min",
        "capacity_max",
        "expansion_threshold_ratio",
        "default_min_sections",
        "default_max_sections",
        "expansion_allowed",
        "source_type",
        "notes",
    ],
    "course_catalog.csv": [
        "course_id",
        "course_name",
        "department",
        "course_category",
        "eligible_grades",
        "periods_required",
        "occupies_school_period",
        "schedule_structure",
        "capacity_override",
        "max_sections_override",
        "demand_tier",
        "protected_core",
        "known_capacity_risk",
        "source_type",
        "confidence",
        "notes",
    ],
    "demand_scenarios.csv": [
        "scenario_id",
        "scenario_name",
        "random_seed",
        "core_multiplier",
        "mainstream_multiplier",
        "popular_multiplier",
        "niche_multiplier",
        "fixed_limited_multiplier",
        "capacity_risk_course_multiplier",
        "affected_capacity_risk_courses",
        "notes",
    ],
    "linked_course_blocks.csv": [
        "block_template_id",
        "course_id",
        "block_type",
        "semester_1_content",
        "semester_2_content",
        "period_sharing_rule",
        "allowed_level_mix",
        "source_type",
        "notes",
    ],
    "grade_request_rules.csv": [
        "grade",
        "rule_key",
        "rule_value",
        "source_type",
        "notes",
    ],
    "course_choice_weights.csv": [
        "scenario_id",
        "grade",
        "scope_type",
        "scope_id",
        "weight",
        "source_type",
        "notes",
    ],
    "fixed_course_targets.csv": [
        "scenario_id",
        "grade",
        "course_id",
        "target_count",
        "min_count",
        "max_count",
        "source_type",
        "notes",
    ],
    "section_capacity_overrides.csv": [
        "scenario_id",
        "course_id",
        "capacity",
        "source_type",
        "notes",
    ],
    "section_planning_rules.csv": [
        "rule_id",
        "rule_value",
        "source_type",
        "notes",
    ],
    "math_fallbacks.csv": [
        "source_course_id",
        "fallback_course_id",
        "policy_type",
        "enabled",
        "source_type",
        "notes",
    ],
}

TEMPLATE_COLUMNS = {
    "students.csv": [
        "student_id",
        "grade",
        "target_course_count",
        "unscheduled_preference",
        "random_seed_group",
        "priority_protected",
        "priority_reason",
        "priority_valid_school_year",
    ],
    "requests.csv": [
        "student_id",
        "course_id",
        "request_type",
        "request_rank",
        "request_group",
        "must_share_block_id",
    ],
    "sections.csv": [
        "scenario_id",
        "section_id",
        "course_id",
        "period_1",
        "period_2",
        "semester",
        "capacity",
        "block_id",
        "linked_section_group_id",
        "logical_block_id",
        "semester_content",
        "planning_source",
        "teacher_resource_id",
        "room_resource_id",
    ],
    "assignments.csv": [
        "student_id",
        "section_id",
        "course_id",
        "request_type",
        "assignment_source",
        "replaced_primary_course_id",
        "replaced_primary_block_id",
        "period_1",
        "period_2",
        "assignment_reason",
    ],
    "metrics.csv": [
        "scenario_id",
        "algorithm",
        "random_seed",
        "complete_schedule_rate",
        "primary_fulfillment_rate",
        "alternate_use_rate",
        "mean_unmet_primary",
        "max_unmet_primary",
        "assignment_churn_rate",
        "solve_time_seconds",
    ],
    "unmet_requests.csv": [
        "student_id",
        "course_id",
        "logical_block_id",
        "request_type",
        "reason_code",
        "candidate_sections",
        "replacement_course_id",
        "replacement_alternate_rank",
        "replacement_period_units",
        "earns_next_year_priority",
    ],
    "student_outcomes.csv": [
        "student_id",
        "primary_unmet_count",
        "alternate_assigned_count",
        "schedule_complete",
        "earns_next_year_priority",
    ],
}

BASELINE_GRADE_PROFILES = {
    9: (700, 0.00, 0.10, 0.90),
    10: (650, 0.00, 0.20, 0.80),
    11: (640, 0.05, 0.35, 0.60),
    12: (640, 0.20, 0.47, 0.33),
}

BASELINE_CAPACITY_RULES = {
    "normal_academic": {"default_capacity": 40, "expansion_threshold_ratio": 0.50},
    "ap_csa": {"default_capacity": 25, "expansion_threshold_ratio": 0.50},
    "pe": {
        "default_capacity": 50,
        "capacity_min": 40,
        "capacity_max": 60,
        "expansion_threshold_ratio": 0.50,
    },
    "niche": {
        "default_min_sections": 1,
        "expansion_threshold_ratio": 0.50,
    },
    "fixed_limited": {"expansion_threshold_ratio": 0.50},
    "dual_enrollment": {"default_capacity": 40, "capacity_max": 45, "expansion_threshold_ratio": 0.50},
}

BASELINE_COURSE_GRADES = {
    "AP_EURO": ["12"],
    "AP_ART_HISTORY": ["11", "12"],
}

VALID_GRADES = {"9", "10", "11", "12"}
VALID_PERIODS = {f"P{i}" for i in range(1, 8)}
ALLOWED_FREE_PERIODS = {"P1", "P6", "P7"}
VALID_STRUCTURES = {"standard", "double_period", "semester_block"}
VALID_REQUEST_TYPES = {"primary", "alternate"}
VALID_UNSCHEDULED_PREFERENCES = {"morning", "afternoon", "either", "none"}
VALID_SEMESTERS = {"full_year", "semester_1", "semester_2", "paired"}
VALID_UNMET_REASON_CODES = {
    "capacity",
    "period_conflict",
    "section_not_opened",
    "load_limit",
    "other",
}
VALID_ASSIGNMENT_SOURCES = {"primary", "alternate"}
VALID_PRIORITY_REASONS = {"", "prior_year_unmet_primary"}
VALID_WEIGHT_SCOPES = {"demand_tier", "department", "course"}
BASELINE_FIXED_TARGETS = {
    ("stable_year", 10, "AP_CALC_AB"): (4, 6, 8),
    ("stable_year", 10, "AP_CALC_BC"): (20, 20, 20),
    ("stable_year", 11, "AP_STATS"): (50, 50, 50),
    ("stable_year", 12, "AP_STATS"): (75, 75, 75),
    ("stable_year", 11, "CALC_D_LINALG"): (20, 20, 20),
    ("stable_year", 12, "CALC_D_LINALG"): (30, 30, 30),
}
