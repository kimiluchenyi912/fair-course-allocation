# AGENTS.md

## Project Purpose

This project builds a fair and explainable high-school course allocation system.

The primary goal is correct and transparent optimization, not visual complexity.

## Current Scope

Version 1 assigns students to course sections that already have:

- fixed course offerings,
- fixed periods,
- fixed capacities,
- and counselor-approved course requests.

The solver does not decide prerequisites, honors eligibility, AP eligibility,
or graduation eligibility. Those decisions happen before requests become
solver input.

Do not add teacher scheduling, classroom scheduling, or master-schedule generation unless explicitly requested.

## Technology

Use:

- Python
- pandas
- Pydantic
- Google OR-Tools CP-SAT
- pytest

Do not add new dependencies without first explaining why they are necessary.

## Development Rules

1. Read `PLAN.md` before making major changes.
2. Keep hard constraints separate from optimization objectives.
3. Do not change the meaning of fairness without explicit approval.
4. Do not remove or silently relax constraints to make the solver work.
5. Clearly report invalid data and infeasible allocation problems.
6. Use deterministic random seeds in tests and experiments.
7. Use only synthetic or anonymized student data.
8. Prefer simple, readable code over unnecessary abstraction.
9. Do not build the website until the allocation model and metrics are tested.
10. Do not implement features outside the current requested task.

## Testing Rules

1. Every hard constraint must have at least one automated test.
2. Every bug fix must include a regression test.
3. Use small, hand-verifiable datasets before testing larger datasets.
4. Run the full test suite before declaring a task complete.
5. Do not claim that code works without showing the test result.

## Task Completion Report

After completing a coding task, report:

1. Files created or changed
2. What was implemented
3. Tests that were run
4. Test results
5. Known limitations or unanswered questions
