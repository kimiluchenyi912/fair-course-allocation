# Fair Course Allocation Project Plan

## 1. Project Goal

Build a fair and explainable system that allocates students to existing
high-school course sections.

The system should balance:

- student course preferences,
- graduation and academic needs,
- course capacity,
- prerequisite requirements,
- schedule conflicts,
- and fairness between students.

## 2. Version 1 Scope

Version 1 assumes that the school has already decided:

- which courses will be offered,
- how many sections each course has,
- the period of each section,
- and the capacity of each section.

The system will decide which students are assigned to which sections.

## 3. Out of Scope for Version 1

Version 1 will not:

- create the school's master schedule,
- assign teachers,
- assign classrooms,
- decide teacher workloads,
- or automatically determine how many sections the school should offer.

These may become future extensions.

## 4. Inputs

The system will use:

- student information,
- course information,
- section information,
- student course preferences,
- prerequisite eligibility,
- graduation-critical needs,
- and optional submission times for baseline comparison.

Only synthetic or anonymized data will be used.

## 5. Hard Constraints

Every valid allocation must satisfy:

1. A section cannot exceed its capacity.
2. A student cannot take two sections in the same period.
3. A student cannot take multiple sections of the same course.
4. A student must satisfy all prerequisites.
5. A student cannot be assigned to an ineligible course.
6. A student cannot receive more courses than the required course load.
7. The system must clearly report when no feasible allocation exists.

## 6. Optimization Priorities

The fair algorithm will optimize goals in this order:

1. Satisfy graduation-critical course needs.
2. Satisfy other essential academic-sequence needs.
3. Prevent individual students from receiving extremely poor schedules.
4. Maximize overall student preference satisfaction.
5. Use available course seats efficiently.

Students with equal priority should be treated using a reproducible lottery.

## 7. Baseline Algorithms

The fair algorithm will be compared with:

1. Random lottery
2. First-come, first-served
3. Grade-priority allocation

All algorithms must use the same input data and constraints.

## 8. Evaluation Metrics

The project will measure:

- first-choice fulfillment rate,
- top-three preference fulfillment rate,
- average student satisfaction,
- satisfaction of the worst-performing students,
- graduation-critical need fulfillment,
- number of students without a complete schedule,
- seat utilization,
- and algorithm runtime.

## 9. First Test Scenario

The first hand-verifiable dataset will contain approximately:

- 8 students,
- 5 courses,
- 2 or 3 periods,
- limited course capacities,
- one oversubscribed course,
- one prerequisite,
- one graduation-critical request,
- and at least one schedule conflict.

## 10. Version 1 Completion Criteria

Version 1 is complete when:

- all hard constraints have automated tests,
- the fair solver produces valid allocations,
- the three baseline algorithms are implemented,
- all algorithms can be compared on the same dataset,
- fairness and satisfaction metrics are calculated,
- infeasible cases are clearly reported,
- and the results can be explained using a small hand-verifiable example.