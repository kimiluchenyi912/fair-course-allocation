# Project Decisions

## 1. Course Load Semantics

`course_load` represents both:

- the student's target number of assigned courses, and
- the maximum number of courses the student may receive.

It is not an exact hard requirement.

An allocation remains valid when a student receives fewer courses,
but the student must be reported as having an incomplete schedule.

## 2. Historical Prerequisite IDs

`courses.csv` contains only courses that are currently available for
allocation.

Prerequisite IDs and completed-course IDs may refer to historical
courses that are not listed in `courses.csv`.

For example, `ALG2` may be required for `STATS` even when `ALG2` is not
currently being offered.

## 3. Ineligible Preferences

A preference for a course whose prerequisites are not satisfied is not
a fatal dataset error.

The validation layer should report it as a warning, and the allocation
solver must not assign that course to the student.

## 4. Sample Dataset Purpose

The current sample dataset intentionally contains insufficient capacity,
oversubscribed courses, ineligible preferences, and period conflicts.

It is a stress scenario and is not expected to provide every student
with a complete schedule.

A separate fully feasible dataset will be added later for basic solver
verification.

## 5. Solver Boundary After Counselor Approval

The core model now starts after counselors have approved student course
requests.

The solver does not check prerequisites, honors eligibility, AP eligibility,
or graduation eligibility. Those rules may appear in synthetic-generation
configuration, but they are not fixed-section allocation constraints.

## 6. New Scheduling Data Structure

The authoritative data model separates:

- students,
- approved requests,
- concrete sections,
- linked semester blocks,
- assignments,
- unmet requests,
- and run metrics.

The old `data/sample/` files are legacy toy data from the earlier
prerequisite-centered model and are not authoritative for the new simulation.

## 7. Grade Profiles And Free Periods

The first whole-school simulation uses these grade profiles:

- Grade 9: 700 students; 90% request 7 classes and 10% request 6 classes.
- Grade 10: 650 students; 80% request 7 classes and 20% request 6 classes.
- Grade 11: 640 students; 60% request 7 classes, 35% request 6 classes,
  and 5% request 5 classes.
- Grade 12: 640 students; 33% request 7 classes, 47% request 6 classes,
  and 20% request 5 classes.

Free periods may only be placed in Period 1, Period 6, or Period 7 in V1.

## 8. Capacity And Section Planning Assumptions

Normal academic courses default to 40 seats. AP Computer Science A defaults
to 25 seats. PE defaults to 50 seats, with a configurable range of 40 to 60.

All courses use the same 50% waitlist expansion rule in future section
planning. If a course's remaining waitlist reaches `ceil(0.5 * standard
section capacity)`, a later section-planning pass may consider adding another
section. The rule can apply repeatedly after each added section.

Niche courses often have one section, and sometimes two, because demand is
usually low. That is descriptive, not a hard cap. V1 should not impose
course-specific hard maximum section counts for AP CSA, AP Statistics, AP
Physics C, Calc D + Linear Algebra, or niche electives.

This is not part of the fixed-section V1 solver; it is a later planning
analysis.

## 9. Confirmed Course-Structure Rules

Grade 12 Government and Economics share one period. The semester order is
defined by the section, and regular/AP combinations may be mixed.

AP Physics C is a full-year, one-period course, with Mechanics in one
semester and Electricity and Magnetism in the other.

Grade 10 math usually starts at Integrated Math 2 or above. Grade 11 math
usually starts at Integrated Math 3 or above. Grade 12 in-school math usually
starts at Introduction to Calculus or above.

AP European History is Grade 12 only. AP Art History is available to Grades
11 and 12.

Calculus III and Linear Algebra style dual-enrollment courses are not treated
as ordinary TPHS in-school courses.

Known capacity-risk courses include Computer Programming, AP CSP, AP CSA, AP
Statistics, AP Physics C, and Calc D + Linear Algebra.

## 10. One-Year Fairness Protection Boundary

The generator only creates synthetic students and counselor-approved requests.
It does not decide which students lose access to a course.

A future section generator will use demand, capacities, and the uniform 50%
waitlist expansion threshold to decide section counts. A future solver will
assign students to fixed sections and must use ranked alternates to keep
schedules complete when a primary request is unmet.

For ordinary students, future solver policy is at most one unmet logical
primary course. A protected student, meaning a student with a prior-year
involuntary unmet primary, must have zero unmet logical primaries in the
protected year. If fixed sections and periods make that impossible, the solver
must return infeasibility diagnostics rather than silently relaxing the rule.

Logical primary counting follows course/block meaning rather than request rows:
Government/Economics linked blocks count once, and Math 2/3 Honors Accelerated
counts once even though it occupies two periods.

## 11. Synthetic Section Planner Boundary

The section planner consumes generated students and approved requests, counts
primary logical demand, chooses section counts, and assigns section periods. It
does not assign students to sections and does not decide who is unmet.

For V1, positive primary demand opens at least one section as a model
assumption. Additional sections use the uniform 50% waitlist rule with integer
threshold `ceil(0.5 * section capacity)`.

Math 2/3 Honors Accelerated sections are modeled as consecutive double-period
sections using configured adjacent period pairs. Government/Economics linked
blocks produce two semester section rows sharing one linked group and one
period. Teacher, classroom, lab, and teacher-course qualification constraints
remain unmodeled until reliable data exists.
