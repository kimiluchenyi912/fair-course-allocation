# Whole-School Scheduling Simulation Specification v1

## 1. Purpose

Build a synthetic, Torrey Pines–inspired high-school scheduling environment that can test whether different allocation algorithms produce complete, fair, and stable student schedules.

The simulation is not presented as an exact copy of TPHS. It combines:

- public course information;
- student-informed estimates;
- explicit model assumptions;
- randomized yearly demand.

Anonymous school data should be able to replace the synthetic inputs without changing the solver.

## 2. Solver boundary

The solver starts **after counselors approve course requests**.

The solver does not:

- check prerequisites;
- decide whether a student belongs in honors or AP;
- recommend courses;
- evaluate graduation eligibility before requests are submitted.

The solver does:

- assign approved requests to concrete sections and periods;
- respect section capacity and schedule conflicts;
- preserve primary requests when possible;
- use ranked alternates when necessary;
- minimize incomplete schedules;
- distribute unavoidable losses fairly.

## 3. School population

| Grade | Approximate students |
|---|---:|
| 9 | 700 |
| 10 | 650 |
| 11 | 640 |
| 12 | 640 |
| **Total** | **2,630** |

These are simulation parameters, not claims of exact current enrollment.

## 4. Target number of scheduled classes

| Grade | 7 classes | 6 classes | 5 classes |
|---|---:|---:|---:|
| 9 | 90% | 10% | 0% |
| 10 | 80% | 20% | 0% |
| 11 | 60% | 35% | 5% |
| 12 | 33% | 47% | 20% |

`target_course_count` is both the student's desired load and the maximum number of normal-period course slots to fill. An underfilled schedule remains feasible but must be reported.

## 5. Period structure and unscheduled time

The school day has seven periods: P1–P7.

Unscheduled periods are restricted to the edges of the day:

- morning: P1;
- afternoon: P6 and/or P7.

The solver must not create an interior free period in P2–P5 merely to make optimization easier.

Exact allowed patterns should remain configurable. V1 uses the hard rule that all free periods must be a subset of `{P1, P6, P7}`.

## 6. Grade-level request-generation rules

These rules guide synthetic request generation only. They are not eligibility checks inside the solver.

### Grade 9

Typical structure:

- English 9 or English 9 Honors;
- one mathematics course;
- Biology;
- PE or an approved athletic/PE arrangement;
- usually one world language;
- remaining slots are electives.

About 90% request seven classes. A known capacity-risk elective is Computer Programming.

Special ELD, foundational, functional, readiness, and other support pathways are excluded from V1 unless later added as a separate scenario.

### Grade 10

Core distributions:

- English 10 vs English 10 Honors: approximately 50/50;
- World History vs AP World History: approximately 50/50;
- Chemistry vs Honors Chemistry: approximately 1/3 vs 2/3;
- mathematics normally begins at Integrated Math 2 or above.

Math details:

- Integrated Math 2/3 Honors Accelerated requires two periods;
- AP Calculus AB: approximately 4–8 Grade 10 students, base value 6;
- AP Calculus BC: approximately 20 Grade 10 students;
- Introduction to Calculus is not normally generated for Grade 10.

Known capacity risks:

- AP CSP, especially among students without prior Computer Programming;
- AP CSA for any Grade 10 applicant.

### Grade 11

Typical structure:

- English 11 or AP English Language;
- U.S. History or AP U.S. History;
- mathematics normally begins at Integrated Math 3 or above;
- science, language, PE, and electives vary by student;
- AP Seminar is available;
- AP Art History is available to Grades 11–12.

Removed or corrected rules:

- AP Humanities is no longer offered;
- AP European History is not generated for Grade 11;
- world languages have no honors versions.

Special course structures:

- AP Physics C occupies one period for the full year, with Mechanics in one semester and E&M in the other;
- Calculus D + Linear Algebra is a dual-enrollment, school-scheduled block that occupies one period and is capacity constrained.

Known capacity risks:

- Calculus D + Linear Algebra: normal capacity about 40, sometimes overloaded to 45, with roughly five students still unable to enroll in the observed year;
- AP Statistics: historically about three sections of 40, with roughly five students unable to enroll in the observed year;
- AP Physics C: observed capacity pressure, exact waitlist unknown.

### Grade 12

Typical required structure:

- English 12 or AP English Literature;
- one shared-period Government/Economics block;
- other mathematics, science, language, PE, and electives are optional.

Government/Economics rules:

- Government and Economics occupy one period across the year;
- semester order varies by section;
- regular and AP levels may be mixed:
  - Government + Economics;
  - Government + AP Macroeconomics;
  - AP Government + Economics;
  - AP Government + AP Macroeconomics.

Math begins at Introduction to Calculus or above in the normal Grade 12 generator. Math 3 and Math 3 Honors are not generated for ordinary Grade 12 students.

Grade-specific electives:

- AP Research is available to Grade 12;
- AP European History is Grade 12 only;
- AP Art History is available to Grades 11–12.

## 7. Special scheduling representations

### Double-period course

Example: Integrated Math 2/3 Honors Accelerated.

The student receives one course request, but the assigned section consumes two periods. Whether the periods must be consecutive remains configurable until verified.

### Sequential semester block sharing one period

Examples:

- Government/Economics;
- AP Physics C content sequence;
- Calculus D/Linear Algebra dual-enrollment sequence.

The block occupies one period for the full year while the course content changes by semester.

### Zero-period or non-school-period credit

Some athletic credit or external dual-enrollment arrangements may satisfy a requirement without consuming P1–P7. These must be represented explicitly with `occupies_school_period = false` rather than treated as ordinary classes.

## 8. Capacity defaults

| Category | Default capacity |
|---|---:|
| Normal academic section | 40 |
| Most AP sections | 40 |
| AP Computer Science A | 25 |
| PE section | 50 base, configurable 40–60 |
| Calculus D + Linear Algebra | 40 normal, overload up to 45 |

These are default simulation values. Individual courses may override them.

## 9. Section-generation policy

### Protected core courses

Required core courses should open enough sections to serve essentially all approved requests. Their section count is driven by demand and capacity, not by a waitlist lottery.

### Elective and advanced courses

For a course with capacity `C`, after currently planned sections are filled, an additional section may open when the remaining waitlist reaches approximately `0.5 × C`, subject to teacher, room, and course-specific limits.

Illustration for a 40-seat course:

- 5–19 remaining students: no new section;
- 20 or more remaining students: consider one additional section.

This is a planning rule, not an unlimited guarantee. Every course has a configurable `max_sections` or expansion policy.

### Niche electives

Low-demand niche electives should normally have one section and a practical maximum of two in synthetic data. The generator must not create implausible outcomes such as six sections of a small course merely because of uncontrolled randomness.

## 10. Known bottleneck courses

The initial calibration set includes:

- Grade 9 Computer Programming;
- Grade 10 AP CSP;
- AP CSA;
- Calculus D + Linear Algebra;
- AP Statistics;
- AP Physics C.

These courses should be represented as realistic stress points rather than guaranteed failures every year.

## 11. Modeling uncertainty and yearly change

Unknown elective demand is not filled with fake precision. Each course receives:

- a demand tier: core, mainstream, popular, niche, or fixed-limited;
- a base participation rate or expected demand;
- a yearly random multiplier;
- a trend multiplier for increasingly competitive course selection;
- a minimum and maximum section policy.

Recommended test scenarios:

1. **Stable year** — demand close to baseline.
2. **Competitive-growth year** — increased AP, CS, advanced math, and advanced science demand.
3. **Stress year** — several popular courses rise simultaneously.
4. **Randomized robustness runs** — repeat many seeds and compare outcomes.

Randomness must be reproducible through a recorded seed.

## 12. Solver priorities

Use lexicographic optimization or strongly separated weights:

1. maximize the number of students receiving a complete target schedule;
2. maximize fulfilled primary requests;
3. minimize use of alternates;
4. protect the worst-off students and distribute losses fairly;
5. minimize total unmet requests;
6. balance section utilization only after the higher priorities.

The solver must never silently violate hard constraints.

## 13. Required evaluation metrics

Report at minimum:

- complete-schedule rate overall and by grade;
- primary-request fulfillment rate;
- alternate-use rate;
- number of students missing 1, 2, or more requested courses;
- maximum and average student loss;
- waitlist size by course;
- section utilization;
- fairness by grade and request-load group;
- assignment churn across nearby random-demand scenarios;
- variability across random seeds.

Compare against:

- random lottery;
- first-come-first-served;
- grade-priority allocation;
- the fairness-optimized solver.

## 14. V1 implementation order

1. Replace the old prerequisite-centered schema with the new scheduling schema.
2. Validate all configuration files.
3. Generate synthetic students and approved primary/alternate requests.
4. Implement simple baselines.
5. Implement the fixed-section CP-SAT allocator.
6. Add fairness and robustness metrics.
7. Add section planning and 50% waitlist expansion analysis.
9. Build a lightweight website only after the solver is tested.
