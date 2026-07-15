# Standard Rubric Guidance

This document outlines the standard approach for creating rubrics that turn
expert evaluation into objectively gradable tasks. Rubrics codify domain
knowledge into specific, measurable criteria that can be applied consistently
across evaluation, training, and testing.

Use this guidance when writing or reviewing `RubricBuilder` criteria in
`scorer/compute_score.py`.

## Summary

Rubrics help graders evaluate task outputs like experts. A good rubric tells
the grader exactly what a thoughtful professional would look for in a realistic
task. It should not include inessential requirements just to inflate difficulty.
If a task is too easy, design a harder task rather than making the rubric more
arbitrary.

Rubric items should be as simple as possible, but no simpler than necessary.
They may contain sub-parts, but the overall rubric should follow the logically
independent components of task completion.

Rubrics should focus on task-specific requirements, not broad domain-wide
generalities. General domain requirements can be handled by shared review and
grading policy; the task rubric should capture what matters for this specific
prompt and expected output.

## Design Principles

- Rubrics should evaluate like experts on realistic tasks.
- Rubric items should be clear, unambiguous, and measurable.
- Rubric items should align with the prompt. If the prompt does not require a
  criterion, remove it.
- Rubric items can have weights indicating relative importance, but do not
  over-optimize the exact weights.
- Rubrics can evaluate outputs, process, or both.
- Use process criteria only when the process is important to proper task
  performance.
- Prefer outcome criteria when a task can be completed well in multiple ways.
- Do not require one specific solution path unless that path is actually
  required by the task.
- Refer to golden/reference solutions when appropriate, but do not copy every
  golden value into the rubric.

## Golden Solutions

Golden solutions are useful for grounding evaluation, but rubrics should avoid
becoming a long list of copied answer values. Instead, describe the expected
calculation, location, tolerance, or behavior, and refer to ground truth or the
reference solution where appropriate.

For example, if a spreadsheet task has a golden solution with a particular EPS
value, a good criterion is:

> Correctly calculates the 2025 EPS using the appropriate formula and places it
> in the correct cell, matching ground truth to reasonable tolerance.

Avoid copying every number from the golden solution into the rubric unless the
exact literal value is part of the task requirement.

## Outcome vs. Process

Use outcome-focused criteria when there are multiple valid solution paths.

Toy example: a math task has two valid approaches, A and B. The golden solution
uses approach A and gets the answer `42`.

Bad rubric:

- Answer is `42`.
- Applies solution A.

Better rubric:

- Gets the right answer.
- Applies logical mathematical reasoning.

Process criteria are appropriate when the task explicitly requires a process,
such as using a required API, writing a formula instead of hard-coding values,
or producing a file in a mandated format.

## Common Pitfalls

### Overly Prescriptive Criteria

Ask: is this necessary for a high-quality answer?

If not, remove it. Do not lock the grader to one implementation path when
several paths would satisfy the task.

### Ambiguous Criteria

Ask: can this item be misinterpreted by a reasonable reviewer or model?

If so, clarify the wording, add measurable thresholds, or refer to a ground
truth artifact.

### Criteria Not Aligned With the Prompt

Ask: does the prompt actually require this?

If not, remove the item or update the prompt so the requirement is explicit.

### Rubrics That Are Mostly Formatting

Formatting criteria are useful when format is essential, such as XML required
by an API. But do not let formatting crowd out substantive evaluation of the
task answer.

### Rubrics That Copy Too Much Ground Truth

Long lists of literal golden values are hard to maintain and easy to stale.
Prefer criteria that name the correct formula, target field, tolerance, and
ground-truth comparison.

## Examples

### Finance Spreadsheet Task

Avoid long criteria that copy every golden number into the rubric.

Instead of:

```text
[weight: 10] Dividend Yield cells contain a formula yielding:
MSFT: 0.7%, INTU: 0.7%, ORCL: 1.0%, ...
```

Prefer:

```text
Dividend Yield cells for the company and peers match ground truth to reasonable
tolerance and are computed with the appropriate formula: Quarterly Dividend x 4
/ Stock Price. [10 points]
```

This keeps the rubric focused on the expert requirement: correct formula,
correct placement, and agreement with ground truth.

### Structured API Response Task

If an XML format is required, a formatting criterion is appropriate:

```text
Response follows the required XML format exactly, with no text outside XML
tags. [20 points]
```

For content fields, prefer criteria that distinguish unambiguous fields from
judgment calls:

```text
The incident summary contains all important details from the reference example
and does not include extraneous information. [10 points]
```

Avoid criteria that force one classification when the prompt leaves the answer
ambiguous. If reasonable reviewers could disagree, either clarify the prompt or
grade the response against the reference/golden example explicitly.

### Coding Task

Avoid over-prescribing the tool used to reach the outcome.

Overly prescriptive:

```text
Agent used cherry-pick to apply detached HEAD commits to main in correct order.
```

Better:

```text
Agent correctly applied the detached HEAD commits to main in the correct order,
with appropriate commit messages preserved.
```

The revised criterion evaluates the outcome while allowing valid alternative
methods.

## Checklist

Before finalizing a rubric, confirm:

- Each item is required by the prompt or task context.
- Each item is measurable by code or a clearly specified ground-truth artifact.
- Criteria are logically independent enough to provide useful diagnostic signal.
- The rubric does not reward inessential complexity.
- The rubric does not force a single solution path unless the process matters.
- Ambiguous fields are either clarified or compared against an explicit
  reference.
- Weights roughly match importance without excessive tuning.
- The rubric captures what a thoughtful domain expert would actually evaluate.

