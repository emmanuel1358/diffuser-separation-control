# CFD & Structures: two ways to clear review

This is a short, labeler-facing update for **non-eval CFD and structures**
tasks. It does not replace your full authoring guide. It only changes when you
may **submit for review** and how you tell the reviewer which path you cleared.

## What changed (this is an easing)

You now have **two independent lanes** to clear the submit-for-review /
acceptance bar. Clear **either** lane and you may submit. Clear both and say
so.

| Lane | What its **score** bar is based on | When to use it |
| --- | --- | --- |
| **Prometheus** | Prometheus rollout mean + stddev | Your Prometheus numbers look good |
| **Achilles** | Boreal mean — **independent of Prometheus** | Prometheus is weak or noisy, but Boreal difficulty is strong |

The lanes differ only in **which score** clears the difficulty bar. CI and
Boreal QA apply to both.

You no longer need one combined path where every signal must line up. Pick the
lane that your evidence supports.

Eval Prometheus projects are unchanged by this note (Boreal remains non-blocking
for eval). Use your eval guide for those rows.

---

## Boreal QA is required on **both** lanes

Clearing a score lane is **not** permission to submit. Regardless of lane:

1. **Wait for Boreal QA to run** on your current PR head. If the required QAs
   (`transcript_health`, `data_quality`, `env_linter`, `reward_hacking`) have
   not posted results yet, you are **not** eligible to submit — even with a
   perfect Prometheus mean and stddev.
2. **Clear every critical finding**, or document in your PR comment why a
   remaining critical is a false positive (see the Data Quality example below).
3. Warnings and info findings do **not** block.

A green `trusted-ci/grade` and a passing Prometheus score gate say nothing about
Boreal QA. The Prometheus Eval comment reports only the Prometheus **score**
component; Boreal QA lands separately on the **LBx Validation** comment and the
dashboard.

---

## Lane 1 — Prometheus

Submit for review on this lane only when **all** of the following are true:

1. **CI passes** (`trusted-ci/grade` green on the current PR head).
2. **Prometheus mean ≤ 0.6** and **Prometheus stddev ≥ 0.08**.
3. **Boreal QA has run** on this head (required QAs posted on the LBx
   Validation comment or the dashboard).
4. **No Boreal critical findings**, **or** you believe remaining criticals are
   false positives and you leave a PR comment explaining why (see template
   below).

Warnings and info findings do not block. Trainability is **not** a labeler
submit gate (reviewers may still inspect trainability artifacts when marking
Done).

---

## Lane 2 — Achilles (independent of Prometheus)

Submit for review on this lane only when **all** of the following are true:

1. **CI passes**.
2. **Boreal mean score ≤ 0.4**.
3. **Boreal QA has run** on this head (same rule as Lane 1).
4. **No Boreal critical findings**, **or** false-positive criticals documented
   in a PR comment (same rule as Lane 1).

Prometheus mean, stddev, and trainability are **not** required for this lane.
If Achilles is clean, you may submit even when Prometheus looks weak.

---

## Both lanes

If both Lane 1 and Lane 2 pass on the same head, submit as **both**. That is
the strongest case for acceptance.

---

## Common Boreal false positive (Data Quality)

Data Quality QA sometimes treats a **required output filename** mentioned in the
prompt as if it were a required **input** that must already be in the payload.

Example of a clear false positive:

```text
1 prompt-referenced file(s) not in payload

Prompt references ['frame_design.json']; payload has 6 files.
```

If `frame_design.json` (or similar) is the artifact the agent must **write**
under `/tmp/output/...` (or another disclosed output path), and it is not meant
to ship as public input under `data/`, this finding is a false positive.

**What to do:** do not rewrite a correct task to silence it. Submit with
`REVIEW_LANE` filled in and explain the FP in the comment template below
(point at the instruction line that defines the output path).

Fix real criticals yourself when the finding is genuine (missing public input,
broken env, grader mismatch, and so on).

---

## PR comment template (required when you submit)

Copy this onto the PR when you request review. Fill every field. Use exactly
one of: `prometheus`, `achilles`, or `both`.

````markdown
## Submit for review

```
REVIEW_LANE: prometheus | achilles | both
PR_HEAD_SHA: <full sha>
ASK: acceptance | coaching
```

| Field | Value |
| --- | --- |
| CI (`trusted-ci/grade`) | Pass |
| Prometheus mean | <x.xx or N/A> (Prometheus lane gate ≤ 0.6) |
| Prometheus stddev | <x.xx or N/A> (Prometheus lane gate ≥ 0.08) |
| Boreal mean | <x.xx or N/A> (Achilles lane gate ≤ 0.4) |
| Boreal QA completed | Yes — <which required QAs posted, on which surface> |
| Boreal criticals | None / FP documented below |
| Latest Boreal surface | PR comment / dashboard / both — <link or note> |

### Boreal critical false positives (if any)

- Finding: <title or short quote>
- Why FP: <e.g. Data Quality treated required output `frame_design.json` as a missing input; instruction asks the agent to write it to `/tmp/output/...`>
- Evidence: <instruction line / path>

### Notes for reviewer

<optional: anything else short>
````

---

## Quick do / don't

**Do**

- Submit when **either** score lane is met **and** Boreal QA has cleared.
- Prefer self-fixing clear, real Boreal criticals before asking for review.
- Use the comment template so the reviewer knows which lane to apply.

**Don't**

- Submit with red CI.
- Submit before Boreal QA has posted results for the current head — a passing
  Prometheus score gate is not a substitute.
- Submit on Prometheus lane when mean > 0.6 or stddev < 0.08.
- Submit on Achilles lane when Boreal mean > 0.4.
- Submit with undocumented Boreal criticals.
- Chase warning/info-only findings as a condition of submit.

---

## Where the full guides live

- CFD (non-eval Prometheus): `project_guidelines/cfd/prometheus_cfd_environments.md`
- Structures (non-eval Prometheus): `project_guidelines/strctural_engineering/PROMETHEUS_STRUCTURAL_ENGINEER_OPENSEES_AUTHORING.md`
- Starter READMEs under `alignerr_plugin/.../prometheus-cfd` and `prometheus-structures`
- Eval Prometheus guides keep a single Prometheus mean gate; dual-lane does not apply there.