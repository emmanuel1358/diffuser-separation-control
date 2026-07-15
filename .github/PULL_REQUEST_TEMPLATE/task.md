## Task Summary

**Task:** `problems/<task_id>`

**Goal:**  
Describe what the agent must accomplish.

**Expected behavior:**  
Describe the successful agent strategy at a high level.

**Required outputs:**  
List files or artifacts the agent must produce.

## Task Instructions

<!-- lbx-task-instructions:start -->
Paste the current contents of `problems/<task_id>/instruction.md` here. The
template workflow will keep this section synchronized for task PRs.
<!-- lbx-task-instructions:end -->

## Data & Assets

**Public inputs:**  
Describe files in `data/`.

**Hidden grading data:**  
Describe files in `scorer/data/` without leaking answers.

**Generated/reviewer artifacts:**  
Mention renderings, plots, videos, logs, or other `.alignerr/ground_truth/`
artifacts, if applicable.

## Grading Strategy

**Score shape:** `score_dict`, `rubric_grade`, or other return shape.

**Scoring logic:**  
Explain how `scorer/compute_score.py` evaluates submissions.

**Key criteria / metrics:**  
List the rubric criteria, continuous metrics, tolerances, or formulas.

**Determinism:**  
Explain why repeated runs are stable.

**Security notes:**  
Confirm private data stays inaccessible to the agent and submitted code is
executed through the approved sandbox helpers.

## Local Validation

- [ ] I ran `uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>`.
- [ ] I committed `problems/<task_id>/.alignerr/build_proof.json`.
- [ ] Ground truth scores `1.000`.
- [ ] Trivial/no-op score is within `[ground_truth].max_trivial_score`.
- [ ] I set `[difficulty].task_type` to the correct supported task type.
- [ ] Reviewer artifacts are committed, if applicable.
- [ ] No local secrets, provider keys, `.env.local`, `.harness-runs/`, or generated credentials are committed.
- [ ] I understand trusted CI runs in ISO mothership, but posts
      `trusted-ci/grade` and QA comments back on this template PR.
- [ ] For MuJoCo tasks, I know `run_qa`, `run_adversarial`, or
      `run_mujoco_adversarial` triggers the non-blocking adversarial review
      comment on this PR.

## Reviewer Notes

Mention intended difficulty, known limitations, or anything reviewers should
inspect closely.
