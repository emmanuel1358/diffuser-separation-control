## Task Description

Describe the task goal, expected agent behavior, and required output files.

## Task Instructions

<!-- lbx-task-instructions:start -->
Paste the current contents of `problems/<task_id>/instruction.md` here. The
template workflow will keep this section synchronized for task PRs.
<!-- lbx-task-instructions:end -->

## Dataset / Assets

Describe any public data in `data/`, hidden grading fixtures in `scorer/data/`,
simulator assets, or generated inputs. If the task has no dataset, say so.

## Grading / Compute Score Strategy

Explain how `scorer/compute_score.py` evaluates submissions. Include the
headline score shape, rubric criteria or continuous metric formula, hidden
fixtures used, and why the scoring is deterministic.

## Local Validation

- [ ] I ran `uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>`.
- [ ] I committed the generated `problems/<task_id>/.alignerr/build_proof.json`.
- [ ] I set `[difficulty].task_type` to `mujoco` or `ml`.
- [ ] For rendered tasks, I committed generated reviewer artifacts under `problems/<task_id>/.alignerr/ground_truth/`.
- [ ] I verified `.env.local`, `.harness-runs/`, provider keys, credentials, and other local secrets are not committed.
- [ ] I understand trusted CI runs from ISO mothership, but the `trusted-ci/grade`
      check and diagnostics are posted back on this template PR.
- [ ] For MuJoCo tasks, I know `run_qa`, `run_adversarial`, or
      `run_mujoco_adversarial` triggers the non-blocking adversarial review
      comment on this PR.
