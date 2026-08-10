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

- [ ] I ran `uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>` (or `lbx-rl-template check`) for local feedback.
- [ ] I did **not** commit `problems/<task_id>/.alignerr/build_proof.json` or generated calibration locks; trusted CI regenerates authoritative proof.
- [ ] I set `[difficulty].task_type` to `ml`, `mujoco`, `cfd`, `structures`, or `software_engineering`.
- [ ] For rendered tasks (MuJoCo or declared `render_outputs`), I committed reviewer artifacts under `problems/<task_id>/.alignerr/ground_truth/`.
- [ ] I verified `.env.local`, `.harness-runs/`, provider keys, credentials, and other local secrets are not committed.
- [ ] I understand trusted CI is dispatched by ISO mothership, while the
      `trusted-ci/grade` check and diagnostics are posted back on this PR.
- [ ] I did not add workflow credentials or cross-repository dispatch to this
      template or task fork.
