# Reward-Hacking Mitigations

The grader runs as **root** inside the task image and can read the held-out
truth under `/mcp_server`. Untrusted agent code and artifacts must never reach
the grader's process, its stdout, or its trusted file reads. This doc catalogs
the mitigations the template enforces and what an author must do to stay inside
them.

Two failure classes drive everything:

- **Score forgery / truth exfiltration** -- the agent reaches a perfect score
  with no real work (reads the answer key, forges a score, runs code as root).
- **Training-data corruption** -- a `0.0` is mis-sorted into "agent earned it"
  (keep) vs "infra/author broke it" (discard), poisoning RL training.

## 1. The `agent_fault` validator stage (blocking)

`lbx-rl-template validate` statically scans every `scorer/*.py` and **fails the
task** on three patterns:

| Pattern | Why it is a hack | Fix |
| --- | --- | --- |
| A broad `except` that returns a score (`except Exception: return 0.0` / `return _failure(...)`) | Over-keep: silently converts author/infra bugs into a kept `0.0` and teaches the agent that crashing the grader yields a defined score | Catch the specific agent-controlled exception and `raise grading.faults.AgentFault(...)`; let author/infra faults propagate |
| Pickle deserialization of an agent artifact (`pickle.load`, `joblib.load`, `torch.load`, `read_pickle`, `numpy.load(allow_pickle=True)`) | RCE-as-root via `__reduce__` on a submission the agent controls | Load through the sandboxed `helpers.run_model_module(...)` / `run_policy(...)`, or ship a non-pickle format |
| Read of an agent-writable path via a raw reader (`open`/`read_csv`/`np.load`/`h5py.File`, or the `Path` methods `.read_text()`/`.read_bytes()`, on `workspace` or `/tmp/output`) | **Symlink exfiltration**: an agent script (e.g. `predict.py`, which the grader runs) re-plants a symlink to the held-out truth AFTER the runner's pre-grade scrub, so a raw read as root follows it and scores the truth as the agent's own submission. A bare `except OSError` does **not** close this — a symlink read succeeds with no error. Also over-discard: a planted directory/FIFO raises `OSError` out of `compute_score`, discarding the earned `0.0` | Use the sanctioned loader for the type (below), which `lstat`s + rejects non-regular files at load time; for a hand-rolled read, call `helpers.require_regular_file(path)` (or `os.lstat` + `stat.S_ISREG`) BEFORE the read |

The lint is scoped to the live path (reachable from `compute_score`) and spares
intermediate sentinels (e.g. a helper returning a zeroed action on bad input).

## 2. AgentFault keep-vs-discard discipline

This is the training-integrity contract:

- **Agent-controlled failure** (missing / malformed / oversized / wrong-shape
  submission, policy load/act failure) -> `raise grading.faults.AgentFault(...)`.
  Recorded as a real `0.0`, **kept** for RL training (`env_internal_failure`
  stays `False`).
- **Author / infra failure** (missing hidden truth, broken import, simulator
  crash, OOM) -> let it **propagate**. Marked `env_internal_failure` and the
  rollout is **discarded** rather than trained on.

A generic `except Exception: return 0.0` collapses both into a kept zero and is
rejected on sight. Move author/infra reads (hidden truth) outside the agent
`try`; wrap only agent code and `raise AgentFault`.

## 3. Submission loaders (per-type attack closures)

Use the sanctioned loader for each submission type instead of reading the
artifact by hand. Each runs the read/execution as the unprivileged `agent`
account and raises `AgentFault` for malformed input.

| Submission | Loader | Closes |
| --- | --- | --- |
| CSV / dataframe | `helpers.load_submission_or_fault` | symlink/FIFO, oversize, schema, non-finite |
| NumPy `.npz` / `.npy` | `helpers.load_submission_npz_or_fault` | symlink-to-truth (`O_NOFOLLOW`), FIFO-hang (`O_NONBLOCK`), oversize, and pickle-RCE (`allow_pickle=False`) |
| Any format read by hand | `helpers.require_regular_file(path)` before the read | symlink/FIFO/dir/device + oversize (an `os.lstat` + `stat.S_ISREG` guard as a callable; a bare `except OSError` does not stop a symlink) |
| Python policy / model module | `helpers.run_policy` / `helpers.run_model_module` | RCE-as-root: runs in a non-root subprocess; **score comes from the return value, never stdout** |
| Executable | `helpers.run_submitted_executable` | uid-drop + caps stdout and discards stderr so it cannot exhaust grader memory + converts an agent timeout / stdout-flood into an `AgentFault` (kept 0.0, not a discarded rollout). In `streaming=True` it also strips any `RUBRIC_SCORE=` line so the binary cannot forge a score on stdout; **capture mode returns raw stdout for you to parse — never relay it to the score parser** |
| HDF5 | `helpers.load_submission_h5_or_fault` | rejects external links, soft links, virtual datasets, and cross-file datasets that map `/mcp_server` truth into a "dataset" |
| K-fold CV of a model module | `score_kfold_cv` | cross-fold label leakage via memory (fresh worker per fold), IPC (`CLONE_NEWIPC`), and disk (per-fold pristine-snapshot rebuild) |
| Submitted policy (hidden-env tasks) | `grading.load_submitted_policy` | uid-drop sandbox; result over a dedicated pipe (never stdout); `AgentFault` on missing / non-regular / oversized / load-crash |

Golden rule: **never trust the agent's stdout or self-reported score.** Compute
the score yourself from the declared output artifacts.

## 3b. Hidden-environment (env server) closures

For `[environment].hidden_env = "env"|"hybrid"` tasks (the agent interacts with a
black-box env over `/tmp/env.sock`; see `docs/HIDDEN_ENV.md`):

- **Source stays hidden**: the env (`scorer/data/env.py`) is baked root-only
  (0700) to `/mcp_server/data/`; the blocking `hidden_env` validator stage fails
  the task if the env module is also shipped under the public `data/`.
- **Private methods are unreachable**: the server rejects any `_`-prefixed method
  and anything outside an env's `_env_public_methods` allow-list (which fails
  CLOSED if malformed), so an oracle / held-out-target / budget method cannot be
  called over the socket.
- **No path redirection via env_kwargs**: agent-supplied `__create__` kwargs whose
  string values resolve under `/mcp_server` are rejected; optional
  `allowed_env_kwargs` (in `env_config.json`) pins the create contract.
- **No grade-time env access**: `stop_env_server()` closes the socket before the
  grader runs, defeating free `reset()`-seed fingerprinting, twin-env oracles,
  and private-method probing; the grader loads the held-out env in-process via
  `grading.load_env_module` (trusted-roots restricted to `/mcp_server`).
- **No traceback leakage**: server error replies to the agent never include the
  server-side traceback (logged root-side only).

## 4. Runtime hardening (automatic)

The grading runtime (`grader/src/grading/runtime_hardening.py`, applied before
every grade) closes the classic exploits with no author action:

- runs the scorer in a fresh PID namespace with `/proc` remounted (defeats the
  PID-hopper that leaks the answer key via `/proc/<pid>/root|fd`);
- quiesces uid >= 1000 processes before grading (freezes the filesystem so a
  `while true; ln -sf <truth> submission.csv` cannot re-plant in the read
  window);
- scrubs escaping symlinks and non-regular files (FIFOs/sockets/devices) under
  the output dir (symlink-to-truth and FIFO-hang-to-discard defenses);
- traps non-finite scores (`NaN`/`inf` cannot clamp up to `1.0`);
- enforces the uid-1000 / `0700`-private filesystem model and an
  agent-unwritable grader cache.

> **The scrub + quiesce protect the PRE-grade filesystem only.** A grader that
> itself runs agent code *during* `compute_score` — e.g. executes
> `/tmp/output/predict.py` via `run_submitted_executable`, then loads
> `/tmp/output/predictions.npz` — reopens the re-plant window: that uid-1000
> process can re-create a symlink to the held-out truth AFTER the scrub. Only a
> **load-time** guard closes it. Read the artifact with the sanctioned loader for
> its type (which `lstat`s at the moment of load), or call
> `helpers.require_regular_file(path)` first — never a raw `np.load` / `open`
> behind a bare `try/except`.

See [`POLICY_ISOLATION.md`](POLICY_ISOLATION.md) for the sandbox details.

## 5. Calibration gates (continuous tasks)

- **Trivial-submission gate**: a no-op submission (and, when extractable, the
  prompt's own example) must not out-score `[ground_truth].max_trivial_score`
  (default `0.5`). Stops "submit nothing" or "copy the example" beating real
  work.
- **Zero-anchor gate**: `continuous_scoring_function` tasks must additionally
  map "no attempt" to **0**: an empty submission may not exceed
  `[ground_truth].zero_anchor_epsilon` (default `0.01`; the tolerance only
  absorbs float/curve noise). Host-probed tasks are checked by the validator's
  no-op probe; in-container / ML_Envs tasks are checked by the ground-truth
  harness run, which grades an empty workspace in the task image and records
  the result as `ground_truth_result.trivial_baseline_score` in the build
  proof (required — the validator fails a continuous in-container task whose
  proof is missing it).
- **Learnability gate**: committed `baselines/<name>/` submissions must score
  clearly below the reference target. A naive baseline that reaches the
  reference means the task is not learnable (a constant guess matches the
  expert) or `FLOOR` is miscalibrated. See [`GRADING.md`](GRADING.md) and
  `grading.calibration`.

## 6. Physics / world-integrity (MuJoCo)

`helpers.world_integrity(model, ...)` flags MJCF reward-hacks: gravity disabled,
gravity-compensation abuse, welds/equality constraints that cheat the task,
contacts turned off. Gate the score on it for control tasks.

## Author checklist

- [ ] No bare/broad `except` that returns a score; agent faults `raise AgentFault`.
- [ ] Author/infra reads (hidden truth) are outside the agent `try`.
- [ ] Submissions are read via the sanctioned loader for their type.
- [ ] The score is computed from artifacts, never the agent's stdout/claimed score.
- [ ] No-op and naive baselines score `~0`; the reference scores its target.
- [ ] `lbx-rl-template validate` passes (the `agent_fault` stage is green).
