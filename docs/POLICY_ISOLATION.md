# Policy Isolation For Executable Submissions

Some MuJoCo/control tasks ask the agent to submit executable Python such as
`/tmp/output/policy.py`. Do not import that policy directly into
`compute_score.py` when hidden cases, fault schedules, or private fixtures are
live in the grader process.

Use `PolicyWorker` instead:

```python
from grading import PolicyWorker

def compute_score(workspace, trajectory, private):
    policy_path = workspace / "policy.py"
    with PolicyWorker(policy_path, timeout_s=0.1) as policy:
        obs = {"qpos": [0.0, 0.1], "qvel": [0.0, 0.0]}
        action = policy.act(obs)
```

`PolicyWorker` runs the submitted policy in a child Python process. The grader
keeps hidden state in the parent process and sends only public observations over
a JSON line protocol. This prevents submitted code from using Python frame
inspection or module monkeypatching to read grader locals such as hidden cases,
perturbation schedules, or target labels.

`PolicyWorker` is intentionally compatible with common scorer patterns:

- `policy.act(obs)` for normal `act(obs)` / `Policy().act(obs)` submissions.
- `policy(obs)` as an alias for `policy.act(obs)`, useful when existing scorer
  code stores the loaded policy in a variable such as `policy_fn`.
- `policy.call("method_name", *args, **kwargs)` for tasks whose submitted file
  exposes task-specific methods such as `ctrl(...)`, `identify_preset(...)`, or
  `get_action(...)`.
- `policy.init_model_xml(xml_text)` for policies that need to load the submitted
  `/tmp/output/model.xml` themselves.

## When To Use It

Use `PolicyWorker` when all of these are true:

- The task asks for executable Python (`policy.py`, `controller.py`, etc.).
- The grader calls that submitted code during scoring.
- The grader has hidden data, hidden initial states, perturbation schedules, or
  private targets that should not be visible to the submitted code.

You do not need `PolicyWorker` for static artifact tasks such as:

- `model.xml` scored by MuJoCo model inspection,
- `submission.csv` scored against hidden labels,
- JSON/text answers parsed by the grader.

The checked-in examples under `examples/` are static-output examples, so they do
not import submitted Python during scoring.

## Recommended Pattern

Keep the simulation and scoring logic in `compute_score.py`; only move the
submitted policy call behind `PolicyWorker`.

```python
from grading import PolicyWorker, RubricBuilder

def rollout_case(model, case, policy):
    obs = build_public_observation(model, case)
    action = policy.act(obs)
    apply_action(model, action)
    return metrics

def compute_score(workspace, trajectory, private):
    cases = load_hidden_cases(private)
    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)

    with PolicyWorker(workspace / "policy.py", timeout_s=0.1) as policy:
        metrics = [rollout_case(model, case, policy) for case in cases]

    @rb.criterion(id="success_rate", weight=1.0)
    def _():
        return success_rate(metrics)

    return rb.grade().to_dict()
```

Always close long-lived workers. The `with` block does this automatically. If a
scorer creates a worker outside a context manager, close it in `finally`:

```python
policy = PolicyWorker(workspace / "policy.py", timeout_s=0.25)
try:
    metrics = rollout(policy)
finally:
    policy.close()
```

## Common Patterns

### Drop-In Replacement For `policy_fn(obs)`

If existing scorer code already treats the policy as a callable, `PolicyWorker`
can be used directly:

```python
policy_fn = PolicyWorker(workspace / "policy.py", timeout_s=0.25)
ctrl = policy_fn(obs)
```

This was the common migration path for per-step controller tasks.

### Named Methods

Some tasks ask for richer APIs than `act(obs)`. Use `call`:

```python
with PolicyWorker(workspace / "sysid.py", timeout_s=15.0) as sysid:
    ctrl = sysid.call("ctrl", "model.xml")
    params = sysid.call("identify_preset", "model.xml", ctrl, obs)
```

This keeps hidden cases and generated observations in the grader process while
still supporting submitted multi-function APIs.

### Policies That Need The Submitted Model XML

Some policies load `/tmp/output/model.xml` to compute sensor addresses or model
constants. Because the worker process is isolated, pass the submitted XML to the
worker first:

```python
with PolicyWorker(workspace / "policy.py", timeout_s=0.25) as policy:
    policy.init_model_xml((workspace / "model.xml").read_text())
    action = policy.call("get_action", obs, model=None)
```

Inside the worker, `/tmp/output/model.xml` is made available with that XML text.
This supports existing policies that fall back to loading the model from
`/tmp/output/model.xml`.

### Fresh Policy State Per Case

If the old scorer re-imported the policy for each hidden case to reset internal
controller state, preserve that behavior by creating one worker per case:

```python
for case in hidden_cases:
    with PolicyWorker(workspace / "controller.py", timeout_s=0.25) as controller:
        metrics.append(rollout_case(case, controller))
```

If state should persist across cases, create a single worker outside the loop.
Be deliberate: the worker preserves Python module globals for as long as it is
alive.

### Numpy Observations And Actions

`PolicyWorker` preserves NumPy arrays using the shared JSON protocol. You can
send observations containing arrays and receive array-shaped actions:

```python
obs = {"qpos": data.qpos.copy(), "qvel": data.qvel.copy()}
action = np.asarray(policy.act(obs), dtype=float)
```

The protocol also redirects policy `print(...)` output to stderr so stdout stays
reserved for machine-readable messages.

## Migration Checklist

When migrating an existing scorer:

- Replace direct `importlib` loading of submitted code with `PolicyWorker`.
- Keep the public observation exactly the same unless the task itself is being
  redesigned.
- Use `policy(obs)` or `policy.act(obs)` for existing `act(obs)` APIs.
- Use `policy.call(...)` for custom API methods.
- Use `policy.init_model_xml(...)` if submitted policy code may read
  `/tmp/output/model.xml`.
- Preserve old policy lifetime semantics: one worker per case if the old code
  re-imported per case; one worker per grading run if state was meant to
  persist.
- Run `ground-truth` after migration and verify the reference score is preserved
  before committing.

## Limits

`PolicyWorker` isolates Python frames and module state between the grader and
policy, but it is not a complete OS sandbox. The worker still runs in the task
container. Do not send hidden data in observations, and do not make hidden
fixtures available through public paths.

For stronger future isolation, the same observation/action protocol can be moved
behind a stricter subprocess, user, namespace, or container sandbox without
changing task author code.

## Contributing Improvements

If `PolicyWorker`, the harness, docs, examples, templates, or any other shared
code does not cover a task pattern you need, open a PR to improve the codebase.
Contributors are welcome to update shared tooling, docs, tests, starter
templates, examples, or validators when that makes task authoring safer or more
reliable.
