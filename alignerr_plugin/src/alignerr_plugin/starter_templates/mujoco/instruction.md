# MuJoCo Robotics Task

Create a policy for the provided MuJoCo simulation.

Your solution must write a policy implementation to:

```text
/tmp/output/policy.py
```

The policy should expose either:

```python
def act(obs):
    ...
```

or:

```python
class Policy:
    def act(self, obs):
        ...
```

The grader will roll out the policy in hidden MuJoCo episodes and score the mean return.
