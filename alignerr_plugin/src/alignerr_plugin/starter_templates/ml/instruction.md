# Task

Replace this with the instructions the agent should follow.

The public training data is under `/data/`. Write `/tmp/output/predictor.py`
defining `load_predictor()` and an object with `predict(rows)`. The grader
commits the artifact and evaluates it on private challenge rows; document the
required input fields and returned target lists here.

The `load_predictor()`/first-call timeout is 120 seconds. Each subsequent
`predict(rows)` call has a 60-second deadline. The predictor may receive up to
100000 rows, and its serialized output must stay within 64 MiB.

Name any specific Python libraries or command-line tools the agent should use
directly in these instructions.
