# Complete the synthetic live cutover

Work only in `/workspace/repo`.

The state service exposes records with both a legacy value and the desired
current value. The customer service is already issuing concurrent requests
while you work. Repair `app.py` so every lookup returns the current value in
the existing response schema.

Keep the implementation deterministic and standard-library-only. Do not modify
the state, customer, init, capture, or verifier services. The runtime will
finalize load over HTTP, atomically snapshot service state, and capture your git
workspace before the separate verifier runs.

The runtime maps the captured final repository to `/tmp/output/repo`; do not
write grading results or service evidence there yourself.
