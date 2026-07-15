"""Re-export of the privilege-dropped ``load_submitted_policy`` (from
``grading.policy_runner``) at the ``env_server.policy_loader`` import path used
by env / hybrid / sim_policy graders."""

from __future__ import annotations

from grading.policy_runner import (
    PolicyHandle,
    PolicyWorker,
    PolicyWorkerError,
    load_submitted_policy,
)

__all__ = [
    "load_submitted_policy",
    "PolicyHandle",
    "PolicyWorker",
    "PolicyWorkerError",
]
