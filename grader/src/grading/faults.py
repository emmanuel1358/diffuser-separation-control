"""Fault types with special training semantics.

``AgentFault`` means the submitted artifact itself earned a clean 0.0:
missing output, malformed CSV, non-finite predictions, symlink/FIFO in place
of a regular file, and similar agent-controlled problems. Hardened runners
catch only this type and keep the rollout as score 0.0. Any other exception is
treated as a grader/infra failure and marked ``env_internal_failure``.
"""

from __future__ import annotations


class AgentFault(Exception):
    """The agent's submission is the cause of a 0.0 score."""
