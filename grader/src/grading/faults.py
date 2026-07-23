"""Fault types with special training semantics.

``AgentFault`` means the submitted artifact itself earned a clean 0.0:
missing output, malformed CSV, non-finite predictions, symlink/FIFO in place
of a regular file, and similar agent-controlled problems. Hardened runners
keep this type as score 0.0. ``GraderFault`` and ``InfrastructureFault`` are
discarded; an untyped candidate-evaluation exception is kept as zero with a
critical operator alert so it cannot become a free episode/group veto.
"""

from __future__ import annotations


class AgentFault(Exception):
    """The agent's submission is the cause of a 0.0 score."""


class GraderFault(Exception):
    """Trusted grader code, configuration, or private data is invalid."""


class InfrastructureFault(Exception):
    """The execution environment cannot perform a valid grade."""


__all__ = ["AgentFault", "GraderFault", "InfrastructureFault"]
