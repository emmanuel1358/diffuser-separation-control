"""Repository slug normalization."""


def normalize_slug(value: str) -> str:
    """Return a lowercase slug.

    The seeded implementation is incomplete. See the repository README and
    public tests for the supported interface.
    """

    return value.strip().lower().replace(" ", "-")
