"""Delivery-platform helpers shared by schema validation and mothership CI."""

from __future__ import annotations

from typing import Any

ALLOWED_DELIVERY_PLATFORMS: frozenset[str] = frozenset({"taiga", "prometheus"})
DEFAULT_DELIVERY_PLATFORM = "taiga"


def normalize_delivery_platform(value: object) -> str:
    """Normalize ``[delivery].platform`` to a canonical lowercase enum value."""
    platform = str(value or DEFAULT_DELIVERY_PLATFORM).strip().lower()
    if platform not in ALLOWED_DELIVERY_PLATFORMS:
        raise ValueError(
            f"[delivery].platform must be one of {sorted(ALLOWED_DELIVERY_PLATFORMS)} "
            f"(got {value!r})"
        )
    return platform


def delivery_platform_from_toml(data: dict[str, Any] | None) -> str:
    """Return a normalized delivery platform with a safe Taiga default."""
    if not isinstance(data, dict):
        return DEFAULT_DELIVERY_PLATFORM

    delivery = data.get("delivery")
    if not isinstance(delivery, dict):
        return DEFAULT_DELIVERY_PLATFORM

    try:
        return normalize_delivery_platform(delivery.get("platform"))
    except ValueError:
        return DEFAULT_DELIVERY_PLATFORM
