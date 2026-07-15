"""Taiga ``required_resources`` enum values and helpers.

Source: live Taiga OpenAPI ``required_resources`` enum from
``https://taiga.ant.dev/api/docs/openapi.json``.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

TAIGA_RESOURCE_OPTIONS: tuple[str, ...] = (
    "2vcpu+6gib",
    "4vcpu+16gib",
    "6vcpu+32gib",
    "8vcpu+64gib",
    "16vcpu+64gib",
    "16vcpu+128gib",
    "2vcpu+6gib+perf",
    "4vcpu+16gib+perf",
    "6vcpu+32gib+perf",
    "8vcpu+64gib+perf",
    "16vcpu+64gib+perf",
    "16vcpu+128gib+perf",
    "13vcpu+32gib+tpuv5e1x1",
    "50vcpu+128gib+tpuv5e2x2",
    "16vcpu+64gib+tpuv5e2x2",
    "3vcpu+25gib+h100/8",
    "6vcpu+50gib+h100/4",
    "12vcpu+100gib+h100/2",
    "24vcpu+200gib+h100/1",
    "3vcpu+25gib+h100/8+graphics",
    "6vcpu+50gib+h100/4+graphics",
    "12vcpu+100gib+h100/2+graphics",
    "24vcpu+200gib+h100/1+graphics",
)

TaigaRequiredResources: TypeAlias = Literal[
    "2vcpu+6gib",
    "4vcpu+16gib",
    "6vcpu+32gib",
    "8vcpu+64gib",
    "16vcpu+64gib",
    "16vcpu+128gib",
    "2vcpu+6gib+perf",
    "4vcpu+16gib+perf",
    "6vcpu+32gib+perf",
    "8vcpu+64gib+perf",
    "16vcpu+64gib+perf",
    "16vcpu+128gib+perf",
    "13vcpu+32gib+tpuv5e1x1",
    "50vcpu+128gib+tpuv5e2x2",
    "16vcpu+64gib+tpuv5e2x2",
    "3vcpu+25gib+h100/8",
    "6vcpu+50gib+h100/4",
    "12vcpu+100gib+h100/2",
    "24vcpu+200gib+h100/1",
    "3vcpu+25gib+h100/8+graphics",
    "6vcpu+50gib+h100/4+graphics",
    "12vcpu+100gib+h100/2+graphics",
    "24vcpu+200gib+h100/1+graphics",
]

TAIGA_RESOURCE_SET = frozenset(TAIGA_RESOURCE_OPTIONS)
CPU_RESOURCE_OPTIONS = TAIGA_RESOURCE_OPTIONS[:6]
CPU_PERF_RESOURCE_OPTIONS = TAIGA_RESOURCE_OPTIONS[6:12]
TPU_RESOURCE_OPTIONS = TAIGA_RESOURCE_OPTIONS[12:15]
H100_RESOURCE_OPTIONS = TAIGA_RESOURCE_OPTIONS[15:19]
H100_GRAPHICS_RESOURCE_OPTIONS = TAIGA_RESOURCE_OPTIONS[19:]
RESOURCE_RANK = {
    "2vcpu+6gib": 0,
    "2vcpu+6gib+perf": 1,
    "4vcpu+16gib": 2,
    "4vcpu+16gib+perf": 3,
    "6vcpu+32gib": 4,
    "6vcpu+32gib+perf": 5,
    "8vcpu+64gib": 6,
    "8vcpu+64gib+perf": 7,
    "16vcpu+64gib": 8,
    "16vcpu+64gib+perf": 9,
    "16vcpu+128gib": 10,
    "16vcpu+128gib+perf": 11,
    "13vcpu+32gib+tpuv5e1x1": 100,
    "16vcpu+64gib+tpuv5e2x2": 101,
    "50vcpu+128gib+tpuv5e2x2": 102,
    "3vcpu+25gib+h100/8": 200,
    "6vcpu+50gib+h100/4": 201,
    "12vcpu+100gib+h100/2": 202,
    "24vcpu+200gib+h100/1": 203,
    "3vcpu+25gib+h100/8+graphics": 300,
    "6vcpu+50gib+h100/4+graphics": 301,
    "12vcpu+100gib+h100/2+graphics": 302,
    "24vcpu+200gib+h100/1+graphics": 303,
}


def normalize_required_resources(value: object) -> str:
    return str(value or "").strip().lower()


def valid_resource_options_display() -> str:
    return ", ".join(repr(resource) for resource in TAIGA_RESOURCE_OPTIONS)


def validate_required_resources(value: object) -> str:
    resource = normalize_required_resources(value)
    if resource not in TAIGA_RESOURCE_SET:
        raise ValueError(
            "[environment].required_resources must be one of "
            f"{valid_resource_options_display()} (got {value!r})"
        )
    return resource


def is_tpu_resource(required_resources: str) -> bool:
    return "tpu" in normalize_required_resources(required_resources)


def is_h100_resource(required_resources: str) -> bool:
    return "+h100/" in normalize_required_resources(required_resources)


def is_graphics_resource(required_resources: str) -> bool:
    return normalize_required_resources(required_resources).endswith("+graphics")


def is_accelerator_resource(required_resources: str) -> bool:
    return is_tpu_resource(required_resources) or is_h100_resource(required_resources)


def is_cpu_resource(required_resources: str) -> bool:
    resource = normalize_required_resources(required_resources)
    return resource in CPU_RESOURCE_OPTIONS or resource in CPU_PERF_RESOURCE_OPTIONS
