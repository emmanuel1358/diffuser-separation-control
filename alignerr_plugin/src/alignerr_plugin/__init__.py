"""Alignerr plugin for generic Boreal/Harbor tasks."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alignerr.plugins import PluginRegistry


def setup_plugins(registry: "PluginRegistry") -> None:
    """Register the universal task benchmark and supporting commands."""
    from alignerr_plugin.commands import (
        create_problem,
        evaluate,
        export_harbor,
        export_taiga,
        score,
    )
    from alignerr_plugin.result_handler import TaskResultHandler
    from alignerr_plugin.validators.task.creator import TaskCreator
    from alignerr_plugin.validators.task.validator import TaskValidator

    registry.register_benchmark("taiga_task", TaskValidator(), TaskCreator())
    registry.register_command("create-problem", create_problem)
    registry.register_command("evaluate", evaluate)
    registry.register_command("score", score)
    registry.register_command("export-taiga", export_taiga)
    registry.register_command("export-harbor", export_harbor)
    registry.register_result_handler(TaskResultHandler())
