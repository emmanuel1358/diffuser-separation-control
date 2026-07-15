"""Result handler for the universal task benchmark."""

from alignerr.commands.results import DefaultResultHandler


class TaskResultHandler(DefaultResultHandler):
    """Display task results using the default Alignerr behavior."""

    benchmark = "taiga_task"
