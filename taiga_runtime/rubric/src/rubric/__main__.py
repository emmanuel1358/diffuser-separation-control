"""CLI entrypoint for the generic MCP runtime."""

import sys

import typer

app = typer.Typer()


@app.command()
def mcp() -> None:
    """Run the MCP server over stdio."""
    from rubric.server import main

    main()


def main() -> None:
    """Dispatch CLI commands while preserving the `rubric mcp` contract."""
    if len(sys.argv) >= 2 and sys.argv[1] == "mcp":
        mcp()
        return
    app()


if __name__ == "__main__":
    main()
