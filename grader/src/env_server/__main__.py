"""Make ``python -m env_server`` work.

The supervisor spawns the env server via this entry point.
"""

from .server import serve

if __name__ == "__main__":
    serve()
