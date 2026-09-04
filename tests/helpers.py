"""Small helpers shared by the server-backed test scripts."""
from __future__ import annotations

import socket


def free_port() -> int:
    """Ask the OS for an unused port.

    Fixed ports make these tests flaky: a stub server orphaned by an earlier
    timed-out run keeps listening and the next run talks to the wrong process.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
