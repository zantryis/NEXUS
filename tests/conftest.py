"""Root conftest: force-exit after test session to avoid aiosqlite thread hang."""

import atexit
import os

_exit_code = 0


def pytest_sessionfinish(session, exitstatus):
    global _exit_code
    _exit_code = exitstatus


def pytest_unconfigure(config):
    """Force process exit after all output is printed.

    aiosqlite creates background threads for each connection. When many async
    test fixtures create KnowledgeStore instances, the threads can outlive the
    event loop and prevent clean shutdown. This hook fires after the terminal
    summary, ensuring the process exits promptly.
    """
    atexit.register(os._exit, _exit_code)
