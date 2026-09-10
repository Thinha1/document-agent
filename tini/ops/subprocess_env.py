"""Environment stripping for isolated benchmark subprocesses."""

from __future__ import annotations

import os

_DENY_VAR = "TINI_BENCHMARK_ENV_DENY"


def benchmark_env() -> dict[str, str]:
    """Copy the process environment minus explicitly denied variable names."""
    denied = {name.strip() for name in os.getenv(_DENY_VAR, "").split(",") if name.strip()}
    return {name: value for name, value in os.environ.items() if name not in denied}
