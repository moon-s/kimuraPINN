"""Project-local interpreter defaults."""

from __future__ import annotations

import os
import sys


if any(arg.endswith("pytest") or arg == "-m pytest" for arg in sys.argv):
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

