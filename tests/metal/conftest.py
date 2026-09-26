"""Explicit opt-in: these tests compile shaders and use the Apple GPU."""
import os

# Do not even import GPU test modules during the normal CPU/Linux test run.
collect_ignore_glob = [] if os.environ.get("MICRODUCK_RUN_METAL_TESTS") == "1" else ["test_*.py"]
