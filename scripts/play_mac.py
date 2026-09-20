#!/usr/bin/env python3
"""Launch the existing policy viewer with native macOS Python library discovery.

Run with the repository's virtual environment; all arguments go to infer_policy.py.
"""

import os
from pathlib import Path
import platform
import sys


def main():
    if platform.system() != 'Darwin':
        raise SystemExit('This launcher is for macOS; use infer_policy.py on other platforms.')
    repository = Path(__file__).resolve().parents[1]
    mjpython = Path(sys.executable).parent / 'mjpython'
    if not mjpython.is_file():
        raise SystemExit('mjpython is missing. Run uv sync --locked --extra mac-gpu --python 3.12 first.')
    # mjpython dlopens the interpreter, changing @executable_path. A uv venv
    # symlink can make its bundled discovery resolve .venv/lib instead of the
    # base interpreter's lib directory. Scope the correction to this process.
    paths = [str(Path(sys.base_prefix) / 'lib')]
    paths.extend(os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '/usr/local/lib:/usr/lib').split(':'))
    os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = ':'.join(p for p in paths if p)
    os.environ.pop('MUJOCO_GL', None)  # Native macOS viewer, not Linux EGL/OSMesa.
    os.execv(sys.executable, [sys.executable, str(mjpython),
                             str(repository / 'scripts/infer_policy.py'), *sys.argv[1:]])


if __name__ == '__main__':
    main()
