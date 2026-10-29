"""One command to start everything:  python run_dashboard.py

Uses the project's own .venv automatically (so the global Python needs no packages), builds the
web UI on first run or when its sources changed, starts the server and opens the browser.
Extra flags are passed through:  --port 8080  --host 0.0.0.0  --no-browser  --rebuild
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _venv_python() -> Path:
    sub = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return ROOT.joinpath(".venv", *sub)


if __name__ == "__main__":
    os.chdir(ROOT)
    venv = _venv_python()
    inside_venv = Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    if venv.is_file() and not inside_venv:
        sys.exit(subprocess.call([str(venv), str(Path(__file__).resolve()), *sys.argv[1:]]))
    sys.path.insert(0, str(ROOT))
    from dashboard.api.main import main

    sys.exit(main(sys.argv[1:]))
