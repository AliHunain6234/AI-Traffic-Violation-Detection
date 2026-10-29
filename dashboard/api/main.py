"""FastAPI app: REST + WebSocket + MJPEG for the traffic-violation dashboard.

    uvicorn dashboard.api.main:app --reload

The config path comes from the TVD_CONFIG environment variable (default config.yaml).
If dashboard/web/dist exists (``npm run build``) it is served from the same port.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.utils import load_config

from .routes import analytics, events, live, setup
from .services.context import AppContext
from .services.events_service import EventsService
from .services.live_service import LiveService

log = logging.getLogger("dashboard")


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Build the app from a config file."""
    path = config_path or os.environ.get("TVD_CONFIG", "config.yaml")
    cfg = load_config(path)
    dcfg = cfg.get("dashboard", {})
    app = FastAPI(title="Traffic Violation Dashboard API", version="1.0.0")
    app.state.ctx = AppContext(cfg=cfg, events=EventsService(cfg), live=LiveService(cfg),
                               token=dcfg.get("api_token") or os.environ.get("TVD_TOKEN"), config_path=str(path))
    app.add_middleware(CORSMiddleware, allow_origins=dcfg.get("cors_origins", []),
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    for module in (analytics, events, live, setup):
        app.include_router(module.router)

    dist = Path(dcfg.get("serve_static", "dashboard/web/dist"))
    if dist.is_dir():
        index = dist / "index.html"

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            """Serve built assets; unknown non-API paths fall back to the SPA entry point."""
            if path == "api" or path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            target = (dist / path).resolve()
            if path and target.is_file() and target.is_relative_to(dist.resolve()):
                return FileResponse(target)
            return FileResponse(index)
    if not app.state.ctx.token:
        log.warning("No dashboard.api_token set: the API is open. Set one before exposing it publicly.")
    return app


def _lower_priority() -> None:
    """Run below normal priority: the detector then only takes the CPU the browser and Windows leave free."""
    try:
        if os.name == "nt":
            import ctypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p          # a 64-bit pseudo-handle; the default int type breaks it
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass.restype = ctypes.c_int
            if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000):    # BELOW_NORMAL_PRIORITY_CLASS
                log.debug("SetPriorityClass failed: %s", ctypes.get_last_error())
        else:
            os.nice(5)
    except Exception as exc:                          # purely an optimisation
        log.debug("Could not lower the process priority: %s", exc)


def main(argv: Optional[list[str]] = None) -> int:
    """One-command launcher: builds the UI if needed, starts the server, opens the browser.

        python -m dashboard.api.main
    """
    import argparse
    import shutil
    import subprocess
    import threading
    import webbrowser

    import uvicorn

    p = argparse.ArgumentParser(description="Start the traffic-violation dashboard")
    p.add_argument("--host", help="default: dashboard.host (use 127.0.0.1 for local-only)")
    p.add_argument("--port", type=int, help="default: dashboard.port")
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    p.add_argument("--rebuild", action="store_true", help="rebuild the web UI even if dist/ exists")
    args = p.parse_args(argv)

    os.chdir(ROOT)
    dcfg = load_config(os.environ.get("TVD_CONFIG", "config.yaml")).get("dashboard", {})
    host = args.host or "127.0.0.1"
    port = args.port or int(dcfg.get("port", 8000))
    web = ROOT / "dashboard" / "web"
    built = web / "dist" / "index.html"
    stale = built.is_file() and any(
        f.stat().st_mtime > built.stat().st_mtime
        for f in [*(web / "src").rglob("*"), web / "index.html", web / "tailwind.config.js"] if f.is_file())
    if args.rebuild or stale or not built.is_file():
        npm = shutil.which("npm")
        if npm is None:
            log.warning("npm not found: starting the API only (install Node.js to build the web UI)")
        else:
            log.info("Building the web UI (first run or after UI changes, takes a minute)...")
            steps = [[npm, "run", "build"]]
            if not (web / "node_modules").is_dir():
                steps.insert(0, [npm, "install", "--no-audit", "--no-fund"])
            for cmd in steps:
                if subprocess.run(cmd, cwd=web).returncode != 0:
                    log.error("Web UI build failed; starting the API only")
                    break
    if dcfg.get("low_priority", True):
        _lower_priority()
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    if not args.no_browser:
        threading.Timer(2.0, webbrowser.open, args=(url,)).start()
    print(f"\nDashboard: {url}   (Ctrl+C to stop)\n")
    uvicorn.run("dashboard.api.main:app", host=host, port=port, log_level="warning")
    return 0


ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    os.chdir(ROOT)  # config.yaml and all relative paths are resolved from the project root

app = create_app()

if __name__ == "__main__":
    raise SystemExit(main())
