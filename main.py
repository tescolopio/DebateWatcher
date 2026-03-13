"""
main.py – Entry point for the DebateWatcher pipeline.

Usage
-----
    python main.py [--session SESSION_ID] [--no-obs] [--duration SECONDS]

For full option reference::

    python main.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Ensure the project root is on the Python path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from src.core.orchestrator import Orchestrator


def _configure_logging() -> None:
    settings.ensure_data_dirs()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(settings.pipeline.log_file))
    except OSError:
        pass
    logging.basicConfig(
        level=getattr(logging, settings.pipeline.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


async def _run(args: argparse.Namespace) -> None:
    orch = Orchestrator(session_id=args.session)

    loop = asyncio.get_running_loop()

    def _shutdown() -> None:
        logging.getLogger(__name__).info("Shutdown signal received.")
        asyncio.create_task(orch.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    await orch.start()

    if args.duration:
        await asyncio.sleep(args.duration)
        await orch.stop()
    else:
        # Run indefinitely until interrupted
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await orch.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DebateWatcher – real-time AI debate analysis pipeline"
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session identifier (used to namespace the database entries)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Run for SECONDS then exit (default: run until SIGINT/SIGTERM)",
    )
    args = parser.parse_args()

    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting DebateWatcher (session=%s)", args.session)

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
