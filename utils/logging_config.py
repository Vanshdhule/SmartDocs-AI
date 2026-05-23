# utils/logging_config.py
"""
SmartDocs AI — Unified Logging Configuration
─────────────────────────────────────────────
Call setup_logging() once at application startup (app.py).
All other modules simply do:

    import logging
    logger = logging.getLogger("smartdocs.<module>")

This replaces the scattered logging.basicConfig() calls that previously
created two separate log files (ingestion.log at root + logs/app.log).
"""

import logging
import sys
from pathlib import Path

try:
    from config import LOG_DIR, LOG_FILE
except ImportError:
    from config import LOG_DIR, LOG_FILE  # type: ignore


_CONFIGURED = False   # guard: only run setup once per process


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root 'smartdocs' logger.

    Handlers
    --------
    - StreamHandler  → stdout  (INFO and above)
    - FileHandler    → logs/app.log  (DEBUG and above, rotating on restart)

    Safe to call multiple times — only the first call takes effect.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — full DEBUG detail to disk
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above to stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root = logging.getLogger("smartdocs")
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)
    root.propagate = False   # don't bubble up to the root Python logger