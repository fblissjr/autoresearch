"""
Project-wide logging framework for autoresearch-mlx.

Usage:
    from log_utils import logger, is_debug

    logger.info("Training started")
    logger.debug("Step details: %s", details)  # only shown with --debug

Enable debug mode by passing --debug flag to any script, or setting
the AUTORESEARCH_DEBUG=1 environment variable.
"""

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"


def _check_debug():
    """Check if debug mode is enabled via --debug flag or env var."""
    if os.environ.get("AUTORESEARCH_DEBUG", "0") == "1":
        return True
    if "--debug" in sys.argv:
        sys.argv.remove("--debug")
        return True
    return False


is_debug = _check_debug()

logger = logging.getLogger("autoresearch")
logger.setLevel(logging.DEBUG if is_debug else logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG if is_debug else logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    logger.addHandler(handler)
