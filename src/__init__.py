"""EXAQC.

Importing the package caps logging at INFO by default. loguru starts with a
DEBUG-level handler on stderr, and the code logs a great deal at DEBUG (every
encoder, decoder and gate built), which on long runs is enough to fill a disk.
Anything that imports ``src`` -- every entry point, script and test in the repo
-- therefore logs at INFO unless it asks for more, through its own
``--logging_level`` or an explicit ``logger.add(..., level=...)``.
"""

from __future__ import annotations

import sys

from loguru import logger

#: The level used when nothing more verbose has been asked for.
DEFAULT_LOG_LEVEL = "INFO"


def _cap_default_log_level() -> None:
    """Replaces loguru's built-in DEBUG handler with an INFO one.

    Only loguru's own default handler (id 0) is replaced, and only if it is
    still installed, so logging that was already configured before this
    package was imported is left alone.

    Returns:
        None. Removes loguru's default handler and adds a stderr handler at
        :data:`DEFAULT_LOG_LEVEL` in its place.
    """

    try:
        logger.remove(0)
    except ValueError:
        # the default handler was already removed: logging has been
        # configured deliberately, so keep what was chosen
        return
    logger.add(sys.stderr, level=DEFAULT_LOG_LEVEL)


_cap_default_log_level()
