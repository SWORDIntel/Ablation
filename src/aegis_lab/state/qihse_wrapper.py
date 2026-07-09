"""
aegis_lab QIHSE wrapper — thin re-export of the main framewerx.state wrapper.

The aegis_lab subsystem previously had its own toy in-memory stub.
It now delegates to the real process-isolated ctypes wrapper so that
all subsystems share one QIHSE library instance (same .so, same cwd fix).
"""
from framewerx.state.qihse_wrapper import (  # noqa: F401
    QIHSE,
    QihseVectorDBBackend,
    QihseQueryMode,
    QihseDistanceMetric,
    QihseOpenFlags,
    QihseMemoryTier,
)

import logging
logger = logging.getLogger(__name__)
