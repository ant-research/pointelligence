# Register third-party warning filters before any heavy import (spconv,
# torch, numpy) so they are active in every process — launcher, mp.spawn
# DDP ranks, and spawn-based DataLoader workers. See _warning_filters for
# the per-entry rationale. Must stay the first statement in this module.
from . import _warning_filters  # noqa: F401
