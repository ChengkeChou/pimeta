"""``python -m pimeta`` — same entry point as the ``pimeta`` console script."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
