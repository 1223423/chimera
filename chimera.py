"""
Chimera - Mosaic Reconstruction

This file provides backwards compatibility by importing from the chimera module.
For new code, import from the chimera module directly.

Usage:
    python chimera.py [options]
    python -m chimera.main [options]
"""

from chimera.main import main

if __name__ == "__main__":
    main()
