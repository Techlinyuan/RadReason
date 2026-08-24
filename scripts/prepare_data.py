# -*- coding: utf-8 -*-
"""Thin CLI wrapper so `python scripts/prepare_data.py ...` works from the repo root."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from radreason.data import main

if __name__ == "__main__":
    main()
