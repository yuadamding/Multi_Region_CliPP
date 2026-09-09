"""Test the checkout and its source-only tools, independent of install location."""
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPOSITORY.parent), str(REPOSITORY)]
