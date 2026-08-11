"""Make sibling modules (``_harness``) importable when pytest collects the
benchmark tests as a package."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
