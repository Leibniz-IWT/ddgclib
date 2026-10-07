"""Dynamic isolation benchmarks (Tier 1 single-phase, Tier 2/3 multiphase).

Each rung strips exactly one source of complexity from the full dynamic
pipeline so a failure attributes to the newly-added concern.  See the
stabilisation roadmap (Tier 1) and ``_harness.py`` for the shared
decaying-sinusoid setup reused by 1A / 1B.i / 1B.ii.
"""
