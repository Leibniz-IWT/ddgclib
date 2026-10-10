#!/usr/bin/env python3
"""Static capillary rise, 2D slit, on the library integrator (laneI).

Runs ``PRESETS['capillary_rise_static_2D']`` through ``src/_static.py``:
a water column in a 4 mm slit standing in a reservoir settles onto the
Young-Laplace meniscus of its contact angle; the volume-averaged height
is compared with Jurin's height and the surface with the analytical
profile.  Options: ``--n-refine``, ``--n-tac``, ``--ic flat`` (start from
a flat meniscus), ``--arm remap``, ``--alpha-art``, ``--out``.

Usage
-----
    python cases_dynamic/capillary_rise/capillary_rise_2D.py
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.capillary_rise.src._static import run_case  # noqa: E402

if __name__ == '__main__':
    run_case('capillary_rise_static_2D')
