#!/usr/bin/env python3
"""Shared timing helpers for Figure 3-8 recomputation scripts."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")

METHOD_ORDER = [
    "plic_pl",
    "evrard_type_paraboloid",
    "thinc_qq",
    "strobl_sphere_overlap",
    "present_quadric_patch",
]

METHOD_LABELS = {
    "plic_pl": "PLIC / PL",
    "evrard_type_paraboloid": "Evrard-type paraboloid",
    "thinc_qq": "THINC/QQ",
    "strobl_sphere_overlap": "Strobl sphere overlap",
    "present_quadric_patch": "Present quadric patch",
}


def perf_now() -> float:
    return time.perf_counter()


def new_timings() -> dict[str, float]:
    return {method: 0.0 for method in METHOD_ORDER}


def timed(timings: dict[str, float], method: str, func: Callable[..., T], *args, **kwargs) -> T:
    start = perf_now()
    try:
        return func(*args, **kwargs)
    finally:
        timings[method] = timings.get(method, 0.0) + perf_now() - start


def format_duration(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1000.0:.2f} ms"
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f} ms"
    return f"{seconds:.3f} s"


def print_timing_summary(case: str, timings: dict[str, float], total_seconds: float | None = None) -> None:
    print()
    print(f"Timing summary for {case}:", flush=True)
    for method in METHOD_ORDER:
        print(f"  {METHOD_LABELS[method]:28s} {format_duration(timings.get(method, 0.0))}", flush=True)
    if total_seconds is not None:
        print(f"  {'case total':28s} {format_duration(total_seconds)}", flush=True)
