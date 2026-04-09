#!/usr/bin/env python3
"""Interactive polyscope viewer for oscillating droplet data.

Thin wrapper around ``ddgclib.scripts.view_polyscope`` with default
paths for this case study.

Usage
-----
    # 2D (default):
    python cases_dynamic/oscillating_droplet/view_polyscope.py

    # 3D:
    python cases_dynamic/oscillating_droplet/view_polyscope.py --dim 3

    # Or use the generic library viewer directly:
    python -m ddgclib.scripts.view_polyscope \\
        --snapshots results/oscillating_droplet_2D/snapshots/ \\
        --scalars p phase --vectors u
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ddgclib.scripts.view_polyscope import load_history_from_dir


def main():
    parser = argparse.ArgumentParser(
        description="Polyscope viewer for oscillating droplet")
    parser.add_argument('--dim', type=int, default=2, help='2 or 3')
    parser.add_argument('--screenshot-dir', type=str, default=None)
    args = parser.parse_args()

    case_dir = os.path.dirname(os.path.abspath(__file__))
    snap_dir = os.path.join(case_dir, 'results', 'snapshots')

    if not os.path.isdir(snap_dir):
        print(f"Snapshot directory not found: {snap_dir}")
        print("Run the simulation first:")
        print(f"  python cases_dynamic/oscillating_droplet/"
              f"oscillating_droplet_{args.dim}D.py")
        return

    history, HC = load_history_from_dir(snap_dir, fields=('u', 'p'))

    if history is None:
        return

    try:
        from ddgclib.visualization.polyscope_3d import interactive_history_viewer
    except ImportError:
        print("Polyscope not installed.  Install with: pip install polyscope")
        return

    print(f"\nLaunching {args.dim}D polyscope viewer...")
    interactive_history_viewer(
        history, HC,
        scalar_fields=['p'],
        vector_fields=['u'],
        name=f'oscillating_droplet_{args.dim}D',
        screenshot_dir=args.screenshot_dir,
    )


if __name__ == '__main__':
    main()
