#!/usr/bin/env python3
"""Interactive polyscope viewer for ddgclib simulation snapshots.

Loads ``StateHistory`` JSON snapshots from a directory and launches
the interactive polyscope timeline viewer with configurable scalar
and vector field overlays.

Works with any ddgclib case study that saves snapshots via
``StateHistory(save_dir=...)``.

Usage
-----
    # View 2D/3D snapshots from any case:
    python -m ddgclib.scripts.view_polyscope --snapshots results/snapshots/

    # Specify fields:
    python -m ddgclib.scripts.view_polyscope \\
        --snapshots results/snapshots/ \\
        --scalars p phase \\
        --vectors u

    # Save screenshots:
    python -m ddgclib.scripts.view_polyscope \\
        --snapshots results/snapshots/ \\
        --screenshot-dir results/screenshots/
"""
import argparse
import glob
import os
import sys


def load_history_from_dir(snapshot_dir, fields=('u', 'p')):
    """Reconstruct a StateHistory from saved JSON snapshots.

    Parameters
    ----------
    snapshot_dir : str
        Directory containing ``state_*.json`` files.
    fields : tuple of str
        Field names to load from each snapshot.

    Returns
    -------
    history : StateHistory or None
    HC : Complex or None
        The last loaded complex (for dimension detection).
    """
    from ddgclib.data import StateHistory, load_state

    json_files = sorted(glob.glob(os.path.join(snapshot_dir, 'state_*.json')))
    if not json_files:
        print(f"No state_*.json files found in {snapshot_dir}")
        return None, None

    print(f"Loading {len(json_files)} snapshots from {snapshot_dir}")

    history = StateHistory(fields=list(fields), record_every=1)
    HC_last = None

    for i, path in enumerate(json_files):
        HC, bV, meta = load_state(path)
        t = meta.get('time', i * 0.001)

        snapshot = {}
        for v in HC.V:
            key = v.x
            vertex_data = {}
            for f in fields:
                val = getattr(v, f, None)
                if val is not None:
                    vertex_data[f] = (val.copy() if hasattr(val, 'copy')
                                      else val)
            snapshot[key] = vertex_data
        history._snapshots.append((t, snapshot, {}))
        HC_last = HC

        if (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{len(json_files)}")

    print(f"Loaded {history.n_snapshots} snapshots, "
          f"t = [{history.times[0]:.4e}, {history.times[-1]:.4e}]")
    return history, HC_last


def main():
    parser = argparse.ArgumentParser(
        description="Interactive polyscope viewer for ddgclib snapshots",
    )
    parser.add_argument(
        '--snapshots', type=str, required=True,
        help='Directory containing state_*.json files',
    )
    parser.add_argument(
        '--scalars', type=str, nargs='+', default=['p'],
        help='Scalar fields to display (default: p)',
    )
    parser.add_argument(
        '--vectors', type=str, nargs='+', default=['u'],
        help='Vector fields to display (default: u)',
    )
    parser.add_argument(
        '--screenshot-dir', type=str, default=None,
        help='Save polyscope screenshots to this directory',
    )
    parser.add_argument(
        '--name', type=str, default='simulation',
        help='Polyscope structure name prefix',
    )
    args = parser.parse_args()

    if not os.path.isdir(args.snapshots):
        print(f"Directory not found: {args.snapshots}")
        sys.exit(1)

    all_fields = list(set(args.scalars + args.vectors))
    history, HC = load_history_from_dir(args.snapshots, fields=all_fields)

    if history is None or HC is None:
        sys.exit(1)

    try:
        from ddgclib.visualization.polyscope_3d import interactive_history_viewer
    except ImportError:
        print("Polyscope not installed.  Install with:")
        print("  pip install polyscope")
        sys.exit(1)

    print("\nLaunching polyscope viewer...")
    print(f"  Scalar fields: {args.scalars}")
    print(f"  Vector fields: {args.vectors}")
    print("  Controls: frame slider, play/pause, speed")

    interactive_history_viewer(
        history, HC,
        scalar_fields=args.scalars,
        vector_fields=args.vectors,
        name=args.name,
        screenshot_dir=args.screenshot_dir,
    )


if __name__ == '__main__':
    main()
