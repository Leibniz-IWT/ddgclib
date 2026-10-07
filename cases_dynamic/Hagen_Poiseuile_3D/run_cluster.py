#!/usr/bin/env python3
"""
Cluster-ready 3D Hagen-Poiseuille pipe flow with PyTorch/CUDA backend.
=====================================================================

Designed for GPU nodes (e.g. H100).  Runs the shared runner body
(``cases_dynamic/Hagen_Poiseuile/src/_run.py``, preset ``hagen_poiseuille_3D``)
with:

  - Explicit backend selection (``--backend torch|gpu|numpy|multiprocessing``),
    the method axis ``backend`` replaced on the preset.  The backend only
    computes the dual face areas of the 3D retopology (``batch_e_star``);
    the forces of the preset do not read them, so the result is the numpy
    one.  ``torch`` needs PyTorch (ImportError otherwise); ``gpu`` picks
    PyTorch + CUDA, then PyTorch on the CPU, then numpy.
  - CUDA device reporting (driver, memory, compute capability)
  - Wall-clock throughput
  - Headless (no display): all plots saved to ``fig/``
  - SLURM-friendly: reads ``SLURM_*`` env vars for logging

Outputs: ``results/hagen_poiseuille_3D_cluster/``.

Usage on a SLURM cluster::

    sbatch <<'EOF'
    #!/bin/bash
    #SBATCH --job-name=hp3d
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=32G
    #SBATCH --time=04:00:00
    #SBATCH --output=hp3d_%j.log

    module load anaconda3 cuda
    conda activate ddg
    cd $SLURM_SUBMIT_DIR

    python run_cluster.py \
        --backend gpu \
        --n-refine 3 \
        --n-steps 5000 \
        --dt 0.005 \
        --workers 8
    EOF

Local test (quick smoke test; runs without PyTorch as well)::

    python run_cluster.py --backend gpu --n-refine 1 --n-steps 50 --dt 0.01
"""

import argparse
import os
import sys
import time

import numpy as np

# Force headless matplotlib before any other imports touch it
import matplotlib
matplotlib.use('Agg')

# Ensure project root on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ============================================================
# Diagnostics
# ============================================================

def print_env():
    """Print SLURM / system environment for reproducibility."""
    print("=" * 72)
    print("Environment")
    print("=" * 72)
    for key in ('SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_NODELIST',
                'SLURM_GPUS_ON_NODE', 'SLURM_CPUS_PER_TASK',
                'CUDA_VISIBLE_DEVICES', 'HOSTNAME'):
        val = os.environ.get(key)
        if val:
            print(f"  {key} = {val}")
    print(f"  Python = {sys.version}")
    print(f"  NumPy  = {np.__version__}")
    print()


def print_torch_info():
    """Print PyTorch and CUDA diagnostics."""
    try:
        import torch
        print(f"  PyTorch        = {torch.__version__}")
        print(f"  CUDA available = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            print(f"  CUDA device    = {torch.cuda.get_device_name(dev)}")
            cap = torch.cuda.get_device_capability(dev)
            print(f"  Compute cap.   = {cap[0]}.{cap[1]}")
            props = torch.cuda.get_device_properties(dev)
            mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
            print(f"  GPU memory     = {mem / 1e9:.1f} GB")
            print(f"  cuDNN          = {torch.backends.cudnn.version()}")
        else:
            print("  (CUDA not available — will use CPU tensors)")
    except ImportError:
        print("  PyTorch: NOT INSTALLED")
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cluster-ready 3D Hagen-Poiseuille with GPU backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--backend', type=str, default='gpu',
                        choices=['numpy', 'torch', 'gpu', 'multiprocessing'],
                        help='Computation backend for dual mesh ops')
    parser.add_argument('--n-refine', type=int, default=2,
                        help='Mesh refinement level')
    parser.add_argument('--n-steps', type=int, default=3000,
                        help='Number of time steps')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step size')
    parser.add_argument('--workers', type=int, default=8,
                        help='Processes for the force evaluation')
    args = parser.parse_args()

    print_env()
    if args.backend != 'numpy':
        print(f"Backend: {args.backend!r}")
        print_torch_info()

    # The run itself is the shared runner body with the preset
    # hagen_poiseuille_3D; backend and workers are method axes replaced on
    # it (recorded in methods.json).
    from cases_dynamic.Hagen_Poiseuile.src._run import run_case
    argv = ['--n-refine', str(args.n_refine), '--steps', str(args.n_steps),
            '--dt', str(args.dt), '--workers', str(args.workers),
            '--headless', '--tag', 'cluster']
    if args.backend != 'numpy':
        argv += ['--backend', args.backend]

    t0 = time.perf_counter()
    summary = run_case('hagen_poiseuille_3D', argv)
    wall = time.perf_counter() - t0

    print(f"\n  Throughput: {args.n_steps / wall:.1f} steps/s")
    print(f"  Final vertex count: {summary['n_vertices_end']}")
    print(f"  Wall time: {wall:.1f} s ({wall / 60:.1f} min)")
    print("\nVisualize with:")
    print("  python visualize_hp3d.py --no-polyscope "
          "--results-dir results/hagen_poiseuille_3D_cluster")


if __name__ == "__main__":
    main()
