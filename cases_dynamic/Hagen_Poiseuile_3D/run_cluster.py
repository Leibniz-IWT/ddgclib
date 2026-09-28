#!/usr/bin/env python3
"""
Cluster-ready 3D Hagen-Poiseuille pipe flow with PyTorch/CUDA backend.
=====================================================================

Designed for GPU nodes (e.g. H100).  Wraps the existing case script with:

  - Explicit backend selection (``--backend torch|gpu|numpy|multiprocessing``)
  - CUDA device reporting (driver, memory, compute capability)
  - Wall-clock timing per phase and per-step throughput
  - Headless (no display) — all plots saved to ``fig/``
  - SLURM-friendly: reads ``SLURM_*`` env vars for logging

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

Local test (quick smoke test)::

    python run_cluster.py --backend torch --n-refine 1 --n-steps 50 --dt 0.01
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


def get_backend_instance(name, workers=None):
    """Create a backend instance by name, with diagnostics."""
    from hyperct._backend import get_backend

    print(f"Initializing backend: {name!r}")
    print_torch_info()

    if name == "multiprocessing":
        w = workers or 4
        backend = get_backend("multiprocessing", workers=w)
        print(f"  Multiprocessing workers = {w}")
    elif name in ("torch", "gpu"):
        backend = get_backend(name)
        print(f"  Backend device = {backend.device}")
        if hasattr(backend, 'has_cuda'):
            print(f"  Using CUDA     = {backend.has_cuda}")
    elif name == "numpy":
        backend = get_backend("numpy")
    else:
        raise ValueError(f"Unknown backend: {name!r}.  "
                         "Choose from: numpy, torch, gpu, multiprocessing")

    print(f"  Backend ready: {backend.name}")
    print()
    return backend


# ============================================================
# Timer utility
# ============================================================

class PhaseTimer:
    """Simple wall-clock timer for named phases."""

    def __init__(self):
        self.phases = {}
        self._current = None
        self._t0 = None

    def start(self, name):
        self._current = name
        self._t0 = time.perf_counter()

    def stop(self):
        if self._current and self._t0:
            elapsed = time.perf_counter() - self._t0
            self.phases[self._current] = elapsed
            self._current = None
            self._t0 = None
            return elapsed
        return 0.0

    def report(self):
        print("\n" + "=" * 72)
        print("Timing summary")
        print("=" * 72)
        total = 0.0
        for name, dt in self.phases.items():
            print(f"  {name:<30s}  {dt:>10.2f} s")
            total += dt
        print(f"  {'TOTAL':<30s}  {total:>10.2f} s")
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
                        help='Threads for parallel stress computation')
    parser.add_argument('--record-every', type=int, default=100,
                        help='Record snapshot every N steps')
    parser.add_argument('--save-every', type=int, default=500,
                        help='Save state to disk every N steps')
    args = parser.parse_args()

    timer = PhaseTimer()

    # --- Environment ---
    print_env()

    # --- Backend ---
    timer.start("Backend initialization")
    backend = get_backend_instance(args.backend, workers=args.workers)
    timer.stop()

    # --- Import case module (after backend is ready) ---
    import Hagen_Poiseuile_3D as hp3d

    # Override module-level config from CLI args
    hp3d.n_refine = args.n_refine
    hp3d.n_workers = args.workers
    hp3d.record_every = args.record_every
    hp3d.save_every = args.save_every
    hp3d._retop_backend = backend

    hp3d.print_params()

    # --- Step 1: Build domain ---
    timer.start("Domain construction")
    HC, bV, domain_result = hp3d.build_domain(args.n_refine)
    timer.stop()

    n_verts = sum(1 for _ in HC.V)
    print(f"  Total vertices: {n_verts}")

    # --- Step 2: Inlet ghost mesh ---
    timer.start("Inlet mesh construction")
    unit_mesh = hp3d.build_inlet_mesh(args.n_refine)
    timer.stop()

    # --- Step 3: Boundary conditions ---
    timer.start("Boundary conditions setup")
    bc_set = hp3d.build_bc_set(HC, bV, unit_mesh)
    timer.stop()

    # --- Step 4: Initial conditions ---
    timer.start("Initial conditions")
    hp3d.apply_initial_conditions(HC, bV)
    bc_set.apply_all(HC, bV, dt=0.0)

    from hyperct.ddg import compute_vd
    compute_vd(HC, method="barycentric", backend=backend)
    timer.stop()

    # --- Step 5: Run simulation ---
    timer.start("Simulation")
    t0_sim = time.perf_counter()

    t_final, history = hp3d.run_simulation(
        HC, bV, bc_set,
        n_steps_override=args.n_steps,
        dt_override=args.dt,
    )

    sim_wall = time.perf_counter() - t0_sim
    timer.stop()

    # Throughput metrics
    n_verts_final = sum(1 for _ in HC.V)
    steps_per_sec = args.n_steps / sim_wall if sim_wall > 0 else 0
    print(f"\n  Throughput: {steps_per_sec:.1f} steps/s")
    print(f"  Final vertex count: {n_verts_final}")
    print(f"  Wall time: {sim_wall:.1f} s "
          f"({sim_wall/60:.1f} min)")

    # --- Step 6: Save and analyse ---
    timer.start("Post-processing")
    hp3d.save_results(HC, bV, t_final, history)
    hp3d.analyse_profile(HC, bV)
    timer.stop()

    # --- Timing report ---
    timer.report()

    print("Done. Results in:")
    print(f"  {hp3d._RESULTS}/")
    print(f"  {hp3d._FIG}/")
    print("\nVisualize with:")
    print(f"  python visualize_hp3d.py --no-polyscope")


if __name__ == "__main__":
    main()
