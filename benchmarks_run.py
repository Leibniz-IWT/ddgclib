# benchmarks_run.py (at repo root)
import argparse
from benchmarks._benchmark_cases import QuadricCoeffsFromMsh
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="Path to .msh")
    # Optional knobs passed straight to Part-1 (tweak or omit; these are safe defaults)
    ap.add_argument("--theta_max_deg", type=float, default=35.0)
    ap.add_argument("--min_pts", type=int, default=10)
    ap.add_argument("--plane_rel", type=float, default=1e-3)
    ap.add_argument("--plane_abs", type=float, default=1e-9)
    ap.add_argument("--irls", type=int, default=0)
    ap.add_argument("--snap_rel", type=float, default=5e-6)
    ap.add_argument("--snap_abs", type=float, default=1e-9)
    ap.add_argument("--canonize_abc", action="store_true", default=True)
    ap.add_argument("--no_canonize_abc", dest="canonize_abc", action="store_false")
    ap.add_argument("--debug_j", action="store_true", default=False)
    args = ap.parse_args()

    method = {
        # These are **not** used for Part-1 computations, just to keep the base class happy.
        # (We set valid defaults inside the case anyway.)
        "coeffs_kwargs": {
            "min_pts": args.min_pts,
            "theta_max_deg": args.theta_max_deg,
            "plane_rel": args.plane_rel,
            "plane_abs": args.plane_abs,
            "irls": args.irls,
            "snap_rel": args.snap_rel,
            "snap_abs": args.snap_abs,
            "canonize_abc": args.canonize_abc,
            "debug_j": args.debug_j,
        }
    }

    bench = QuadricCoeffsFromMsh(args.mesh, method=method)
    bench.run_benchmark()

    print(f"Benchmark: {bench.name}")
    for k, (v, _) in bench.summary().items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
