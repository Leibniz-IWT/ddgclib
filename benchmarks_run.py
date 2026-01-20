import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'benchmarks'))

from benchmarks._benchmark_cases import TorusBenchmark, EllipsoidMshBenchmark

if __name__ == "__main__":
    benchmark = TorusBenchmark()
    benchmark.run_benchmark()
    results = benchmark.summary()

    for metric, (computed, analytical) in results.items():
        print(f"{metric}:\n  Computed = {computed:.6f}\n  Analytical = {analytical:.6f}\n")


    # ---------------- Ellipsoid (new) ----------------
    ellip = EllipsoidMshBenchmark(
        msh_path="test_cases/Ellip_0_sub0_full.msh",
        ax=1.5, ay=1.0, az=0.8,
        workdir="benchmarks_out/ellipsoid",
    )
    ellip.run()

    print("Ellipsoid:")
    print(f"  V_theory = {ellip.V_theory:.8f}")
    print(f"  V_flat   = {ellip.V_flat:.8f}")
    print(f"  V_sum    = {ellip.V_sum:.8f}")
    print(f"  V_total  = {ellip.V_total:.8f}")
    print(f"  rel_err% = {ellip.rel_err_percent:.6f}")