import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'benchmarks'))

from benchmarks._benchmark_cases import TorusBenchmark

if __name__ == "__main__":
    benchmark = TorusBenchmark()
    benchmark.run_benchmark()
    results = benchmark.summary()

    for metric, (computed, analytical) in results.items():
        print(f"{metric}:\n  Computed = {computed:.6f}\n  Analytical = {analytical:.6f}\n")
