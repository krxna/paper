"""Benchmark all nine controller paths as a Markdown table."""

import argparse

from config import CONTROLLER_REFERENCE_HZ, PROFILE_FOG_SIZES
from controller_profile import _benchmark_paths, _measure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--fog-sizes", default=",".join(map(str, PROFILE_FOG_SIZES)),
        help="comma-separated fog sizes; N_IoT is 10*N_FOG")
    args = parser.parse_args()
    if args.warmups < 1 or args.samples < 1:
        parser.error("warmups and samples must be positive")

    try:
        fog_sizes = [int(value) for value in args.fog_sizes.split(",")]
    except ValueError:
        parser.error("--fog-sizes must be comma-separated integers")
    if not fog_sizes or min(fog_sizes) < 2:
        parser.error("--fog-sizes must contain values >= 2")

    for n_fog in fog_sizes:
        results = {
            name: _measure(
                path, warmups=args.warmups, samples=args.samples,
                reference_frequency_hz=CONTROLLER_REFERENCE_HZ)
            for name, path in _benchmark_paths(n_fog, 10 * n_fog).items()
        }
        reference = results["stage1_without_spotis"]["mean_cpu_s"]
        print(f"## N_FOG={n_fog}, N_IoT={10 * n_fog}\n")
        print("| path | mean ms | median ms | p95 ms | vs gated |")
        print("|---|---:|---:|---:|---:|")
        for name, stats in results.items():
            print(f"| {name} | {stats['mean_cpu_s'] * 1000:.4f} | "
                  f"{stats['median_cpu_s'] * 1000:.4f} | "
                  f"{stats['p95_cpu_s'] * 1000:.4f} | "
                  f"{stats['mean_cpu_s'] / reference:.2f}x |")
        print()


if __name__ == "__main__":
    main()
