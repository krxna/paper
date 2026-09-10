"""Hierarchical training-seed sensitivity for the Figure 7 load study.

Example:
  python3 training_seed_sensitivity.py \
    --run seed42=figures \
    --run seed43=figures/seed43 \
    --run seed44=figures/seed44
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


LOADS = (0.75, 1.00, 1.25, 1.50)
BASELINES = ("random", "FODAS", "ReLIEF")
METRICS = ("productive", "wasted", "backlog_exposure")


def _read_run(spec: str) -> Tuple[str, List[Dict]]:
    if "=" not in spec:
        raise ValueError("--run must use LABEL=OUTPUT_DIRECTORY")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path) / "fig7_seed_load_policy.csv"
    if not label or not path.exists():
        raise ValueError(f"invalid sensitivity run {spec!r}")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return label, [
        row for row in rows if row["controller_placement"] == "native"
        and any(abs(float(row["target_load"]) - load) < 1e-9 for load in LOADS)]


def _paired_values(rows: Sequence[Mapping], baseline: str, metric: str) -> Dict[int, List[float]]:
    proposed = {
        (int(row["seed"]), float(row["target_load"])): float(row[metric])
        for row in rows if row["policy"] == "Attention+PPO"}
    other = {
        (int(row["seed"]), float(row["target_load"])): float(row[metric])
        for row in rows if row["policy"] == baseline}
    output: Dict[int, List[float]] = {}
    for (seed, load), value in proposed.items():
        if (seed, load) in other:
            output.setdefault(seed, []).append(value - other[(seed, load)])
    expected = len(LOADS)
    if not output or any(len(values) != expected for values in output.values()):
        raise ValueError(
            f"incomplete paired grid for baseline={baseline}, metric={metric}")
    return output


def hierarchical_summary(
    runs: Sequence[Tuple[str, Sequence[Mapping]]], *,
    bootstrap_samples: int = 10_000, seed: int = 7127,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    summaries = []
    for baseline in BASELINES:
        for metric in METRICS:
            nested = {
                label: _paired_values(rows, baseline, metric)
                for label, rows in runs}
            labels = list(nested)
            observed = float(np.mean([
                value for per_scenario in nested.values()
                for values in per_scenario.values() for value in values]))
            boot = np.empty(bootstrap_samples, dtype=float)
            for index in range(bootstrap_samples):
                sampled_training = rng.choice(labels, size=len(labels), replace=True)
                values = []
                for label in sampled_training:
                    scenario = nested[str(label)]
                    scenario_ids = list(scenario)
                    sampled_scenarios = rng.choice(
                        scenario_ids, size=len(scenario_ids), replace=True)
                    values.extend(
                        np.mean(scenario[int(scenario_seed)])
                        for scenario_seed in sampled_scenarios)
                boot[index] = float(np.mean(values))
            summaries.append({
                "proposed": "Attention+PPO", "baseline": baseline,
                "metric": metric, "loads": ",".join(map(str, LOADS)),
                "training_seeds": len(labels),
                "scenario_seeds_per_training_seed": len(next(iter(nested.values()))),
                "mean_paired_difference": observed,
                "hierarchical_ci95_low": float(np.percentile(boot, 2.5)),
                "hierarchical_ci95_high": float(np.percentile(boot, 97.5)),
                "bootstrap_samples": bootstrap_samples,
            })
    return summaries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True,
        help="training-seed label and study directory as LABEL=PATH")
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/fig7_training_seed_sensitivity.csv"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args(argv)
    if len(args.run) < 3:
        parser.error("at least three independently trained runs are required")
    rows = hierarchical_summary(
        [_read_run(spec) for spec in args.run],
        bootstrap_samples=args.bootstrap_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
