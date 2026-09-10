"""Recover fog-head summaries and figures from an interrupted raw-data sweep.

The experiment runner writes its task/tick gzip streams incrementally but only
writes summary files and figures after every trial finishes.  This utility
reconstructs those final artifacts from all *complete* policy/rate groups in
the raw streams, without running any simulations or training.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from collections import defaultdict

import numpy as np

import fog_head_experiments as fhe
from config import N_FOG, P_COMM
from experiments import P_UAV_TX_W, SIGNAL_KB
from physics import data_rate_bps


META = ("arrival_rate", "episode_seconds", "trial", "seed", "policy")


def _key(row):
    return (float(row["arrival_rate"]), float(row["episode_seconds"]),
            int(row["trial"]), int(row["seed"]), row["policy"])


def _finite_mean(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _load_tasks(path):
    groups = defaultdict(lambda: defaultdict(list))
    with gzip.open(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            group = groups[_key(row)]
            for name in ("delay_ms", "success", "control_ms", "selector_ms",
                         "handover_ms", "control_j", "net_j"):
                group[name].append(float(row[name]))
    return groups


def _load_ticks(path):
    groups = defaultdict(lambda: defaultdict(list))
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.DictReader(fh)
        if "head_availability" not in (reader.fieldnames or ()):
            raise ValueError(
                "Raw tick data predates the 10-second head-availability metric; "
                "rerun fog_head_experiments.py instead of relabelling the old "
                "one-task reliability values."
            )
        for row in reader:
            group = groups[_key(row)]
            for name in ("t", "prop_power_w", "head_id", "head_soc",
                         "head_iot_distance_m", "head_fog_distance_m",
                         "head_workload_cycles", "head_availability",
                         "selector_us", "selector_ran", "head_changed"):
                group[name].append(float(row[name]))
            group["switch_block_reason"].append(row["switch_block_reason"])
    return groups


def _signal_energy(ticks):
    """Rebuild election signalling energy from initial/change tick geometry."""
    total = 0.0
    for index, (changed, distance) in enumerate(zip(
            ticks["head_changed"], ticks["head_fog_distance_m"])):
        if index != 0 and not changed:
            continue
        if not math.isfinite(distance):
            continue
        rate = data_rate_bps(distance, P_UAV_TX_W)
        seconds = SIGNAL_KB * 1024.0 * 8.0 / rate * (N_FOG - 1)
        total += (P_UAV_TX_W + P_COMM) * seconds
    return total


def _summarize(key, task, tick):
    rate, seconds, trial, seed, policy = key
    delays = np.asarray(task["delay_ms"], dtype=float)
    success = np.asarray(task["success"], dtype=float)
    n_tasks = len(delays)
    n_ticks = len(tick["t"])
    reasons = tick["switch_block_reason"]
    changes = int(sum(tick["head_changed"]))
    control_energy = sum(task["control_j"]) + _signal_energy(tick)
    return {
        "arrival_rate": rate, "episode_seconds": seconds,
        "trial": trial, "seed": seed, "policy": policy,
        "n_tasks": n_tasks,
        "duration_s": max(tick["t"]) if n_ticks else seconds,
        "mean_delay_ms": float(delays.mean()),
        "p50_delay_ms": float(np.percentile(delays, 50)),
        "p95_delay_ms": float(np.percentile(delays, 95)),
        "p99_delay_ms": float(np.percentile(delays, 99)),
        "success_rate": float(success.mean()),
        "deadline_miss_rate": 1.0 - float(success.mean()),
        "mean_control_delay_ms": _finite_mean(task["control_ms"]),
        "mean_selector_delay_ms": _finite_mean(task["selector_ms"]),
        "mean_handover_delay_ms": _finite_mean(task["handover_ms"]),
        "control_energy_mj_per_task": 1000.0 * control_energy / n_tasks,
        # Compute energy and executor identity were not retained in raw output.
        "nonprop_energy_j_per_delivered_task": float("nan"),
        "whole_swarm_energy_j": float("nan"),
        "head_changes": float(changes),
        "head_changes_per_1000_tasks": 1000.0 * changes / n_tasks,
        "mean_head_tenure_ticks": fhe._mean_tenure_ticks(tick["head_id"]),
        "selector_run_pct": 100.0 * sum(tick["selector_ran"]) / n_ticks,
        "mean_selector_runtime_us": _finite_mean(tick["selector_us"]),
        "mean_head_soc_pct": 100.0 * _finite_mean(tick["head_soc"]),
        "mean_head_iot_distance_m": _finite_mean(tick["head_iot_distance_m"]),
        "mean_head_fog_distance_m": _finite_mean(tick["head_fog_distance_m"]),
        "mean_head_workload_cycles": _finite_mean(tick["head_workload_cycles"]),
        "mean_head_availability": _finite_mean(tick["head_availability"]),
        "kl_gate_pct": 100.0 * reasons.count("kl_gate") / n_ticks,
        "tenure_block_pct": 100.0 * reasons.count("minimum_tenure") / n_ticks,
        "margin_block_pct": 100.0 * reasons.count("switch_margin") / n_ticks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="fog_head_figures")
    parser.add_argument("--output-dir", default="fog_head_recovered_12")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    task_path = os.path.join(args.raw_dir, "fog_head_task_results.csv.gz")
    tick_path = os.path.join(args.raw_dir, "fog_head_tick_results.csv.gz")
    tasks, ticks = _load_tasks(task_path), _load_ticks(tick_path)
    shared = sorted(set(tasks) & set(ticks))
    if not shared:
        raise RuntimeError("No matching task/tick result groups were found")

    expected_ticks = {
        key: int(round(key[1] / 0.1)) for key in shared
    }
    complete = [key for key in shared
                if len(ticks[key]["t"]) == expected_ticks[key]
                and len(tasks[key]["delay_ms"]) > 0]
    rows = [_summarize(key, tasks[key], ticks[key]) for key in complete]
    rates = sorted({key[0] for key in complete})
    trials = sorted({key[2] for key in complete})

    # A complete trial must contain every policy at every arrival rate.
    required = {(rate, policy) for rate in rates for policy in fhe.POLICIES}
    complete_trials = [trial for trial in trials if {
        (key[0], key[4]) for key in complete if key[2] == trial
    } == required]
    rows = [row for row in rows if row["trial"] in complete_trials]

    highest_rate = max(rates)
    representative_trial = min(complete_trials)
    representatives = {}
    for policy in fhe.POLICIES:
        key = next(key for key in complete if key[0] == highest_rate
                   and key[2] == representative_trial and key[4] == policy)
        representatives[policy] = {
            "rec": {"delay_ms": tasks[key]["delay_ms"]},
            "series": {"t": ticks[key]["t"], "head_id": ticks[key]["head_id"]},
        }

    os.makedirs(args.output_dir, exist_ok=True)
    metadata = {
        "design": "recovered complete trials from interrupted raw-data sweep",
        "source_task_file": os.path.abspath(task_path),
        "source_tick_file": os.path.abspath(tick_path),
        "arrival_rates_tasks_per_s": rates,
        "episode_seconds": sorted({key[1] for key in complete}),
        "trials": len(complete_trials),
        "trial_indices": complete_trials,
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "head_reevaluation_ticks": fhe.HEAD_REEVALUATION_TICKS,
        "head_min_tenure_ticks": fhe.HEAD_MIN_TENURE_TICKS,
        "payload_path": "direct IoT-to-executor; elected head carries control messages only",
        "bootstrap_samples": args.bootstrap_samples,
        "confidence_interval": "paired/nonparametric bootstrap, 95%",
        "recovery_note": (
            "All plotted metrics were reconstructed. nonprop_energy_j_per_"
            "delivered_task and whole_swarm_energy_j are NaN because compute "
            "energy/executor identity were not stored in the raw CSV files."
        ),
    }
    fhe.OUT_DIR = os.path.abspath(args.output_dir)
    fhe._write_summary(rows, metadata)
    fhe.make_figures(rows, representatives, args.bootstrap_samples)
    with open(os.path.join(args.output_dir, "RECOVERY_METADATA.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Recovered {len(rows)} conditions from {len(complete_trials)} complete trials")
    print(f"Results and figures written to {fhe.OUT_DIR}")


if __name__ == "__main__":
    main()
