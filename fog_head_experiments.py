"""Journal-oriented fog-head selector comparison.

The experiment holds the episode horizon fixed and sweeps Poisson arrival rate,
so the x-axis represents offered load rather than episode length.  Every policy
uses the same seed, task generator, mobility, and frozen Stage-2 actor.  Summary,
per-task, and per-tick results plus vector/raster figures are written to
``fog_head_figures``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import io
import json
import os
import signal
import tempfile
import time
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(),
                                                   "uav_fog_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(),
                                                     "uav_fog_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from broker import BrokerSelector
from config import (CONTROL_TICK_S, HEAD_MIN_TENURE_TICKS,
                    HEAD_REEVALUATION_TICKS, LEARNING_RATE, N_CRITERIA,
                    N_FOG, N_IOT, SEED, set_global_seed)
from experiments import HANDOVER_SETUP_MS, run_exp
from neural import AttentionEncoder, Actor, Critic, behavior_clone_fodas
import simulation


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fog_head_figures")
POLICIES = ("spotis", "fu_serve", "2dp_fhs", "3d_pos")
LABELS = {
    "spotis": "Proposed SPOTIS-MCDM",
    "fu_serve": "FU-Serve",
    "2dp_fhs": "2DP-FHS",
    "3d_pos": "3D-POS",
}
# Color-blind-safe palette; the proposed method has the strongest visual weight.
COLORS = {
    "spotis": "#0072B2",
    "fu_serve": "#555555",
    "2dp_fhs": "#D55E00",
    "3d_pos": "#009E73",
}
MARKERS = {"spotis": "o", "fu_serve": "s", "2dp_fhs": "^", "3d_pos": "D"}

_STOP_AFTER_TRIAL = False

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes:d}m {secs:02d}s"


def _shared_nets(seed: int, ppo_episodes: int, trial_number: int,
                 total_trials: int):
    label = f"[TRAIN {trial_number:02d}/{total_trials:02d}]"
    started = time.monotonic()
    print(f"{label} seed={seed}: behavior cloning started", flush=True)
    set_global_seed(seed)
    encoder, actor, critic = AttentionEncoder(), Actor(), Critic()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(actor.parameters()) +
        list(critic.parameters()), lr=LEARNING_RATE)
    with contextlib.redirect_stdout(io.StringIO()):
        behavior_clone_fodas(encoder, actor)
    print(f"{label} behavior cloning complete; PPO 0/{ppo_episodes}", flush=True)
    report_every = max(1, min(10, ppo_episodes // 5 or 1))
    for episode in range(ppo_episodes):
        with contextlib.redirect_stdout(io.StringIO()):
            set_global_seed(seed + 1000 + episode)
            simulation.run_episode(
                episode + 1, actor, critic, BrokerSelector(), encoder,
                optimizer, train=True)
        completed = episode + 1
        if completed == ppo_episodes or completed % report_every == 0:
            print(f"{label} PPO {completed}/{ppo_episodes}", flush=True)
    print(f"{label} training complete in "
          f"{_format_duration(time.monotonic() - started)}", flush=True)
    return encoder, actor


def _safe_mean(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def _mean_tenure_ticks(head_ids: Iterable[int]) -> float:
    ids = [int(v) for v in head_ids if int(v) >= 0]
    if not ids:
        return float("nan")
    lengths: List[int] = []
    run = 1
    for previous, current in zip(ids, ids[1:]):
        if current == previous:
            run += 1
        else:
            lengths.append(run)
            run = 1
    lengths.append(run)
    return float(np.mean(lengths))


def _summarize(run: Dict) -> Dict[str, float]:
    rec, series, energy = run["rec"], run["series"], run["E"]
    delays = np.asarray(rec["delay_ms"], dtype=float)
    successes = np.asarray(rec["success"], dtype=float)
    n_tasks = len(delays)
    delivered = max(int(successes.sum()), 1)
    n_ticks = max(len(series["t"]), 1)
    reasons = series["switch_block_reason"]

    summary = {
        "n_tasks": n_tasks,
        "duration_s": float(run["dur"]),
        "mean_delay_ms": float(np.mean(delays)) if n_tasks else float("nan"),
        "p50_delay_ms": float(np.percentile(delays, 50)) if n_tasks else float("nan"),
        "p95_delay_ms": float(np.percentile(delays, 95)) if n_tasks else float("nan"),
        "p99_delay_ms": float(np.percentile(delays, 99)) if n_tasks else float("nan"),
        "success_rate": float(np.mean(successes)) if n_tasks else 0.0,
        "deadline_miss_rate": 1.0 - float(np.mean(successes)) if n_tasks else 1.0,
        "mean_control_delay_ms": _safe_mean(rec["control_ms"]),
        "mean_selector_delay_ms": _safe_mean(rec["selector_ms"]),
        "mean_handover_delay_ms": _safe_mean(rec["handover_ms"]),
        "control_energy_mj_per_task":
            1000.0 * (energy["control"] + energy["signal"]) / max(n_tasks, 1),
        "nonprop_energy_j_per_delivered_task":
            (energy["comp"] + energy["net"] + energy["control"] +
             energy["signal"]) / delivered,
        "whole_swarm_energy_j":
            energy["prop"] + energy["comp"] + energy["net"] +
            energy["control"] + energy["signal"],
        "head_changes": float(run["head_changes"]),
        "head_changes_per_1000_tasks":
            1000.0 * run["head_changes"] / max(n_tasks, 1),
        "mean_head_tenure_ticks": _mean_tenure_ticks(series["head_id"]),
        "selector_run_pct": 100.0 * sum(series["selector_ran"]) / n_ticks,
        "mean_selector_runtime_us": _safe_mean(series["selector_us"]),
        "mean_stage1_ms_per_tick": _safe_mean(series["selector_us"]) / 1000.0,
        "stage1_cpu_share_pct": (_safe_mean(series["selector_us"]) / 1000.0
                                 / (CONTROL_TICK_S * 1000.0) * 100.0),
        "mean_head_soc_pct": 100.0 * _safe_mean(series["head_soc"]),
        "mean_head_iot_distance_m": _safe_mean(series["head_iot_distance_m"]),
        "mean_head_fog_distance_m": _safe_mean(series["head_fog_distance_m"]),
        "mean_head_workload_cycles": _safe_mean(series["head_workload_cycles"]),
        "mean_head_control_service_ms": _safe_mean(
            series["head_control_service_ms"]),
        "kl_gate_pct": 100.0 * sum(r == "kl_gate" for r in reasons) / n_ticks,
        "tenure_block_pct":
            100.0 * sum(r == "minimum_tenure" for r in reasons) / n_ticks,
        "switch_cost_block_pct":
            100.0 * sum(r == "switch_cost" for r in reasons) / n_ticks,
        "head_recovery_ticks": float(run["head_recovery_ticks"]),
        "tasks_dropped_in_gap": int(run["tasks_dropped_in_gap"]),
        "delay_spike_ms": float(run["delay_spike_ms"]),
        "control_jobs_lost": int(run["control_jobs_lost"]),
    }
    for index in range(N_CRITERIA):
        summary[f"decision_power_{index}_pct"] = 100.0 * _safe_mean(
            series[f"decision_power_{index}"])
    return summary


def _detail_writers(enabled: bool, append: bool = False):
    if not enabled:
        return None
    os.makedirs(OUT_DIR, exist_ok=True)
    task_path = os.path.join(OUT_DIR, "fog_head_task_results.csv.gz")
    tick_path = os.path.join(OUT_DIR, "fog_head_tick_results.csv.gz")
    task_has_header = append and os.path.exists(task_path) and os.path.getsize(task_path) > 0
    tick_has_header = append and os.path.exists(tick_path) and os.path.getsize(tick_path) > 0
    mode = "at" if append else "wt"
    task_fh = gzip.open(task_path, mode, newline="")
    tick_fh = gzip.open(tick_path, mode, newline="")
    return {"task_fh": task_fh, "tick_fh": tick_fh,
            "task_writer": None, "tick_writer": None,
            "task_needs_header": not task_has_header,
            "tick_needs_header": not tick_has_header}


def _write_details(writers, metadata: Dict, run: Dict) -> None:
    if writers is None:
        return
    for kind, values in (("task", run["rec"]), ("tick", run["series"])):
        keys = list(values)
        writer_key = f"{kind}_writer"
        if writers[writer_key] is None:
            fieldnames = list(metadata) + keys
            writers[writer_key] = csv.DictWriter(
                writers[f"{kind}_fh"], fieldnames=fieldnames)
            if writers[f"{kind}_needs_header"]:
                writers[writer_key].writeheader()
                writers[f"{kind}_needs_header"] = False
        writer = writers[writer_key]
        count = len(values[keys[0]]) if keys else 0
        for index in range(count):
            writer.writerow({**metadata, **{key: values[key][index] for key in keys}})


def _close_detail_writers(writers) -> None:
    if writers is not None:
        writers["task_fh"].close()
        writers["tick_fh"].close()


def _flush_detail_writers(writers) -> None:
    if writers is not None:
        writers["task_fh"].flush()
        writers["tick_fh"].flush()


def _representative_payload(representatives: Dict[str, Dict]) -> Dict[str, Dict]:
    return {
        policy: {
            "rec": {"delay_ms": run["rec"]["delay_ms"]},
            "series": {"t": run["series"]["t"],
                       "head_id": run["series"]["head_id"],
                       "head_soc": run["series"]["head_soc"]},
        }
        for policy, run in representatives.items()
    }


def _write_json_atomic(path: str, payload: Dict) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w") as fh:
        json.dump(payload, fh, indent=2, allow_nan=True)
    os.replace(temp_path, path)


def _write_progress(metadata: Dict, representatives: Dict[str, Dict]) -> None:
    _write_json_atomic(os.path.join(OUT_DIR, "fog_head_progress.json"), metadata)
    if representatives:
        _write_json_atomic(
            os.path.join(OUT_DIR, "fog_head_representatives.json"),
            _representative_payload(representatives))


def run_sweep(arrival_rates: Iterable[float], fog_sizes: Iterable[int],
              scale_arrival_rate: float, episode_seconds: float,
              trials: int, seed0: int, ppo_episodes: int,
              handover_ms: float, save_raw: bool = True,
              start_trial: int = 0, existing_rows: List[Dict] | None = None,
              existing_representatives: Dict[str, Dict] | None = None,
              checkpoint_metadata: Dict | None = None,
              pause_after_trial: int | None = None,
              controller_placement: str = "head",
              dispatch: str = "rl",
              head_failure_s: float | None = None,
              ) -> Tuple[List[Dict], Dict[str, Dict], bool]:
    rates = sorted(float(v) for v in arrival_rates)
    sizes = sorted(int(v) for v in fog_sizes)
    conditions = [
        ("fog_size", n_fog, 10 * n_fog,
         scale_arrival_rate * n_fog / N_FOG, None)
        for n_fog in sizes
    ] + [
        ("arrival_rate", N_FOG, N_IOT, arrival_rate, None)
        for arrival_rate in rates
    ]
    if head_failure_s is not None:
        conditions.append(("head_failure", N_FOG, N_IOT,
                           scale_arrival_rate, head_failure_s))
    rows = list(existing_rows or [])
    representatives = dict(existing_representatives or {})
    writers = _detail_writers(save_raw, append=start_trial > 0)
    run_started = time.monotonic()
    paused = False
    conditions_per_trial = len(conditions) * len(POLICIES)
    try:
        for trial in range(start_trial, trials):
            trial_number = trial + 1
            seed = seed0 + trial
            print(f"\n[TRIAL {trial_number:02d}/{trials:02d}] START seed={seed}; "
                  f"{conditions_per_trial} evaluation conditions", flush=True)
            nets = (_shared_nets(seed, ppo_episodes, trial_number, trials)
                    if dispatch == "rl" else None)
            condition = 0
            for (study_axis, n_fog, n_iot, arrival_rate,
                 failure_at_s) in conditions:
                for policy in POLICIES:
                    condition += 1
                    print(
                        f"[EVAL  {trial_number:02d}/{trials:02d}] "
                        f"condition {condition:02d}/{conditions_per_trial:02d}: "
                        f"{policy}, axis={study_axis}, N={n_fog}, "
                        f"lambda={arrival_rate:g}", flush=True)
                    result = run_exp(
                        f"{policy}/N={n_fog}/lambda={arrival_rate:g}/seed={seed}",
                        move_mode="event_driven", head_mode=policy,
                        dispatch=dispatch, coordinated=True, nets=nets,
                        seed=seed, task_arrival_rate=arrival_rate,
                        episode_duration_s=episode_seconds,
                        handover_setup_ms=handover_ms,
                        controller_placement=controller_placement,
                        n_fog=n_fog, n_iot=n_iot,
                        head_failure_s=failure_at_s)
                    metadata = {
                        "study_axis": study_axis,
                        "n_fog": n_fog,
                        "n_iot": n_iot,
                        "arrival_rate": arrival_rate,
                        "arrival_rate_per_fog": arrival_rate / n_fog,
                        "episode_seconds": episode_seconds,
                        "trial": trial,
                        "seed": seed,
                        "policy": policy,
                        "head_failure_s": failure_at_s,
                    }
                    rows.append({**metadata, **_summarize(result)})
                    _write_details(writers, metadata, result)
                    if (trial == 0 and study_axis == "arrival_rate"
                            and arrival_rate == rates[-1]):
                        representatives[policy] = result

            _flush_detail_writers(writers)
            completed_trials = trial_number
            elapsed = time.monotonic() - run_started
            completed_this_run = completed_trials - start_trial
            seconds_per_trial = elapsed / max(completed_this_run, 1)
            eta = seconds_per_trial * (trials - completed_trials)
            status = "in_progress" if completed_trials < trials else "complete"
            current_metadata = {
                **(checkpoint_metadata or {}),
                "status": status,
                "completed_trials": completed_trials,
                "requested_trials": trials,
                "last_completed_seed": seed,
                "elapsed_this_run_s": elapsed,
                "estimated_remaining_s": eta,
            }
            _write_summary(rows, current_metadata)
            _write_progress(current_metadata, representatives)
            print(
                f"[TRIAL {trial_number:02d}/{trials:02d}] COMPLETE | "
                f"elapsed this run {_format_duration(elapsed)} | "
                f"ETA {_format_duration(eta)} | checkpoint saved",
                flush=True)

            requested_pause = (_STOP_AFTER_TRIAL or
                               (pause_after_trial is not None and
                                completed_trials >= pause_after_trial))
            if requested_pause and completed_trials < trials:
                paused = True
                current_metadata["status"] = "paused"
                _write_summary(rows, current_metadata)
                _write_progress(current_metadata, representatives)
                print(f"[PAUSED] Safely stopped after trial {completed_trials}/{trials}. "
                      f"Resume with --resume.", flush=True)
                break
    finally:
        _close_detail_writers(writers)
    return rows, representatives, paused


def _values(rows: List[Dict], metric: str, study_axis: str,
            x_field: str, x_value: float, policy: str) -> List[float]:
    return [float(r[metric]) for r in rows
            if r["study_axis"] == study_axis
            and float(r[x_field]) == float(x_value)
            and r["policy"] == policy]


def _bootstrap_mean_ci(values: Iterable[float], samples: int,
                       rng: np.random.Generator) -> Tuple[float, float, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size == 1 or samples <= 0:
        return mean, mean, mean
    draws = rng.choice(vals, size=(samples, vals.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return mean, float(low), float(high)


def _save(fig, name: str, legend_space: bool = False) -> None:
    if legend_space:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    else:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_metric(ax, rows: List[Dict], xs: List[float], metric: str,
                 ylabel: str, bootstrap_samples: int, *,
                 study_axis: str, x_field: str, xlabel: str,
                 percent: bool = False) -> None:
    rng = np.random.default_rng(20260720)
    scale = 100.0 if percent else 1.0
    for policy in POLICIES:
        stats = [_bootstrap_mean_ci(
            [scale * v for v in _values(
                rows, metric, study_axis, x_field, x, policy)],
            bootstrap_samples, rng) for x in xs]
        means = np.asarray([s[0] for s in stats])
        lows = np.asarray([s[1] for s in stats])
        highs = np.asarray([s[2] for s in stats])
        width = 2.2 if policy == "spotis" else 1.35
        ax.plot(xs, means, marker=MARKERS[policy], color=COLORS[policy],
                linewidth=width, markersize=5, label=LABELS[policy])
        ax.fill_between(xs, lows, highs, color=COLORS[policy], alpha=0.10)
        for x in xs:
            raw = [scale * v for v in _values(
                rows, metric, study_axis, x_field, x, policy)]
            ax.scatter([x] * len(raw), raw, s=10, color=COLORS[policy],
                       alpha=0.28, linewidths=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def make_figures(rows: List[Dict], representatives: Dict[str, Dict],
                 bootstrap_samples: int) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    sizes = sorted({float(r["n_fog"]) for r in rows
                    if r["study_axis"] == "fog_size"})
    rates = sorted({float(r["arrival_rate"]) for r in rows
                    if r["study_axis"] == "arrival_rate"})
    size_args = {"study_axis": "fog_size", "x_field": "n_fog",
                 "xlabel": "Fog UAVs (N_FOG)"}
    rate_args = {"study_axis": "arrival_rate", "x_field": "arrival_rate",
                 "xlabel": "Task arrival rate (tasks/s)"}

    # Figure 1: headline selector scalability and the real-time boundary.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _plot_metric(ax, rows, sizes, "mean_stage1_ms_per_tick",
                 "Selector cost (ms/control tick)", bootstrap_samples,
                 **size_args)
    ax.axhline(CONTROL_TICK_S * 1000.0, color="#A00000", linestyle="--",
               linewidth=1.0, label="100 ms real-time boundary")
    ax.set_yscale("log")
    # Headroom above the 100 ms line so the legend cannot sit on top of it.
    ax.set_ylim(top=1e3)
    ax.legend(ncol=2, loc="upper left")
    _save(fig, "fh_fig1_selector_scalability")

    # Figure 2: causal system performance as the swarm scales.
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    _plot_metric(axes[0], rows, sizes, "mean_delay_ms", "Mean delay (ms)",
                 bootstrap_samples, **size_args)
    _plot_metric(axes[1], rows, sizes, "p95_delay_ms", "P95 delay (ms)",
                 bootstrap_samples, **size_args)
    _plot_metric(axes[2], rows, sizes, "deadline_miss_rate",
                 "Deadline-miss rate (%)", bootstrap_samples,
                 percent=True, **size_args)
    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(0.02, 0.96, label, transform=ax.transAxes,
                va="top", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.04))
    _save(fig, "fh_fig2_performance_scalability", legend_space=True)

    # Figure 3: fraction of the elected head CPU consumed by Stage 1.
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _plot_metric(ax, rows, sizes, "stage1_cpu_share_pct",
                 "Head CPU consumed by Stage 1 (%)", bootstrap_samples,
                 **size_args)
    ax.axhline(100.0, color="#A00000", linestyle="--", linewidth=1.0)
    # Headroom below zero so the sub-0.1% reading can be labelled in clear
    # space: on a 0-100% axis the proposed curve is one pixel off the floor and
    # would otherwise be misread as exactly zero.
    ax.set_ylim(-11.0, 108.0)
    ax.legend(ncol=1, loc="upper right", fontsize=8)
    # Alternate above/below so the two sub-3% labels cannot collide.
    for policy, dy in (("spotis", -17), ("fu_serve", 9),
                       ("2dp_fhs", 9), ("3d_pos", 9)):
        val = float(np.mean(_values(rows, "stage1_cpu_share_pct", "fog_size",
                                    "n_fog", sizes[-1], policy)))
        ax.annotate(f"{val:.3f}%", xy=(sizes[-1], val), xytext=(0, dy),
                    textcoords="offset points", fontsize=8, ha="right",
                    color=COLORS[policy], fontweight="bold")
    _save(fig, "fh_fig3_control_plane_cpu_share")

    # Figure 4: switching stability at scale.
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _plot_metric(ax, rows, sizes, "head_changes_per_1000_tasks",
                 "Head changes per 1000 tasks", bootstrap_samples,
                 **size_args)
    # The proposed selector is exactly zero and FU-Serve is ~0.18, which are one
    # pixel apart on a linear axis reaching ~83.  symlog keeps the true zero
    # (plain log cannot plot it) while lifting FU-Serve clear of the floor.
    ax.set_yscale("symlog", linthresh=0.05, linscale=0.4)
    ax.set_ylim(0.0, 200.0)
    ax.set_yticks([0.0, 0.1, 1.0, 10.0, 100.0])
    ax.set_yticklabels(["0", "0.1", "1", "10", "100"])
    ax.legend(ncol=1, loc="upper right", fontsize=8)
    for policy, dy in (("spotis", 9), ("fu_serve", 9)):
        val = float(np.mean(_values(rows, "head_changes_per_1000_tasks",
                                    "fog_size", "n_fog", sizes[-1], policy)))
        ax.annotate(f"{val:.3f}", xy=(sizes[-1], val), xytext=(-4, dy),
                    textcoords="offset points", fontsize=8, ha="right",
                    color=COLORS[policy], fontweight="bold")
    _save(fig, "fh_fig4_selection_stability")

    # Figure 5: retain the N=20 load sweep and report the near-tie honestly.
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    _plot_metric(axes[0], rows, rates, "mean_delay_ms", "Mean delay (ms)",
                 bootstrap_samples, **rate_args)
    _plot_metric(axes[1], rows, rates, "p95_delay_ms", "P95 delay (ms)",
                 bootstrap_samples, **rate_args)
    _plot_metric(axes[2], rows, rates, "deadline_miss_rate",
                 "Deadline-miss rate (%)", bootstrap_samples,
                 percent=True, **rate_args)
    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(0.02, 0.96, label, transform=ax.transAxes,
                va="top", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.04))
    _save(fig, "fh_fig5_load_sweep_n20", legend_space=True)

    if representatives:
        # Figure 6: tail-latency distribution for one explicitly identified seed.
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for policy in POLICIES:
            delays = np.sort(np.asarray(
                representatives[policy]["rec"]["delay_ms"], dtype=float))
            cdf = np.arange(1, len(delays) + 1) / max(len(delays), 1)
            ax.plot(delays, cdf, color=COLORS[policy],
                    linewidth=2.0 if policy == "spotis" else 1.35,
                    label=LABELS[policy])
        ax.set_xlabel("End-to-end task delay (ms)")
        ax.set_ylabel("Empirical CDF")
        ax.set_xlim(left=0)
        ax.legend(loc="lower right")
        _save(fig, "fh_fig6_delay_ecdf_representative")

        # Figure 7: selected-head trace exposes switching/churn directly.
        fig, axes = plt.subplots(len(POLICIES), 1, figsize=(9.0, 6.2), sharex=True)
        for ax, policy in zip(axes, POLICIES):
            series = representatives[policy]["series"]
            ax.step(series["t"], series["head_id"], where="post",
                    color=COLORS[policy], linewidth=1.2)
            soc_ax = ax.twinx()
            soc_ax.plot(series["t"], 100.0 * np.asarray(series["head_soc"]),
                        color="#777777", linewidth=0.8, linestyle="--")
            soc_ax.set_ylim(0, 100)
            soc_ax.set_ylabel("SOC %", color="#777777", fontsize=7)
            soc_ax.tick_params(axis="y", colors="#777777", labelsize=7)
            ax.set_ylabel(LABELS[policy], rotation=0, ha="right", va="center")
            ax.set_yticks(sorted(set(int(v) for v in series["head_id"])))
        axes[-1].set_xlabel("Simulation time (s)")
        _save(fig, "fh_fig7_head_timeline_representative")

    failure_rows = [row for row in rows
                    if row["study_axis"] == "head_failure"]
    if failure_rows:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
        for ax, metric, ylabel in (
            (axes[0], "head_recovery_ticks", "Recovery time (control ticks)"),
            (axes[1], "tasks_dropped_in_gap", "Tasks dropped before recovery"),
        ):
            positions = np.arange(len(POLICIES))
            samples = [[float(row[metric]) for row in failure_rows
                        if row["policy"] == policy] for policy in POLICIES]
            ax.bar(positions, [_safe_mean(values) for values in samples],
                   color=[COLORS[policy] for policy in POLICIES], alpha=0.8)
            for position, values in zip(positions, samples):
                ax.scatter([position] * len(values), values, s=15,
                           color="#111111", alpha=0.45, zorder=3)
            ax.set_xticks(positions, [LABELS[policy] for policy in POLICIES],
                          rotation=15, ha="right")
            ax.set_ylabel(ylabel)
        _save(fig, "fh_fig8_failure_recovery")


def _write_summary(rows: List[Dict], metadata: Dict) -> None:
    if not rows:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "fog_head_results.csv")
    csv_temp = f"{csv_path}.tmp"
    with open(csv_temp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_temp, csv_path)
    _write_json_atomic(
        os.path.join(OUT_DIR, "fog_head_results.json"),
        {"metadata": metadata, "rows": rows})


def _load_resume_state(expected: Dict) -> Tuple[int, List[Dict], Dict[str, Dict]]:
    summary_path = os.path.join(OUT_DIR, "fog_head_results.json")
    representatives_path = os.path.join(OUT_DIR, "fog_head_representatives.json")
    if not os.path.exists(summary_path):
        raise ValueError(f"--resume requested but checkpoint is missing: {summary_path}")
    with open(summary_path) as fh:
        payload = json.load(fh)
    metadata = payload.get("metadata", {})
    rows = payload.get("rows", [])
    for key in ("arrival_rates_tasks_per_s", "fog_sizes", "scale_arrival_rate",
                "episode_seconds", "trials",
                "seed0", "ppo_episodes", "handover_setup_ms",
                "raw_details_enabled", "controller_placement", "dispatch",
                "head_failure_s"):
        if metadata.get(key) != expected.get(key):
            raise ValueError(
                f"resume configuration mismatch for {key}: checkpoint has "
                f"{metadata.get(key)!r}, command requests {expected.get(key)!r}")
    if rows and not {"mean_head_control_service_ms", "head_recovery_ticks"} <= rows[0].keys():
        raise ValueError("checkpoint predates the fog-head dynamics metrics")
    completed = int(metadata.get("completed_trials", 0))
    if completed > 0 and expected["raw_details_enabled"]:
        for filename in ("fog_head_task_results.csv.gz",
                         "fog_head_tick_results.csv.gz"):
            if not os.path.exists(os.path.join(OUT_DIR, filename)):
                raise ValueError(f"resume checkpoint is missing raw file {filename}")
    expected_rows = completed * (
        len(expected["arrival_rates_tasks_per_s"]) + len(expected["fog_sizes"])
        + int(expected["head_failure_s"] is not None)
    ) * len(POLICIES)
    if len(rows) != expected_rows:
        raise ValueError(
            f"checkpoint row count is inconsistent: expected {expected_rows}, "
            f"found {len(rows)}")
    representatives: Dict[str, Dict] = {}
    if os.path.exists(representatives_path):
        with open(representatives_path) as fh:
            representatives = json.load(fh)
    if completed > 0 and set(representatives) != set(POLICIES):
        raise ValueError("resume checkpoint is missing representative-run data")
    return completed, rows, representatives


def _request_pause_after_trial(_signum, _frame) -> None:
    global _STOP_AFTER_TRIAL
    if not _STOP_AFTER_TRIAL:
        _STOP_AFTER_TRIAL = True
        print("\n[PAUSE REQUESTED] Current trial will finish, checkpoint, and stop. "
              "Pressing Ctrl-C again is unnecessary.", flush=True)
    else:
        print("\n[PAUSE REQUESTED] Already waiting for the current trial to finish.",
              flush=True)


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrival-rates", default="60,120,180,240,300",
                        help="comma-separated Poisson arrival rates in tasks/s")
    parser.add_argument("--fog-sizes", default="10,20,40,80",
                        help="comma-separated fog-swarm sizes; N_IoT is 10*N_FOG")
    parser.add_argument("--scale-arrival-rate", type=float, default=180.0,
                        help="total tasks/s at N_FOG=20 for the size sweep")
    parser.add_argument("--episode-seconds", type=float, default=10.0,
                        help="fixed simulated horizon for every condition")
    parser.add_argument("--head-failure-s", type=float,
                        help="add a separate N=20 induced-head-failure condition")
    parser.add_argument("--trials", type=int, default=20,
                        help="paired seeds; 20+ recommended for paper results")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ppo-episodes", type=int, default=50)
    parser.add_argument("--handover-ms", type=float, default=HANDOVER_SETUP_MS)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--controller-placement", default="head",
                        choices=("head", "equal_controller"))
    parser.add_argument("--dispatch", default="rl",
                        choices=("rl", "head_fodas"),
                        help="head_fodas runs the deterministic Stage-1 isolation ablation")
    parser.add_argument("--output-dir", default=OUT_DIR,
                        help="directory for summaries, raw data, and figures")
    parser.add_argument("--no-raw", action="store_true",
                        help="do not write compressed per-task/per-tick CSV files")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the last safely completed trial in output-dir")
    parser.add_argument("--pause-after-trial", type=int,
                        help="safely checkpoint and stop after this total trial number")
    args = parser.parse_args()
    OUT_DIR = os.path.abspath(args.output_dir)

    rates = [float(v) for v in args.arrival_rates.split(",")]
    fog_sizes = [int(v) for v in args.fog_sizes.split(",")]
    if (not rates or min(rates) <= 0.0 or not fog_sizes or min(fog_sizes) < 2
            or args.scale_arrival_rate <= 0.0 or args.episode_seconds <= 0.0 or
            args.trials <= 0 or args.ppo_episodes < 0 or
            args.handover_ms < 0.0 or args.bootstrap_samples < 0):
        parser.error("rates, horizon, and trials must be positive; other values non-negative")
    if (args.pause_after_trial is not None and
            not 1 <= args.pause_after_trial <= args.trials):
        parser.error("--pause-after-trial must be between 1 and --trials")
    if (args.head_failure_s is not None and
            not 0 <= args.head_failure_s < args.episode_seconds):
        parser.error("--head-failure-s must be within the episode")

    metadata = {
        "design": "paired fog-size primary sweep plus N=20 arrival-rate sweep",
        "arrival_rates_tasks_per_s": rates,
        "fog_sizes": fog_sizes,
        "iot_per_fog": 10,
        "scale_arrival_rate": args.scale_arrival_rate,
        "episode_seconds": args.episode_seconds,
        "trials": args.trials,
        "seed0": args.seed,
        "ppo_episodes": args.ppo_episodes,
        "handover_setup_ms": args.handover_ms,
        "controller_placement": args.controller_placement,
        "dispatch": args.dispatch,
        "head_failure_s": args.head_failure_s,
        "head_reevaluation_ticks": HEAD_REEVALUATION_TICKS,
        "head_min_tenure_ticks": HEAD_MIN_TENURE_TICKS,
        "payload_path": "direct IoT-to-executor; elected head carries control messages only",
        "raw_details_enabled": not args.no_raw,
        "bootstrap_samples": args.bootstrap_samples,
        "confidence_interval": "paired/nonparametric bootstrap, 95%",
    }

    start_trial = 0
    existing_rows: List[Dict] = []
    existing_representatives: Dict[str, Dict] = {}
    if args.resume:
        try:
            start_trial, existing_rows, existing_representatives = _load_resume_state(metadata)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"[RESUME] {start_trial}/{args.trials} trials already complete; "
              f"continuing with trial {start_trial + 1}.", flush=True)
        if (args.pause_after_trial is not None and
                args.pause_after_trial <= start_trial):
            parser.error("--pause-after-trial must be greater than the number "
                         "of trials already completed")

    if start_trial >= args.trials:
        print(f"[COMPLETE] All {args.trials}/{args.trials} trials are already saved.",
              flush=True)
        make_figures(existing_rows, existing_representatives,
                     args.bootstrap_samples)
        return

    global _STOP_AFTER_TRIAL
    _STOP_AFTER_TRIAL = False
    previous_sigint = signal.signal(signal.SIGINT, _request_pause_after_trial)
    try:
        rows, representatives, paused = run_sweep(
            rates, fog_sizes, args.scale_arrival_rate,
            args.episode_seconds, args.trials, args.seed,
            args.ppo_episodes, args.handover_ms, save_raw=not args.no_raw,
            start_trial=start_trial, existing_rows=existing_rows,
            existing_representatives=existing_representatives,
            checkpoint_metadata=metadata,
            pause_after_trial=args.pause_after_trial,
            controller_placement=args.controller_placement,
            dispatch=args.dispatch, head_failure_s=args.head_failure_s)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    if paused:
        print(f"Progress file: {os.path.join(OUT_DIR, 'fog_head_progress.json')}")
        return
    if args.trials < 10:
        print("[WARN] Fewer than 10 paired trials: outputs are for smoke/diagnostic use, not publication.")
    make_figures(rows, representatives, args.bootstrap_samples)
    print(f"[COMPLETE] Fog-head results and journal figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
