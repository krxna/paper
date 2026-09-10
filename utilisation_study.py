"""Paired six-load system-utilisation study and replacement Figure 7.

The default controller costs are declared deterministic simulation parameters,
not hardware measurements. Each policy/load/seed cell is independently signed
and checkpointed, and optional service-cost scaling supports sensitivity runs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "uav_fog_matplotlib"))
os.environ.setdefault(
    "XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "uav_fog_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from controller_profile import (
    DEFAULT_PROFILE_PATH, load_profile, require_profile, scale_profile)
from execution_engine import EXECUTION_MODEL_VERSION
from neural import Actor, AttentionEncoder
from scenario import (
    DEFAULT_EVAL_SEEDS, LOAD_TARGETS, ScenarioConfig, make_policy, run_scenario,
)


POLICIES = ("Attention+PPO", "random", "FODAS", "ReLIEF")
PLACEMENTS = ("native", "equal_controller")
COLORS = {
    "Attention+PPO": "#0072B2",
    "random": "#999999",
    "FODAS": "#D55E00",
    "ReLIEF": "#009E73",
}
MARKERS = {"Attention+PPO": "o", "random": "s", "FODAS": "^", "ReLIEF": "D"}
LINESTYLES = {policy: "-" for policy in POLICIES}
CATEGORY_COLORS = {
    "productive": "#009E73",
    "wasted": "#D55E00",
    "control": "#CC79A7",
    "idle": "#D9D9D9",
}
CATEGORY_HATCHES = {
    "productive": "", "wasted": "///", "control": "\\\\\\", "idle": ".."}


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_pickle_gz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    os.close(fd)
    try:
        with gzip.open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _code_hash() -> str:
    digest = hashlib.sha256()
    for name in (
        "execution_engine.py", "scenario.py", "controller_profile.py",
        "utilisation_study.py", "physics.py", "neural.py",
        "baselines/fodas_baseline.py", "baselines/relief_baseline.py",
    ):
        digest.update(name.encode())
        digest.update(Path(name).read_bytes())
    return digest.hexdigest()


def experiment_signature(
    profile: Mapping, loads: Sequence[float], seeds: Sequence[int],
    warmup_s: float, measurement_s: float,
    policy_checkpoint_hash: str = "unspecified",
) -> Dict:
    return {
        "execution_model_version": EXECUTION_MODEL_VERSION,
        "controller_profile_hash": profile["profile_hash"],
        "controller_workload_hash": profile["workload_hash"],
        "loads": list(loads),
        "seeds": list(seeds),
        "warmup_s": warmup_s,
        "measurement_s": measurement_s,
        "policy_checkpoint_hash": policy_checkpoint_hash,
        "policies": list(POLICIES),
        "placements": list(PLACEMENTS),
        "code_hash": _code_hash(),
    }


def _cell_name(placement: str, policy: str, load: float, seed: int) -> str:
    safe_policy = policy.lower().replace("+", "_").replace(" ", "_")
    return f"{placement}__{safe_policy}__load-{load:.2f}__seed-{seed}.pkl.gz"


def _load_nets(checkpoint: Path) -> Tuple[AttentionEncoder, Actor]:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"retrained event-model checkpoint is missing: {checkpoint}")
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(checkpoint, map_location="cpu")
    if saved.get("execution_model_version") != EXECUTION_MODEL_VERSION:
        raise ValueError("policy checkpoint was not trained with this execution model")
    encoder, actor = AttentionEncoder(), Actor()
    encoder.load_state_dict(saved["encoder"])
    actor.load_state_dict(saved["actor"])
    encoder.eval()
    actor.eval()
    return encoder, actor


def recompute_capacity_from_audit(
    audit: Sequence[Mapping], frequencies: Mapping[int, float],
    start_s: float, end_s: float,
) -> Dict[int, Dict[str, float]]:
    duration = end_s - start_s
    result = {
        int(node): {
            "productive_cycles": 0.0, "wasted_cycles": 0.0,
            "control_cycles": 0.0,
            "provisioned_cycles": float(freq) * duration,
        }
        for node, freq in frequencies.items()
    }
    for row in audit:
        if row.get("record_type") != "service_slice":
            continue
        start = float(row["start_s"])
        end = float(row["end_s"])
        if end <= start:
            continue
        overlap = max(0.0, min(end_s, end) - max(start_s, start))
        if overlap <= 0:
            continue
        cycles = float(row["cycles"]) * overlap / (end - start)
        key = {
            "productive": "productive_cycles",
            "wasted": "wasted_cycles",
            "control": "control_cycles",
        }[row["fate"]]
        result[int(row["node_id"])][key] += cycles
    for values in result.values():
        busy = (
            values["productive_cycles"] + values["wasted_cycles"]
            + values["control_cycles"])
        values["idle_cycles"] = values["provisioned_cycles"] - busy
    return result


def _assert_audit_agreement(cell: Mapping) -> None:
    recomputed = recompute_capacity_from_audit(
        cell["audit"], cell["frequencies"],
        cell["measurement_start_s"], cell["measurement_end_s"])
    for node, expected in cell["per_node"].items():
        actual = recomputed[int(node)]
        for key in (
            "productive_cycles", "wasted_cycles", "control_cycles", "idle_cycles",
        ):
            if not math.isclose(
                float(expected[key]), float(actual[key]),
                rel_tol=2e-9, abs_tol=1e-4,
            ):
                raise AssertionError(
                    f"audit mismatch node={node} category={key}: "
                    f"{expected[key]} != {actual[key]}")


def run_study(
    *, profile: Mapping, nets: Tuple[AttentionEncoder, Actor],
    out_dir: Path, loads: Sequence[float] = LOAD_TARGETS,
    seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    placements: Sequence[str] = PLACEMENTS,
    warmup_s: float = 10.0, measurement_s: float = 120.0,
    relief_pretrain: bool = True,
    policy_checkpoint_hash: str = "unspecified",
    policies: Sequence[str] = POLICIES,
    base_checkpoint_dir: Path | None = None,
) -> List[Dict]:
    signature = experiment_signature(
        profile, loads, seeds, warmup_s, measurement_s,
        policy_checkpoint_hash)
    policies = tuple(policies)
    if not policies or any(policy not in POLICIES for policy in policies):
        raise ValueError("invalid policy selection")
    base_cells, base_signature = (
        load_checkpoint_cells(base_checkpoint_dir)
        if base_checkpoint_dir is not None else ([], {}))
    if base_checkpoint_dir is not None:
        for key in (
            "execution_model_version", "controller_profile_hash",
            "controller_workload_hash", "loads", "seeds", "warmup_s",
            "measurement_s", "policy_checkpoint_hash", "placements",
        ):
            if base_signature.get(key) != signature.get(key):
                raise ValueError(f"base checkpoint signature mismatch: {key}")
        expected = len(placements) * len(POLICIES) * len(loads) * len(seeds)
        if len(base_cells) != expected:
            raise ValueError(
                f"base checkpoint grid is incomplete: {len(base_cells)}/{expected}")
    base_hash = base_signature.get("code_hash")
    signature["rerun_policies"] = list(policies)
    signature["policy_sources"] = {
        policy: {
            "code_hash": signature["code_hash"] if policy in policies else base_hash,
            "source": "corrected" if policy in policies else "base",
        }
        for policy in POLICIES
    }
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sig_path = checkpoint_dir / "signature.json"
    if sig_path.exists():
        previous = json.loads(sig_path.read_text())
        if previous != signature:
            raise ValueError(
                "checkpoint signature mismatch; use a new output directory")
    else:
        _atomic_json(sig_path, signature)

    def cell_key(cell):
        row = cell["summary"]
        return (
            row["controller_placement"], row["policy"],
            float(row["target_load"]), int(row["seed"]))

    merged = {cell_key(cell): cell for cell in base_cells}
    if len(merged) != len(base_cells):
        raise ValueError("duplicate cells in base checkpoint grid")
    total = len(placements) * len(policies) * len(loads) * len(seeds)
    index = 0
    run_started = time.monotonic()
    completed_durations = []
    for placement in placements:
        for load in loads:
            for seed in seeds:
                for policy in policies:
                    index += 1
                    cell_started = time.monotonic()
                    path = checkpoint_dir / _cell_name(
                        placement, policy, load, seed)
                    was_cached = path.exists()
                    if was_cached:
                        with gzip.open(path, "rb") as fh:
                            payload = pickle.load(fh)
                        if payload.get("signature") != signature:
                            raise ValueError(f"cell signature mismatch: {path}")
                        cell = payload["cell"]
                    else:
                        adapter = make_policy(
                            policy, seed, nets if policy == "Attention+PPO" else None,
                            relief_pretrain=relief_pretrain)
                        cfg = ScenarioConfig(
                            seed=seed, target_load=load, policy=policy,
                            controller_placement=placement,
                            warmup_s=warmup_s, measurement_s=measurement_s)
                        result = run_scenario(cfg, adapter, profile)
                        per_node = result.capacity.per_node
                        frequencies = {
                            node: values["provisioned_cycles"] / measurement_s
                            for node, values in per_node.items()}
                        cell = {
                            "summary": result.summary_row(),
                            "per_node": per_node,
                            "external_control_cycles":
                                result.external_control_cycles,
                            "task_records": result.task_records,
                            "audit": result.audit,
                            "frequencies": frequencies,
                            "measurement_start_s": cfg.measurement_start_s,
                            "measurement_end_s": cfg.measurement_end_s,
                        }
                        _assert_audit_agreement(cell)
                        _atomic_pickle_gz(
                            path, {"signature": signature, "cell": cell})
                    fingerprint = cell["summary"]["input_fingerprint"]
                    siblings = [
                        candidate["summary"]["input_fingerprint"]
                        for key, candidate in merged.items()
                        if key[0] == placement and key[2] == float(load)
                        and key[3] == int(seed) and key[1] != policy]
                    if siblings and any(value != fingerprint for value in siblings):
                        raise AssertionError(
                            f"paired input stream drift at load={load}, seed={seed}")
                    merged[(placement, policy, float(load), int(seed))] = cell
                    cell_duration = time.monotonic() - cell_started
                    if not was_cached:
                        completed_durations.append(cell_duration)
                    eta = ((sum(completed_durations) / len(completed_durations)
                            * (total - index))
                           if completed_durations else None)
                    _atomic_json(out_dir / "fig7_progress.json", {
                        "status": "complete" if index == total else "in_progress",
                        "phase": "utilisation_study",
                        "completed_cells": index,
                        "total_cells": total,
                        "percent": 100.0 * index / total,
                        "current_policy": policy,
                        "current_load": load,
                        "current_seed": seed,
                        "current_placement": placement,
                        "elapsed_this_run_s": time.monotonic() - run_started,
                        "estimated_remaining_s": eta,
                        "last_cell_duration_s": cell_duration,
                    })
                    print(
                        f"[{index:03d}/{total:03d}] {placement} {policy} "
                        f"load={load:.2f} seed={seed} checkpointed "
                        f"| ETA={eta/3600:.2f}h" if eta is not None else
                        f"[{index:03d}/{total:03d}] cached",
                        flush=True)
    expected = len(placements) * len(POLICIES) * len(loads) * len(seeds)
    if len(merged) != expected:
        raise ValueError(f"merged checkpoint grid is incomplete: {len(merged)}/{expected}")
    cells = [
        merged[(placement, policy, float(load), int(seed))]
        for placement in placements for load in loads for seed in seeds
        for policy in POLICIES
    ]
    export_outputs(cells, out_dir, signature, profile)
    return cells


def load_checkpoint_cells(checkpoint_dir: Path) -> Tuple[List[Dict], Dict]:
    paths = sorted(checkpoint_dir.glob("*.pkl.gz"))
    cells, signature = [], None
    for path in paths:
        with gzip.open(path, "rb") as fh:
            payload = pickle.load(fh)
        if signature is None:
            signature = payload["signature"]
        elif payload["signature"] != signature:
            continue
        cells.append(payload["cell"])
    return cells, signature or {}


def export_partial_outputs(cells: Sequence[Mapping], out_dir: Path) -> None:
    if not cells:
        return
    summaries = [dict(c["summary"]) for c in cells]
    per_uav = []
    for cell in cells:
        meta = {key: cell["summary"][key] for key in (
            "policy", "controller_placement", "seed", "target_load",
            "realized_load")}
        for node, values in cell["per_node"].items():
            per_uav.append({"node_id": node, **meta, **values})
    _write_csv(out_dir / "fig7_partial_seed_load_policy.csv", summaries)
    _write_csv(out_dir / "fig7_partial_per_uav.csv", per_uav)
    # Render only when every policy has at least one completed cell.
    if set(POLICIES).issubset({r["policy"] for r in summaries}):
        render_figure7(summaries, per_uav, out_dir)


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as fh:
        normalized = []
        for row in rows:
            normalized.append({
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple)) else value)
                for key, value in row.items()
            })
        writer = csv.DictWriter(fh, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


def _t_ci(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    try:
        from scipy.stats import t
        critical = float(t.ppf(0.975, arr.size - 1))
    except Exception:
        # Exact two-sided 95% critical value for the locked n=10 design.
        critical = 2.262157 if arr.size == 10 else 1.96
    half = critical * float(np.std(arr, ddof=1)) / math.sqrt(arr.size)
    return mean, half


def paired_statistics(summary_rows: Sequence[Mapping]) -> List[Dict]:
    output = []
    metrics = ("productive", "wasted", "backlog_exposure")
    for placement in PLACEMENTS:
        for load in sorted({float(r["target_load"]) for r in summary_rows}):
            subset = [
                r for r in summary_rows
                if r["controller_placement"] == placement
                and math.isclose(float(r["target_load"]), load)]
            proposed = {
                int(r["seed"]): r for r in subset
                if r["policy"] == "Attention+PPO"}
            for baseline in ("random", "FODAS", "ReLIEF"):
                other = {
                    int(r["seed"]): r for r in subset
                    if r["policy"] == baseline}
                paired = sorted(set(proposed) & set(other))
                for metric in metrics:
                    diffs = [
                        float(proposed[s][metric]) - float(other[s][metric])
                        for s in paired]
                    mean, half = _t_ci(diffs)
                    output.append({
                        "controller_placement": placement,
                        "target_load": load,
                        "proposed": "Attention+PPO",
                        "baseline": baseline,
                        "metric": metric,
                        "n_pairs": len(diffs),
                        "mean_paired_difference": mean,
                        "ci95_low": mean - half,
                        "ci95_high": mean + half,
                    })
    return output


def relief_random_differences(summary_rows: Sequence[Mapping]) -> List[Dict]:
    output = []
    for placement in PLACEMENTS:
        for load in sorted({float(r["target_load"]) for r in summary_rows}):
            subset = [
                r for r in summary_rows
                if r["controller_placement"] == placement
                and math.isclose(float(r["target_load"]), load)]
            by_policy = {
                policy: {int(r["seed"]): r for r in subset if r["policy"] == policy}
                for policy in ("random", "ReLIEF")
            }
            paired = sorted(set(by_policy["random"]) & set(by_policy["ReLIEF"]))
            diffs = [
                100.0 * (float(by_policy["ReLIEF"][seed]["productive"])
                         - float(by_policy["random"][seed]["productive"]))
                for seed in paired]
            mean, half = _t_ci(diffs)
            output.append({
                "controller_placement": placement,
                "target_load": load,
                "comparison": "ReLIEF-random",
                "metric": "productive_percentage_points",
                "n_pairs": len(diffs),
                "mean_paired_difference": mean,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
            })
    return output


def export_outputs(
    cells: Sequence[Mapping], out_dir: Path, signature: Mapping,
    profile: Mapping,
) -> None:
    summaries = [dict(c["summary"]) for c in cells]
    per_uav = []
    for cell in cells:
        meta = {
            key: cell["summary"][key] for key in (
                "policy", "controller_placement", "seed", "target_load",
                "realized_load")
        }
        for node, values in cell["per_node"].items():
            per_uav.append({"node_id": node, **meta, **values})
    _write_csv(out_dir / "fig7_seed_load_policy.csv", summaries)
    _write_csv(out_dir / "fig7_per_uav.csv", per_uav)
    _write_csv(
        out_dir / "fig7_paired_statistics.csv",
        paired_statistics(summaries))
    _write_csv(
        out_dir / "fig7_relief_random_delta.csv",
        relief_random_differences(summaries))
    with gzip.open(out_dir / "fig7_event_audit.jsonl.gz", "wt") as fh:
        for cell in cells:
            meta = {
                key: cell["summary"][key] for key in (
                    "policy", "controller_placement", "seed", "target_load")}
            for row in cell["audit"]:
                fh.write(json.dumps({**meta, **row}, allow_nan=True) + "\n")
    _atomic_json(
        out_dir / "fig7_experiment_manifest.json",
        {**signature, "controller_profile": profile,
         "statistical_unit": "paired evaluation seed",
         "confidence_interval": "two-sided Student-t 95%",
         "training_seed_uncertainty_included": False})
    _atomic_json(out_dir / "controller_profile.json", profile)
    render_figure7(summaries, per_uav, out_dir)


def render_figure7(
    rows: Sequence[Mapping], per_uav: Sequence[Mapping], out_dir: Path,
) -> None:
    rows = [r for r in rows if r["controller_placement"] == "native"]
    per_uav = [
        r for r in per_uav if r["controller_placement"] == "native"]
    n_seeds = len({int(r["seed"]) for r in rows})
    fig = plt.figure(figsize=(12.8, 4.05))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.08, 1.60, 1.18), wspace=0.32)
    axa, axb, axc = [fig.add_subplot(grid[0, i]) for i in range(3)]

    loads = sorted({float(r["target_load"]) for r in rows})
    realized_x = []
    for load in loads:
        realized_x.append(float(np.mean([
            float(r["realized_load"]) for r in rows
            if math.isclose(float(r["target_load"]), load)])))
    for policy in POLICIES:
        means, cis = [], []
        for load in loads:
            vals = [
                100.0 * float(r["productive"]) for r in rows
                if r["policy"] == policy
                and math.isclose(float(r["target_load"]), load)]
            mean, ci = _t_ci(vals)
            means.append(mean)
            cis.append(ci)
        axa.errorbar(
            realized_x, means, yerr=cis, label=policy, color=COLORS[policy],
            marker=MARKERS[policy], ls=LINESTYLES[policy], lw=1.7,
            capsize=2.5, ms=4.5,
            markerfacecolor="white" if policy == "ReLIEF" else COLORS[policy])
    ideal_x = np.linspace(min(realized_x), max(realized_x), 200)
    axa.plot(ideal_x, 100 * np.minimum(ideal_x, 1.0), "--",
             color="#555555", lw=1.0, label=r"ceiling $\min(\rho,1)$")
    if min(realized_x) <= 0.96 <= max(realized_x):
        axa.axvline(0.96, color="#777777", ls=":", lw=0.8)
        axa.text(
            0.965, 3, "180 tasks/s", rotation=90, fontsize=7,
            va="bottom", clip_on=True)
    axa.set(
        xlabel=r"realized offered compute load $\rho_{\rm in}$",
        ylabel="on-time compute goodput\n(% provisioned capacity)",
        title="(a) Useful work under load",
        xlim=(min(realized_x) - 0.05, max(realized_x) + 0.05), ylim=(0, 105))
    axa.legend(fontsize=6.8, loc="upper left", handlelength=2.0)
    axa.text(
        0.98, 0.03, f"n={n_seeds} paired seeds",
        transform=axa.transAxes, ha="right", fontsize=6.4,
        color="#555555")
    representative = tuple(
        load for load in (0.50, 1.00, 1.50)
        if any(math.isclose(load, available) for available in loads))
    if not representative:
        representative = tuple(loads)
    x_positions, labels = [], []
    gap = 1.1
    for group, load in enumerate(representative):
        base = group * (len(POLICIES) + gap)
        for offset, policy in enumerate(POLICIES):
            x = base + offset
            subset = [
                r for r in rows if r["policy"] == policy
                and math.isclose(float(r["target_load"]), load)]
            bottoms = 0.0
            for category in ("productive", "wasted", "control", "idle"):
                value = 100.0 * float(np.mean([
                    float(r[category]) for r in subset]))
                axb.bar(
                    x, value, bottom=bottoms, width=0.82,
                    color=CATEGORY_COLORS[category],
                    hatch=CATEGORY_HATCHES[category],
                    edgecolor="white", linewidth=0.4,
                    label=category if group == 0 and offset == 0 else None)
                bottoms += value
            productive_values = [
                100.0 * float(r["productive"]) for r in subset]
            productive_mean, productive_ci = _t_ci(productive_values)
            axb.errorbar(
                x, productive_mean, yerr=productive_ci, fmt="none",
                ecolor="#222222", elinewidth=0.75, capsize=1.8, zorder=5)
            if not math.isclose(bottoms, 100.0, abs_tol=1e-6):
                raise AssertionError("capacity stack does not sum to 100%")
            x_positions.append(x)
            labels.append({
                "Attention+PPO": "Attn+PPO", "random": "Random",
                "FODAS": "FODAS", "ReLIEF": "ReLIEF"}[policy])
        center = base + (len(POLICIES) - 1) / 2
        axb.text(center, 103.0, rf"$\rho={load:.2f}$",
                 ha="center", va="bottom", fontsize=8)
    axb.set_xticks(x_positions, labels, rotation=32, ha="right", fontsize=6.5)
    axb.set(
        ylabel="provisioned capacity (%)",
        title="(b) Where provisioned capacity goes", ylim=(0, 108))
    axb.legend(ncol=2, fontsize=6.8, loc="lower center")

    panel_c_load = min(loads, key=lambda value: abs(value - 1.0))
    load_one = [
        r for r in per_uav
        if math.isclose(float(r["target_load"]), panel_c_load)]
    data = [
        [100.0 * float(r["busy_utilization"]) for r in load_one
         if r["policy"] == policy]
        for policy in POLICIES]
    axc.axhspan(0, 20, color="#56B4E9", alpha=0.10, label="<20% near-idle")
    axc.axhspan(90, 100, color="#D55E00", alpha=0.10, label=">90% saturated")
    violins = axc.violinplot(
        data, positions=np.arange(len(POLICIES)), widths=0.75,
        showmeans=False, showmedians=True, showextrema=False)
    for body, policy in zip(violins["bodies"], POLICIES):
        body.set_facecolor(COLORS[policy])
        body.set_alpha(0.35)
    rng = np.random.default_rng(1207)
    for idx, (policy, values) in enumerate(zip(POLICIES, data)):
        jitter = rng.uniform(-0.20, 0.20, len(values))
        axc.scatter(
            idx + jitter, values, s=5, alpha=0.20,
            color=COLORS[policy], edgecolors="none")
        if values:
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            axc.errorbar(
                idx, median, yerr=[[median - q1], [q3 - median]],
                fmt="_", color=COLORS[policy], capsize=4,
                markersize=12, linewidth=1.4, zorder=4)
        seed_backlog = [
            100.0 * float(r["backlog_exposure"]) for r in rows
            if r["policy"] == policy
            and math.isclose(float(r["target_load"]), panel_c_load)]
        mean, ci = _t_ci(seed_backlog)
        axc.text(
            idx, 102.0, f"$B$\n{mean:.1f}±{ci:.1f}%",
            ha="center", va="bottom", fontsize=6.5,
            color=COLORS[policy], linespacing=0.9)
    axc.set_xticks(
        np.arange(len(POLICIES)),
        [p.replace("Attention+", "Attn+\n") for p in POLICIES],
        rotation=20, ha="right", fontsize=7)
    axc.set(
        ylabel="per-UAV true busy utilisation (%)",
        title=rf"(c) Node load and congestion at $\rho={panel_c_load:.2f}$",
        ylim=(0, 110))
    axc.legend(fontsize=6.5, loc="lower right")
    axc.text(
        0.02, 0.03, "Pooled nodes are descriptive\n$B$: seed mean ± 95% CI",
        transform=axc.transAxes, fontsize=6.2, color="#555555")

    for ax in (axa, axb, axc):
        ax.grid(axis="y", alpha=0.20)
        ax.spines[["top", "right"]].set_visible(False)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig7_system_utilisation.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(out_dir / "fig7_system_utilisation.pdf",
                bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("figures/event_ppo_final.pt"))
    parser.add_argument(
        "--controller-scale", type=float, default=1.0,
        help="sensitivity multiplier for every declared controller service demand")
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--loads", type=float, nargs="+", default=LOAD_TARGETS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    parser.add_argument(
        "--placements", choices=PLACEMENTS, nargs="+", default=PLACEMENTS)
    parser.add_argument(
        "--policies", choices=POLICIES, nargs="+", default=POLICIES,
        help="policies to execute; a subset requires --base-checkpoints")
    parser.add_argument(
        "--base-checkpoints", type=Path,
        help="complete checkpoint directory whose unselected cells are preserved")
    parser.add_argument(
        "--plot-partial", action="store_true",
        help="render figures/CSVs from completed cell checkpoints and exit")
    args = parser.parse_args(argv)
    if args.plot_partial:
        cells, _signature = load_checkpoint_cells(args.output / "checkpoints")
        if not cells:
            parser.error("no completed utilisation checkpoints found")
        export_partial_outputs(cells, args.output)
        print(f"Rendered partial outputs from {len(cells)} completed cells")
        return 0
    profile = require_profile(args.profile)
    if not math.isclose(args.controller_scale, 1.0):
        profile = scale_profile(profile, args.controller_scale)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    nets = _load_nets(args.checkpoint)
    try:
        run_study(
            profile=profile, nets=nets, out_dir=args.output,
            loads=tuple(args.loads), seeds=tuple(args.seeds),
            placements=tuple(args.placements),
            policies=tuple(args.policies),
            base_checkpoint_dir=args.base_checkpoints,
            policy_checkpoint_hash=checkpoint_hash)
    except KeyboardInterrupt:
        cells, _ = load_checkpoint_cells(args.output / "checkpoints")
        export_partial_outputs(cells, args.output)
        progress = args.output / "fig7_progress.json"
        if progress.exists():
            payload = json.loads(progress.read_text())
            payload["status"] = "paused"
            _atomic_json(progress, payload)
        print(f"\nPaused safely after {len(cells)} completed cells. "
              "Re-run the same command to resume.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
