"""
experiments.py
==============
Paper-figure generation harness.

Runs instrumented evaluation episodes of the two-stage pipeline under controlled
ablations and produces every results figure for the paper into ./figures :

  fig1_total_delay        — total delay: headless (edge->uniform-random node direct) vs full methodology
                            (Stage-1 MEREC->Kalman->KL->SPOTIS head + Stage-2 distribution)
  fig2_comm_overhead      — communication overhead: headless direct-to-random-node vs optimal-head system
  fig3_propagation_delay  — task-distribution propagation delay: event_driven / random / static mobility
  fig4_energy_mobility    — energy consumption: event_driven / random / static mobility
  fig5_edge_energy        — edge->fog upload energy: uniform-random node vs fog-head-mediated
  fig6_fairness           — Jain fairness index: headless random scheduling vs fair (attention+PPO) scheduling
  fig7_utilisation        — system utilisation: per-node CPU utilisation, running utilisation, memory occupancy

Every run re-seeds SEED so all configurations see the SAME world and task stream;
only the ablated mechanism differs.  The shared encoder/actor are pre-trained for
a few offline episodes once, then frozen (greedy eval) for all runs.

Usage:  conda run -n venv python3 experiments.py
"""

import io
import os
import math
import random
import time
import tempfile
import contextlib
import argparse
import csv
import json
import pickle
import signal
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(),
                                                   "uav_fog_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(),
                                                     "uav_fog_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from config import *
import world
from world import (build_fog_swarm, build_iot_devices, generate_task_stream,
                   generate_task_stream_for_duration, assign_mobility,
                   step_mobility)
from physics import (slant_distance_m, uav_uav_distance_m, data_rate_bps,
                     transmission_delay_ms, execution_delay_ms, queue_delay_ms,
                     aero_power_w, processing_energy_j,
                     update_memory_occupancy, memory_efficiency,
                     system_workload_imbalance, base_reliability, total_delay_ms,
                     head_operational_availability,
                     primary_backup_reliability, system_reliability_nines)
from broker import BrokerSelector
from head_baselines import make_head_selector
from neural import (AttentionEncoder, Actor, Critic,
                    build_candidate_features, filter_candidates,
                    behavior_clone_fodas, _NO_NODE)
import simulation
from baselines.fodas_baseline import select_node_for_task as _fodas_select
from baselines.relief_baseline import (
    discretize_state as _relief_state,
    update_reliability_log_ema as _relief_update_ema,
    WL_TARGET_CYCLES as _RELIEF_WL_TARGET,
    pretrain_agent as _relief_pretrain,
)

# ---------------------------------------------------------------------------
# Boot-time API contract: assert every neural symbol this file calls exists.
# If neural.py renames or removes a function, this block raises an explicit
# ImportError here (line ~52) instead of a NameError deep inside a eval loop.
# Update this list whenever you add/remove a neural call in experiments.py.
# ---------------------------------------------------------------------------
_REQUIRED_NEURAL_API = [
    "AttentionEncoder", "Actor", "Critic",
    "build_candidate_features", "filter_candidates", "behavior_clone_fodas",
    "_NO_NODE",
]
import neural as _neural_mod
for _sym in _REQUIRED_NEURAL_API:
    if not hasattr(_neural_mod, _sym):
        raise ImportError(
            f"experiments.py requires neural.{_sym} but it no longer exists. "
            f"Update the call site in experiments.py and this guard list."
        )
del _neural_mod, _sym  # keep namespace clean

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TASK_PROGRESS_PATH = os.path.join(FIG_DIR, "task_distribution_progress.json")
TASK_TRAINING_CHECKPOINT = os.path.join(
    FIG_DIR, "_task_distribution_training_checkpoint_v2.pt")
TASK_RUNS_CHECKPOINT = os.path.join(
    FIG_DIR, "_task_distribution_runs_checkpoint.pkl")
_STOP_AFTER_STAGE = False

# Experiment-only constants ----------------------------------------------------
CONTROL_REQUEST_KB: float = 0.25  # KB — IoT admission request sent to the elected head
CONTROL_ASSIGN_KB:  float = 0.25  # KB — head-to-executor assignment/control message
N_PRETRAIN:     int   = 300    # offline PPO fine-tuning episodes after FODAS behavior cloning
N_EVAL_TASKS:   int   = 25_000 # tasks per instrumented eval episode (~139 s sim) — long enough
                                # for propulsion+compute draw to pull weak batteries below the
                                # SoC floor, so SoC-aware dispatch separates from blind random
LIGHT_SPEED:    float = 3.0e8  # m/s

COL = {"optimal": "#1668b4", "headless": "#d1621e",
       "event_driven": "#1668b4", "random": "#d1621e", "static": "#7f7f7f",
       "rl": "#1668b4", "rand_sched": "#d1621e",
       "fodas": "#2ca02c",   # FODAS baseline — green
       "relief": "#9467bd"}  # ReLIEF baseline — purple

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return (f"{hours:d}h {minutes:02d}m {secs:02d}s" if hours
            else f"{minutes:d}m {secs:02d}s")


def _write_json_atomic(path: str, payload: Dict) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w") as fh:
        json.dump(payload, fh, indent=2, allow_nan=True)
    os.replace(temp_path, path)


def _write_pickle_atomic(path: str, payload) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "wb") as fh:
        pickle.dump(payload, fh)
    os.replace(temp_path, path)


def _write_torch_atomic(path: str, payload: Dict) -> None:
    temp_path = f"{path}.tmp"
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def _write_task_progress(**values) -> None:
    payload = {
        "design": "task-distribution paper-figure experiment",
        "seed": SEED,
        "requested_ppo_episodes": N_PRETRAIN,
        "evaluation_tasks_per_run": N_EVAL_TASKS,
        "requested_evaluation_runs": 6,
        "progress_file": TASK_PROGRESS_PATH,
        **values,
        "updated_unix_s": time.time(),
    }
    _write_json_atomic(TASK_PROGRESS_PATH, payload)


def _request_task_pause(_signum, _frame) -> None:
    global _STOP_AFTER_STAGE
    if not _STOP_AFTER_STAGE:
        _STOP_AFTER_STAGE = True
        print("\n[PAUSE REQUESTED] The active PPO episode or evaluation run will "
              "finish, checkpoint, and stop.", flush=True)
    else:
        print("\n[PAUSE REQUESTED] Already waiting for the active stage to finish.",
              flush=True)


def _set_mode(mode: str) -> None:
    """Point every module's MOVE_MODE binding at `mode` (import-* copies)."""
    config.MOVE_MODE = mode
    world.MOVE_MODE = mode
    simulation.MOVE_MODE = mode


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] {name}.png / .pdf written")


def jain(xs: List[float]) -> float:
    """Jain's fairness index J = (sum x)^2 / (n * sum x^2), in (0, 1]."""
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    n = len(xs)
    return (s1 * s1) / (n * s2) if s2 > 0 else 1.0


def control_plane_cost(dev, head, executor) -> Dict[str, float]:
    """Physical control path from an IoT device through the elected fog head.

    Task payloads still travel directly from the IoT device to the executor.
    The small admission request travels IoT->head, and the assignment travels
    head->executor when they are different nodes.  Returned swarm energy covers
    receiver energy at the head plus head transmit/executor receive energy; IoT
    transmit energy is reported separately.
    """
    if head is None:
        return {"delay_ms": 0.0, "swarm_energy_j": 0.0,
                "edge_energy_j": 0.0, "head_energy_j": 0.0,
                "executor_energy_j": 0.0}

    request_rate = data_rate_bps(
        slant_distance_m(head, dev.x, dev.y), dev.p_tx_w)
    request_ms = transmission_delay_ms(CONTROL_REQUEST_KB, request_rate)
    edge_energy = dev.p_tx_w * request_ms / 1000.0
    head_rx_energy = P_COMM * request_ms / 1000.0

    assign_ms = 0.0
    head_tx_energy = 0.0
    executor_rx_energy = 0.0
    if executor.node_id != head.node_id:
        assign_rate = data_rate_bps(
            uav_uav_distance_m(head, executor), P_UAV_TX_W)
        assign_ms = transmission_delay_ms(CONTROL_ASSIGN_KB, assign_rate)
        head_tx_energy = P_UAV_TX_W * assign_ms / 1000.0
        executor_rx_energy = P_COMM * assign_ms / 1000.0

    return {
        "delay_ms": request_ms + assign_ms,
        "swarm_energy_j": head_rx_energy + head_tx_energy + executor_rx_energy,
        "edge_energy_j": edge_energy,
        "head_energy_j": head_rx_energy + head_tx_energy,
        "executor_energy_j": executor_rx_energy,
    }


# ==============================================================================
# Instrumented evaluation episode
# ==============================================================================

def _legacy_run_exp(label: str,
            move_mode: str = "event_driven",
            head_mode: str = "spotis",      # spotis | fu_serve | 2dp_fhs | 3d_pos | random | none
            dispatch:  str = "rl",           # 'rl' | 'random' | 'nearest'
            coordinated: bool = True,        # True: head coordinates (CONTROL PLANE only —
                                             # payload always goes IoT->executor direct);
                                             # False: headless, no coordinator at all
            nets=None,
            n_tasks: int = TASKS_PER_EPISODE,
            seed: int = SEED,
            task_arrival_rate: float = TASK_ARRIVAL_RATE,
            episode_duration_s: Optional[float] = None,
            task_stream: Optional[Sequence] = None,
            handover_setup_ms: float = HANDOVER_SETUP_MS,
            controller_placement: str = "equal_controller") -> Dict:
    """
    One instrumented eval episode.  Returns a dict of per-task records, per-tick
    series, per-node accumulators, and episode energy totals.
    """
    _set_mode(move_mode)
    set_global_seed(seed)                                  # identical world for every config

    fog = build_fog_swarm()
    iot, centres = build_iot_devices()
    assign_mobility(fog, centres)
    if task_stream is not None:
        tasks = list(task_stream)
    elif episode_duration_s is not None:
        tasks = generate_task_stream_for_duration(
            episode_duration_s, task_arrival_rate)
    else:
        tasks = generate_task_stream(n_tasks, task_arrival_rate)
    n_tasks = len(tasks)
    if n_tasks == 0 and episode_duration_s is None:
        raise ValueError("the evaluation task stream is empty")
    simulation_end_s = (episode_duration_s if episode_duration_s is not None
                        else tasks[-1].arrival_s)
    broker = BrokerSelector() if head_mode == "spotis" else (
        make_head_selector(head_mode) if head_mode in ("fu_serve", "2dp_fhs", "3d_pos") else None)
    encoder, actor = nets
    encoder.eval(); actor.eval()

    mean_p_tx = (N_LOW_POWER_IOT * P_TX_LOW_W + N_HIGH_POWER_IOT * P_TX_HIGH_W) / N_IOT
    nom_kb = 0.5 * (TASK_SIZE_MIN_KB + TASK_SIZE_MAX_KB)
    from models import Task
    nominal = Task(-1, nom_kb, nom_kb * 1024.0 * CYCLES_PER_BYTE,
                   deadline_for_size_ms(nom_kb), 0.0, False)

    # accumulators ------------------------------------------------------------
    rec: Dict[str, List] = {k: [] for k in
        ("task_id", "deadline_ms", "is_small", "delay_ms", "broker_ms",
         "selector_ms", "handover_ms", "control_ms", "up_ms", "queue_ms",
         "exec_ms", "prop_us", "occupancy_ms", "kb_hops", "edge_j",
         "net_j", "control_j", "success", "dev_id", "head_id")}
    series: Dict[str, List] = {k: [] for k in
        ("t", "prop_power_w", "soc", "jain_load", "prop_us", "busy_frac",
         "mem_occ", "run_util", "mean_workload_cycles", "head_id",
         "head_soc", "head_iot_distance_m", "head_fog_distance_m",
         "head_workload_cycles", "head_availability", "selector_us",
         "selector_ran", "head_changed", "candidate_head_id", "kl_value",
         "switch_block_reason")}
    cum_cycles = {n.node_id: 0.0 for n in fog}             # assigned cycles per node
    E = {"prop": 0.0, "comp": 0.0, "net": 0.0, "control": 0.0,
         "edge": 0.0, "edge_control": 0.0, "edge_wasted": 0.0,
         "net_wasted": 0.0, "signal": 0.0}
    head_changes = 0
    prev_head_id = -1

    # ReLIEF: pretrained Q-agent + an EMA of log10(R_i) tracking "current"
    # system reliability for state discretization (see
    # baselines/relief_baseline.update_reliability_log_ema).
    _relief_agent = _relief_pretrain(SEED) if dispatch == "relief" else None
    _relief_log10_ema = 0.0
    _relief_state_now = None

    sim_t, tick, next_idx = 0.0, 0, 0
    while tick * CONTROL_TICK_S < simulation_end_s or next_idx < n_tasks:
        tick += 1
        sim_t = tick * CONTROL_TICK_S
        tick_end_ms = sim_t * 1000.0
        handover_stall_ms = 0.0    # decision stall for tasks in a handover tick

        # (1) mobility + propulsion
        step_mobility(fog, centres, CONTROL_TICK_S)
        for n in fog:
            p = aero_power_w(n.speed_ms)
            E["prop"] += p * CONTROL_TICK_S
            n.E_res_j = max(0.0, n.E_res_j - p * CONTROL_TICK_S)

        # (2) geometry
        mean_iot_dist = {n.node_id: max(float(np.mean(
            [slant_distance_m(n, d.x, d.y) for d in iot])), 1.0) for n in fog}
        tr_repr = {n.node_id: data_rate_bps(mean_iot_dist[n.node_id], mean_p_tx) for n in fog}

        # (3) Stage 1 — head election
        update_memory_occupancy(fog)
        me = memory_efficiency(fog)
        wl_pn, wl_sys = system_workload_imbalance(fog)
        spotis_ran = False
        st1 = {"run_spotis": False, "kl_value": float("nan"),
               "switch_block_reason": "", "head_changed": False}
        selector_start_ns = time.perf_counter_ns()
        if head_mode == "spotis":
            pm = {n.node_id: {"me": me[n.node_id],
                              "r0": base_reliability(n, nominal.cycles, nominal.size_kb, tr_repr[n.node_id]),
                              "d_ms": total_delay_ms(nominal, n, mean_iot_dist[n.node_id], mean_p_tx, False)["total_ms"],
                              "wl": wl_pn[n.node_id]} for n in fog}
            st1 = broker.select_head(fog, pm, 0, tick)
            head = st1["head_node"]
            spotis_ran = st1["run_spotis"]
        elif head_mode in ("fu_serve", "2dp_fhs", "3d_pos"):
            pm = {n.node_id: {"me": me[n.node_id], "wl": wl_pn[n.node_id]}
                  for n in fog}
            st1 = broker.select_head(fog, pm, 0, tick, iot_devices=iot)
            head = st1["head_node"]
        elif head_mode == "random":
            head = random.choice(fog)
        else:
            # headless system: no fog-head at all — edge devices talk straight to
            # the chosen fog node; no election, no signalling, no coordination
            head = None
        selector_us = ((time.perf_counter_ns() - selector_start_ns) / 1000.0
                       if head is not None else 0.0)

        # head-handover signalling: broadcast SIGNAL_KB state-sync to every UAV
        head_changed_this_tick = (
            head is not None and prev_head_id >= 0 and head.node_id != prev_head_id)
        if head is not None and head.node_id != prev_head_id:
            if prev_head_id >= 0:
                head_changes += 1
                if coordinated:
                    # re-election setup phase stalls admission decisions this tick
                    handover_stall_ms = handover_setup_ms
            mean_d = float(np.mean([uav_uav_distance_m(head, n) for n in fog if n is not head]))
            r_sig = data_rate_bps(mean_d, P_UAV_TX_W)
            t_sig_s = (SIGNAL_KB * 1024 * 8 / r_sig) * (len(fog) - 1)
            signal_energy = (P_UAV_TX_W + P_COMM) * t_sig_s
            E["signal"] += signal_energy
            head.E_res_j = max(0.0, head.E_res_j - P_UAV_TX_W * t_sig_s)
            receiver_energy = P_COMM * t_sig_s / max(len(fog) - 1, 1)
            for receiver in fog:
                if receiver.node_id != head.node_id:
                    receiver.E_res_j = max(0.0, receiver.E_res_j - receiver_energy)
            prev_head_id = head.node_id

        # (4) route arriving tasks
        tick_tasks = []
        while next_idx < n_tasks and tasks[next_idx].arrival_s * 1000.0 < tick_end_ms:
            tick_tasks.append(tasks[next_idx]); next_idx += 1

        # FODAS: EDF-sort the tick batch before per-task dispatch
        if dispatch == "fodas" and tick_tasks:
            from baselines.fodas_baseline import edf_sort as _edf_sort
            tick_tasks = _edf_sort(tick_tasks)

        for task in tick_tasks:
            dev = iot[random.randint(0, N_IOT - 1)]

            if dispatch == "rl":
                # Build task-aware, source-aware feature matrix and
                # physics-guided top-K candidate mask for this specific task + IoT device.
                x_feat    = build_candidate_features(fog, task, dev, me)
                cand_mask, h_scores = filter_candidates(fog, task, dev)
                with torch.no_grad():
                    H, c = encoder(x_feat, update_stats=False)
                    soc_vec = torch.tensor([n.soc for n in fog], dtype=torch.float32)
                    f_p, f_b, *_ = actor.select_action(
                        H, c, soc_vec, greedy=True,
                        candidate_mask=cand_mask,
                        heuristic_scores=h_scores,
                    )
            elif dispatch == "fodas":
                # FODAS heuristic: EDF admission + deadline-feasible candidate
                # filtering + completion-time/energy ranked node choice.
                # select_node_for_task() returns (f_p, f_b) indices into fog[].
                f_p, f_b = _fodas_select(
                    task=task,
                    iot_dev=dev,
                    fog_nodes=fog,
                    sim_time_s=sim_t,
                )
            elif dispatch == "relief":
                # ReLIEF Q-learning: state = discretized (system reliability,
                # workload balance); action = greedy (primary, backup) pair
                # from the pretrained Q-table (Eq.2). wl_sys is this tick's
                # scalar workload imbalance (Eq.17), already computed above.
                _relief_nines_now = system_reliability_nines(_relief_log10_ema)
                _relief_state_now = _relief_state(
                    _relief_nines_now, wl_sys, _RELIEF_WL_TARGET, _relief_agent.cfg)
                f_p, f_b = _relief_agent.select_pair(
                    fog, task, dev, _relief_state_now, greedy=True)
            elif dispatch == "nearest":
                feas = [j for j, n in enumerate(fog)
                        if n.soc >= SOC_PRIMARY_MIN
                        and len(n.queue_primary) + len(n.queue_backup) < MAX_QUEUE_DEPTH]
                if not feas:
                    f_p, f_b = _NO_NODE, _NO_NODE
                else:
                    feas.sort(key=lambda j: slant_distance_m(fog[j], dev.x, dev.y))
                    f_p = feas[0]
                    f_b = feas[1] if len(feas) > 1 else _NO_NODE
            else:
                # random scheduling: ANY fog node picked uniformly — no distance
                # preference, no battery/queue feasibility screen, and no backup
                # replica (headless system has no coordinator to know node state
                # or to arrange delayed-backup recovery)
                f_p = random.randrange(len(fog))
                f_b = _NO_NODE

            if f_p == _NO_NODE:
                for k in rec:
                    if k == "task_id":
                        value = task.task_id
                    elif k == "deadline_ms":
                        value = task.deadline_ms
                    elif k == "is_small":
                        value = int(task.is_small)
                    elif k == "dev_id":
                        value = dev.dev_id
                    elif k == "head_id":
                        value = -1 if head is None else head.node_id
                    elif k == "success":
                        value = 0
                    else:
                        value = 0.0
                    rec[k].append(value)
                continue

            prim = fog[f_p]

            # --- delay path ---
            # The payload goes IoT->executor directly, while the physical
            # request/assignment control path traverses the elected head.
            d_up = slant_distance_m(prim, dev.x, dev.y)
            r_up = data_rate_bps(d_up, dev.p_tx_w)
            t_up = transmission_delay_ms(task.size_kb, r_up)
            # Charge every policy its measured Stage-1 wall time for this tick,
            # plus the common Stage-2 inference cost. This avoids treating the
            # three baseline selectors as computationally free.
            t_selector = selector_us / 1000.0
            if dispatch == "rl":
                t_selector += T_PPO_INFER_MS
            t_handover = handover_stall_ms if coordinated else 0.0
            t_brk = t_selector + t_handover
            control = (control_plane_cost(dev, head, prim) if coordinated else
                       control_plane_cost(dev, None, prim))
            t_control = control["delay_ms"]
            t_q = queue_delay_ms(prim)
            t_ex = execution_delay_ms(task.cycles, prim.fr_avg_hz())
            D = t_brk + t_control + t_up + t_q + t_ex

            # --- comm / energy accounting ---
            e_edge = dev.p_tx_w * (t_up / 1000.0)                  # IoT uplink tx
            e_net = P_COMM * (t_up / 1000.0)                       # executor rx
            E["edge"] += e_edge
            E["net"] += e_net
            E["edge_control"] += control["edge_energy_j"]
            E["control"] += control["swarm_energy_j"]
            if head is not None:
                head.E_res_j = max(
                    0.0, head.E_res_j - control["head_energy_j"])
            prim.E_res_j = max(
                0.0, prim.E_res_j - control["executor_energy_j"])

            # --- execute (queue push, survival, success) ---
            deadline_abs = task.arrival_s * 1000.0 + task.deadline_ms
            prim.queue_primary.append((task, deadline_abs))
            cum_cycles[f_p] += task.cycles
            E["comp"] += processing_energy_j(prim, task.cycles)
            prim.E_res_j = max(0.0, prim.E_res_j - processing_energy_j(prim, task.cycles) - e_net)

            expo_s = (t_q + t_ex) / 1000.0
            # a node below the SoC floor cannot execute — matters for random
            # dispatch, which selects blind; screened dispatchers never hit it
            surv = (prim.soc >= SOC_PRIMARY_MIN
                    and random.random() < math.exp(-(prim.lambda_fail + prim.mu_fail) * expo_s))
            success = 1 if (surv and D <= task.deadline_ms) else 0

            if not success and f_b != _NO_NODE:                    # delayed backup
                # coordinator-arranged recovery: the payload already sits at the
                # primary, so it is forwarded UAV->UAV from primary to backup
                bak = fog[f_b]
                d_b = uav_uav_distance_m(prim, bak)
                r_b = data_rate_bps(d_b, P_UAV_TX_W)
                t_ub = transmission_delay_ms(task.size_kb, r_b)
                t_qb = queue_delay_ms(bak)
                t_eb = execution_delay_ms(task.cycles, bak.fr_avg_hz())
                if max(0.0, task.deadline_ms - D) >= t_ub + t_qb + t_eb:
                    bak.queue_backup.append((task, deadline_abs))
                    cum_cycles[f_b] += task.cycles
                    E["comp"] += processing_energy_j(bak, task.cycles)
                    e_b = P_UAV_TX_W * t_ub / 1000.0
                    E["net"] += e_b; e_net += e_b
                    expo_b = (t_qb + t_eb) / 1000.0
                    surv_b = (bak.soc >= SOC_PRIMARY_MIN
                              and random.random() < math.exp(-(bak.lambda_fail + bak.mu_fail) * expo_b))
                    success = 1 if (surv_b and D + t_ub + t_qb + t_eb - t_q - t_ex <= task.deadline_ms) else success

            if not success:
                E["edge_wasted"] += e_edge + control["edge_energy_j"]
                E["net_wasted"] += e_net + control["swarm_energy_j"]

            if dispatch == "relief" and _relief_state_now is not None:
                # ReLIEF online Q-update (Algorithm 2 lines 17-19): reward
                # Eq.(21) from this task's realised delay D, workload after
                # assignment (wl_sys already reflects the queue push above
                # only on the NEXT tick's system_workload_imbalance() call,
                # so we recompute it here for the post-assignment state),
                # and pair reliability Eq.(10)-(11) via base_reliability().
                r0_p = base_reliability(prim, task.cycles, task.size_kb, r_up)
                if f_b != _NO_NODE:
                    bak = fog[f_b]
                    d_b2 = uav_uav_distance_m(prim, bak)
                    r0_b = base_reliability(bak, task.cycles, task.size_kb,
                                             data_rate_bps(d_b2, P_UAV_TX_W))
                    r_pair = primary_backup_reliability(r0_p, r0_b)
                else:
                    r_pair = r0_p
                _relief_log10_ema = _relief_update_ema(_relief_log10_ema, r_pair, _relief_agent.cfg)

                wl_pn_after, wl_sys_after = system_workload_imbalance(fog)
                nines_after = system_reliability_nines(_relief_log10_ema)
                next_state = _relief_state(
                    nines_after, wl_sys_after, _RELIEF_WL_TARGET, _relief_agent.cfg)
                rwd = _relief_agent.reward(
                    wl_sys_after, _RELIEF_WL_TARGET, D, task.deadline_ms, r_pair)
                _relief_agent.observe(_relief_state_now, (f_p, f_b), rwd, next_state, fog)

            rec["task_id"].append(task.task_id)
            rec["deadline_ms"].append(task.deadline_ms)
            rec["is_small"].append(int(task.is_small))
            rec["delay_ms"].append(D); rec["broker_ms"].append(t_brk)
            rec["selector_ms"].append(t_selector)
            rec["handover_ms"].append(t_handover)
            rec["control_ms"].append(t_control)
            rec["up_ms"].append(t_up)
            rec["queue_ms"].append(t_q); rec["exec_ms"].append(t_ex)
            rec["prop_us"].append(d_up / LIGHT_SPEED * 1e6)
            rec["occupancy_ms"].append(t_up)
            rec["kb_hops"].append(task.size_kb)
            rec["edge_j"].append(e_edge); rec["net_j"].append(e_net)
            rec["control_j"].append(control["swarm_energy_j"])
            rec["success"].append(success)
            rec["dev_id"].append(dev.dev_id)
            rec["head_id"].append(-1 if head is None else head.node_id)

        # (5) queue cleanup
        cutoff = sim_t * 1000.0
        for n in fog:
            n.queue_primary = [(t, d) for (t, d) in n.queue_primary if d >= cutoff]
            n.queue_backup = [(t, d) for (t, d) in n.queue_backup if d >= cutoff]

        # (6) per-tick series
        series["t"].append(sim_t)
        series["prop_power_w"].append(float(np.mean([aero_power_w(n.speed_ms) for n in fog])))
        series["soc"].append(float(np.mean([n.soc for n in fog])))
        series["jain_load"].append(jain([cum_cycles[n.node_id] + 1.0 for n in fog]))
        k = len(tick_tasks)
        series["prop_us"].append(float(np.mean(rec["prop_us"][-k:])) if k else np.nan)
        series["busy_frac"].append(np.mean([1.0 if (n.queue_primary or n.queue_backup) else 0.0 for n in fog]))
        series["mem_occ"].append(float(np.mean([n.MP_occ_gb / n.MP_tot_gb for n in fog])))
        series["run_util"].append(float(np.mean(
            [min(1.0, cum_cycles[n.node_id] / (n.fr_avg_hz() * sim_t)) for n in fog])))
        series["mean_workload_cycles"].append(float(np.mean(list(wl_pn.values()))))
        series["head_id"].append(-1 if head is None else head.node_id)
        if head is None:
            head_soc = head_iot_d = head_fog_d = head_wl = head_avail = float("nan")
        else:
            head_soc = head.soc
            head_iot_d = mean_iot_dist[head.node_id]
            head_fog_d = float(np.mean([
                uav_uav_distance_m(head, n) for n in fog if n is not head]))
            head_wl = wl_pn[head.node_id]
            head_avail = head_operational_availability(
                head, HEAD_AVAILABILITY_HORIZON_S)
        series["head_soc"].append(head_soc)
        series["head_iot_distance_m"].append(head_iot_d)
        series["head_fog_distance_m"].append(head_fog_d)
        series["head_workload_cycles"].append(head_wl)
        series["head_availability"].append(head_avail)
        series["selector_us"].append(selector_us)
        selector_ran = (spotis_ran if head_mode == "spotis" else
                        head_mode in ("fu_serve", "2dp_fhs", "3d_pos", "random"))
        series["selector_ran"].append(int(selector_ran))
        series["head_changed"].append(int(head_changed_this_tick))
        series["candidate_head_id"].append(
            st1.get("candidate_head_id", -1 if head is None else head.node_id))
        series["kl_value"].append(float(st1.get("kl_value", float("nan"))))
        series["switch_block_reason"].append(st1.get("switch_block_reason", ""))

    dur = sim_t
    util = {n.node_id: min(1.0, cum_cycles[n.node_id] / (n.fr_avg_hz() * dur)) for n in fog}
    out = {"label": label, "rec": rec, "series": series, "E": E, "util": util,
           "cum_cycles": cum_cycles, "head_changes": head_changes, "dur": dur,
           "task_arrival_rate": task_arrival_rate,
           "episode_duration_s": episode_duration_s,
           "succ_rate": float(np.mean(rec["success"])) if rec["success"] else 0.0,
           "fr": {n.node_id: n.fr_avg_ghz for n in fog}}
    print(f"  [RUN ] {label:22s} tasks={len(rec['success'])} succ={out['succ_rate']*100:.1f}% "
          f"meanD={np.mean(rec['delay_ms']):.1f}ms headChg={head_changes} "
          f"E_prop={E['prop']/1e3:.0f}kJ")
    return out


def run_exp(label: str,
            move_mode: str = "event_driven",
            head_mode: str = "spotis",
            dispatch: str = "rl",
            coordinated: bool = True,
            nets=None,
            n_tasks: int = TASKS_PER_EPISODE,
            seed: int = SEED,
            task_arrival_rate: float = TASK_ARRIVAL_RATE,
            episode_duration_s: Optional[float] = None,
            task_stream: Optional[Sequence] = None,
            handover_setup_ms: float = HANDOVER_SETUP_MS,
            controller_placement: str = "equal_controller",
            n_fog: int = N_FOG,
            n_iot: int = N_IOT,
            head_failure_s: Optional[float] = None) -> Dict:
    """Compatibility view over the shared event-driven scenario runner.

    The legacy analytical evaluator is retained privately for forensic
    reproduction, but active paper experiments use the same CPU timeline as
    Figure 7. ``handover_setup_ms`` remains accepted for API compatibility;
    controller service itself is now the causal decision dependency.
    """
    if task_stream is not None or dispatch == "nearest":
        if head_failure_s is not None:
            raise ValueError("head failure requires the shared scenario runner")
        return _legacy_run_exp(
            label, move_mode, head_mode, dispatch, coordinated, nets,
            n_tasks, seed, task_arrival_rate, episode_duration_s,
            task_stream, handover_setup_ms)
    from controller_profile import require_profile
    from scenario import ScenarioConfig, make_policy, run_scenario

    policy = {
        "rl": "Attention+PPO", "random": "random",
        "fodas": "FODAS", "relief": "ReLIEF",
        "head_fodas": "head_fodas",
    }.get(dispatch)
    if policy is None:
        raise ValueError(f"unsupported dispatch policy {dispatch}")
    if policy == "Attention+PPO" and nets is None:
        raise ValueError("Attention+PPO evaluation requires (encoder, actor)")
    if nets is not None:
        nets[0].eval()
        nets[1].eval()
    duration = float(
        episode_duration_s if episode_duration_s is not None
        else max(n_tasks / task_arrival_rate, CONTROL_TICK_S))
    adapter = make_policy(
        policy, seed, nets if policy == "Attention+PPO" else None,
        head_mode=head_mode,
        final_head_selection=controller_placement == "head")
    result = run_scenario(
        ScenarioConfig(
            seed=seed, target_load=0.96, policy=policy,
            controller_placement=controller_placement, warmup_s=0.0,
            measurement_s=duration, move_mode=move_mode,
            arrival_rate_override=task_arrival_rate,
            handover_setup_ms=handover_setup_ms,
            n_fog=n_fog, n_iot=n_iot, head_failure_s=head_failure_s),
        adapter, require_profile())

    rec_keys = (
        "task_id", "deadline_ms", "is_small", "delay_ms", "broker_ms",
        "selector_ms", "handover_ms", "control_ms", "up_ms", "queue_ms",
        "exec_ms", "prop_us", "occupancy_ms", "kb_hops", "edge_j",
        "net_j", "control_j", "success", "dev_id", "head_id",
        "drop_reason")
    rec = {key: [] for key in rec_keys}
    failed_edge_j = 0.0

    def finite(row, key):
        value = float(row.get(key, 0.0))
        return value if math.isfinite(value) else 0.0

    for row in result.task_records:
        latency = float(row.get("latency_ms", math.nan))
        if not math.isfinite(latency):
            latency = float(row.get("deadline_ms", 0.0))
        queue_ms = float(row.get("queue_delay_ms", math.nan))
        if not math.isfinite(queue_ms):
            queue_ms = float(row.get("deadline_ms", 0.0))
        values = {
            "task_id": int(row["task_id"]),
            "deadline_ms": float(row.get("deadline_ms", 0.0)),
            "is_small": int(row.get("is_small", 0)),
            "delay_ms": latency,
            "broker_ms": float(row.get("decision_ms", 0.0)),
            "selector_ms": float(row.get("selector_ms", 0.0)),
            "handover_ms": 0.0, "control_ms": 0.0,
            "up_ms": finite(row, "upload_ms"),
            "queue_ms": queue_ms,
            "exec_ms": finite(row, "exec_ms"),
            "prop_us": finite(row, "prop_us"),
            "occupancy_ms": finite(row, "upload_ms"),
            "kb_hops": float(row.get("size_kb", 0.0)),
            "edge_j": float(row.get("edge_j", 0.0)),
            "net_j": float(row.get("net_rx_j", 0.0)),
            "control_j": 0.0,
            "success": int(row.get("success", 0)),
            "dev_id": int(row.get("dev_id", -1)),
            "head_id": int(row.get("head_id", -1)),
            "drop_reason": str(row.get("drop_reason", "")),
        }
        if not values["success"]:
            failed_edge_j += values["edge_j"]
        for key in rec:
            rec[key].append(values[key])

    tick_rows = result.tick_records
    final_busy = result.capacity.utilization()["total_busy"]
    head_quality_key = ("head_control_service_ms"
                        if controller_placement == "head"
                        else "head_availability")
    series_keys = (
        "t", "prop_power_w", "soc", "jain_load", "prop_us", "busy_frac",
        "mem_occ", "run_util", "mean_workload_cycles", "head_id", "head_soc",
        "head_iot_distance_m", "head_fog_distance_m", "head_workload_cycles",
        head_quality_key, "selector_us", "selector_ran", "head_changed",
        "candidate_head_id", "kl_value", "switch_block_reason",
        *(f"decision_power_{index}" for index in range(N_CRITERIA)))
    series = {key: [] for key in series_keys}
    for tick_row in tick_rows:
        at_s = float(tick_row["t"])
        recent_prop = [
            float(row.get("prop_us", math.nan)) for row in result.task_records
            if at_s - CONTROL_TICK_S <= float(row.get("arrival_s", -1.0)) < at_s]
        recent_prop = [value for value in recent_prop if math.isfinite(value)]
        values = {
            **tick_row,
            "jain_load": math.nan,
            "prop_us": float(np.mean(recent_prop)) if recent_prop else math.nan,
            "busy_frac": final_busy,
            "mem_occ": 0.0,
            "run_util": final_busy,
            "mean_workload_cycles": 0.0,
        }
        for key in series:
            series[key].append(values.get(key, math.nan if key != "switch_block_reason" else ""))

    energy = result.energy_totals
    edge_total = float(sum(rec["edge_j"]))
    failed_net_j = sum(
        net_j for net_j, success in zip(rec["net_j"], rec["success"])
        if not success)
    signal_energy = float(energy.get("handover_communication_j", 0.0))
    E = {
        "prop": energy["propulsion_j"],
        "comp": energy["task_compute_j"],
        "net": energy["uav_communication_j"] - signal_energy,
        "control": energy["control_compute_j"],
        "edge": edge_total,
        "edge_control": 0.0,
        "edge_wasted": failed_edge_j,
        "net_wasted": failed_net_j,
        "signal": signal_energy,
    }
    util = {
        node_id: values["busy_utilization"]
        for node_id, values in result.capacity.per_node.items()}
    cum_cycles = {
        node_id: values["productive_cycles"] + values["wasted_cycles"]
        for node_id, values in result.capacity.per_node.items()}
    out = {
        "label": label, "rec": rec, "series": series, "E": E,
        "util": util, "cum_cycles": cum_cycles,
        "head_changes": int(sum(series["head_changed"])),
        "dur": duration, "task_arrival_rate": task_arrival_rate,
        "episode_duration_s": duration, "succ_rate": result.success_rate,
        "n_fog": n_fog, "n_iot": n_iot,
        "fr": {
            node_id: values["provisioned_cycles"] / duration / 1.0e9
            for node_id, values in result.capacity.per_node.items()},
        "execution_model_version": result.summary_row()["execution_model_version"],
        "head_recovery_ticks": result.head_recovery_ticks,
        "tasks_dropped_in_gap": result.tasks_dropped_in_gap,
        "delay_spike_ms": result.delay_spike_ms,
        "control_jobs_lost": result.control_jobs_lost,
    }
    failures = Counter(
        reason for success, reason in zip(rec["success"], rec["drop_reason"])
        if not success)
    print(f"  [DROP] {dict(sorted(failures.items()))}")
    print(
        f"  [RUN ] {label:22s} tasks={len(rec['success'])} "
        f"succ={out['succ_rate']*100:.1f}% "
        f"meanD={np.mean(rec['delay_ms']):.1f}ms "
        f"headChg={out['head_changes']} E_prop={E['prop']/1e3:.0f}kJ")
    return out


# ==============================================================================
# Figures
# ==============================================================================

def fig1_total_delay(opt, headless, fodas, relief):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for r, lab, c in (
        (opt,     "Optimal head + full methodology",   COL["optimal"]),
        (headless, "Headless (edge→random node direct)", COL["headless"]),
        (fodas,   "FODAS (EDF heuristic baseline)",     COL["fodas"]),
        (relief,  "ReLIEF (Q-learning baseline)",       COL["relief"]),
    ):
        d = np.sort(r["rec"]["delay_ms"])
        ax[0].plot(d, np.linspace(0, 1, len(d)), label=lab, color=c, lw=1.8)
    ax[0].axvline(DEADLINE_SMALL_MS, ls=":", c="k", lw=0.9)
    ax[0].axvline(DEADLINE_LARGE_MS, ls="--", c="k", lw=0.9)
    ax[0].text(DEADLINE_SMALL_MS, 0.03, " small-task deadline", fontsize=8, rotation=90)
    ax[0].text(DEADLINE_LARGE_MS, 0.03, " large-task deadline", fontsize=8, rotation=90)
    xmax = max(float(np.percentile(r["rec"]["delay_ms"], 99)) for r in (opt, headless, fodas, relief))
    ax[0].set(xlabel="end-to-end task delay (ms)", ylabel="CDF",
              title="(a) Total-delay distribution",
              xlim=(0, max(xmax, DEADLINE_LARGE_MS * 1.3)))
    ax[0].legend(loc="lower right", fontsize=8)

    comps = ["broker_ms", "up_ms", "queue_ms", "exec_ms"]
    names = ["broker pipeline + handover stall", "uplink edge→executor",
             "queue wait", "execution"]
    cols = ["#8c6bb1", "#4292c6", "#ef6548", "#969696"]
    xs = np.arange(4)
    bottoms = np.zeros(4)
    for comp, nm, c in zip(comps, names, cols):
        vals = [
            float(np.mean(opt["rec"][comp])),
            float(np.mean(headless["rec"][comp])),
            float(np.mean(fodas["rec"][comp])),
            float(np.mean(relief["rec"][comp])),
        ]
        ax[1].bar(xs, vals, 0.5, bottom=bottoms, label=nm, color=c)
        bottoms += np.array(vals)
    for x, b in zip(xs, bottoms):
        ax[1].text(x, b + 4, f"{b:.0f} ms", ha="center", fontsize=9, fontweight="bold")
    ax[1].set_xticks(xs, ["Full\nmethodology", "Headless\n(random)", "FODAS", "ReLIEF"])
    ax[1].set(ylabel="mean delay (ms)", title="(b) Mean delay decomposition")
    ax[1].legend(fontsize=8)
    fig.suptitle("Total delay — headless / FODAS / ReLIEF vs optimal fog-head + task distribution", y=1.03)
    _save(fig, "fig1_total_delay_head_selection")


def fig2_comm_overhead(opt, headless, fodas, relief):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    labs = ["Optimal head\nsystem", "Headless\n(random node)", "FODAS", "ReLIEF"]
    cs   = [COL["optimal"], COL["headless"], COL["fodas"], COL["relief"]]
    runs4 = (opt, headless, fodas, relief)

    occ = [sum(r["rec"]["occupancy_ms"]) / 1e3 for r in runs4]
    ax[0].bar(labs, occ, 0.5, color=cs)
    for i, v in enumerate(occ):
        ax[0].text(i, v, f"{v:.1f} s", ha="center", va="bottom", fontsize=9)
    ax[0].set(ylabel="total channel occupancy (s)", title="(a) Air-time for task transfer")

    useful = [r["E"]["net"] - r["E"]["net_wasted"] for r in runs4]
    wasted = [r["E"]["net_wasted"] for r in runs4]
    sig    = [r["E"]["signal"] for r in runs4]
    ax[1].bar(labs, useful, 0.5, color=cs, alpha=0.55, label="forwarding (delivered tasks)")
    ax[1].bar(labs, wasted, 0.5, bottom=useful, color="#c02424", label="wasted (failed tasks)")
    ax[1].bar(labs, sig, 0.5, bottom=np.array(useful) + wasted, color="#7a7a7a",
              label="head-election signalling")
    for i in range(4):
        tot = useful[i] + wasted[i] + sig[i]
        ax[1].text(i, tot, f"{tot:.0f} J", ha="center", va="bottom", fontsize=9)
    ax[1].set(ylabel="energy (J)", title="(b) Network comm energy")
    ax[1].legend(fontsize=8)
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle("Communication overhead — headless / FODAS / ReLIEF vs optimal fog-head system", y=1.03)
    _save(fig, "fig2_comm_overhead_head_selection")


def fig3_propagation(runs):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for lab, r in runs.items():
        t, p = np.array(r["series"]["t"]), np.array(r["series"]["prop_us"], dtype=float)
        m = ~np.isnan(p)
        # 2 s moving average for readability
        pm = np.convolve(np.nan_to_num(p[m]), np.ones(20) / 20, mode="same")
        ax[0].plot(t[m], pm, label=lab, color=COL[lab], lw=1.6)
    ax[0].set(xlabel="simulated time (s)", ylabel="propagation delay (µs)",
              title="(a) Mean per-tick propagation delay")
    ax[0].legend(fontsize=9)

    data = [runs[k]["rec"]["prop_us"] for k in runs]
    bp = ax[1].boxplot(data, tick_labels=list(runs), showfliers=False, patch_artist=True)
    for patch, k in zip(bp["boxes"], runs):
        patch.set_facecolor(COL[k]); patch.set_alpha(0.6)
    ax[1].set(ylabel="propagation delay (µs)", title="(b) Per-task distribution")
    fig.suptitle("Task-distribution propagation delay by mobility mode", y=1.03)
    _save(fig, "fig3_propagation_delay_mobility")


def fig4_energy(runs):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for lab, r in runs.items():
        ax[0].plot(r["series"]["t"], r["series"]["prop_power_w"], label=lab, color=COL[lab], lw=1.6)
    ax[0].axhline(P_HOVER, ls="--", c="k", lw=0.8)
    ax[0].text(1, P_HOVER + 1, f"hover {P_HOVER:.0f} W", fontsize=8)
    ax[0].set(xlabel="simulated time (s)", ylabel="mean propulsion power (W)",
              title="(a) Per-UAV propulsion power (Zeng19 P(V))")
    ax[0].legend(fontsize=9)

    for lab, r in runs.items():
        ax[1].plot(r["series"]["t"], 100 * np.array(r["series"]["soc"]), label=lab, color=COL[lab], lw=1.6)
    ax[1].set(xlabel="simulated time (s)", ylabel="mean state of charge (%)",
              title="(b) Battery state of charge")
    ax[1].legend(fontsize=9)

    labs = list(runs)
    xs = np.arange(len(labs))
    prop = [runs[k]["E"]["prop"] / 1e3 for k in labs]
    comp = [runs[k]["E"]["comp"] / 1e3 for k in labs]
    comm = [(runs[k]["E"]["net"] + runs[k]["E"]["signal"]) / 1e3 for k in labs]
    ax[2].bar(xs, prop, 0.55, label="propulsion", color=[COL[k] for k in labs])
    ax[2].bar(xs, comp, 0.55, bottom=prop, label="compute", color="#bdbdbd")
    ax[2].bar(xs, comm, 0.55, bottom=np.array(prop) + comp, label="communication", color="#fdc086")
    for x, k in zip(xs, labs):
        tot = (runs[k]["E"]["prop"] + runs[k]["E"]["comp"] + runs[k]["E"]["net"] + runs[k]["E"]["signal"]) / 1e3
        ax[2].text(x, tot, f"{tot:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax[2].set_xticks(xs, labs)
    ax[2].set(ylabel="episode energy (kJ)", title="(c) Total episode energy by component")
    ax[2].legend(fontsize=8)
    fig.suptitle("Energy consumption by UAV mobility mode", y=1.03)
    _save(fig, "fig4_energy_mobility")


def fig5_edge_energy(headless, headmed, fodas, relief):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    fig.subplots_adjust(wspace=0.32)
    labs = ["Random node\n(headless)", "Fog-head\nmediated", "FODAS", "ReLIEF"]
    cs   = [COL["headless"], COL["optimal"], COL["fodas"], COL["relief"]]
    runs4 = (headless, headmed, fodas, relief)

    tot = [r["E"]["edge"] + r["E"]["net"] for r in runs4]
    wst = [r["E"]["edge_wasted"] + r["E"]["net_wasted"] for r in runs4]
    ax[0].bar(labs, tot, 0.5, color=cs, alpha=0.55, label="total upload energy")
    ax[0].bar(labs, wst, 0.5, color="#c02424", label="wasted (failed tasks)")
    for i in range(4):
        ax[0].text(i, tot[i], f"{tot[i]:.0f} J", ha="center", va="bottom", fontsize=9)
    ax[0].set(ylabel="energy (J)", title="(a) Upload energy per episode")
    ax[0].legend(fontsize=8)

    # (b) TOTAL system energy per DELIVERED and ATTEMPTED task:
    # the fleet flies either way, so the metric that matters is how much delivered
    # work each joule of system energy buys.
    def _sys_j(r):
        return (r["E"]["prop"] + r["E"]["comp"] + r["E"]["net"]
                + r["E"]["control"] + r["E"]["signal"] + r["E"]["edge"])
    npd = [_sys_j(r) / max(sum(r["rec"]["success"]), 1) for r in runs4]
    npa = [_sys_j(r) / max(len(r["rec"]["success"]), 1) for r in runs4]
    ax[1].bar(labs, npd, 0.5, color=cs)
    for i, (delivered, attempted) in enumerate(zip(npd, npa)):
        ax[1].text(i, delivered, f"{delivered:.1f} J/del.",
                   ha="center", va="bottom", fontsize=8)
        ax[1].text(i, attempted, f"{attempted:.1f} J/att.",
                   ha="center", va="center", fontsize=8,
                   bbox={"facecolor": "white", "alpha": 0.75,
                         "edgecolor": "none", "pad": 1})
    ax[1].set(ylabel="system energy per task (J)",
              ylim=(0, max(npd) * 1.25),
              title="(b) System energy efficiency\n"
                    "(identical propulsion: paired mobility trace)")

    sr = [100 * r["succ_rate"] for r in runs4]
    ax[2].bar(labs, sr, 0.5, color=cs)
    for i, v in enumerate(sr):
        ax[2].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax[2].set(ylabel="task success rate (%)", ylim=(0, 105), title="(c) Delivery success rate")
    fig.suptitle("Edge→fog task-upload energy: random / FODAS / ReLIEF vs fog-head-mediated distribution", y=1.03)
    _save(fig, "fig5_edge_to_fog_energy")


def _per_device_qos(run):
    """Per-IoT-device mean delay (successful tasks) and success rate."""
    by_dev: Dict[int, List] = {}
    r = run["rec"]
    for dv, d, s in zip(r["dev_id"], r["delay_ms"], r["success"]):
        by_dev.setdefault(int(dv), []).append((d, s))
    mean_delay, succ = [], []
    for dv, rows in by_dev.items():
        ds = [d for d, s in rows if s]
        if ds:
            mean_delay.append(float(np.mean(ds)))
        succ.append(float(np.mean([s for _, s in rows])))
    return mean_delay, succ


def fig6_fairness(runs):
    """
    Uploading/service fairness ACROSS IoT DEVICES — the user-facing fairness that
    motivates coordinated task distribution: with a fair scheduler every device
    receives similar QoS regardless of where it sits, whereas uncoordinated
    policies give geometry-lucky devices much better service than unlucky ones.
    (Raw per-UAV load fairness is deliberately NOT the metric: equalising cycles
    across heterogeneous 1.5–3 GHz nodes is neither feasible nor desirable.)
    Now extended to 3 runs: RL / random / FODAS.
    """
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
    labs = list(runs)
    # Derive per-run colour: fall back to fodas colour for the FODAS key
    _key_to_col = {"fair (attention+PPO)": COL["rl"],
                   "random": COL["rand_sched"],
                   "FODAS": COL["fodas"],
                   "ReLIEF": COL["relief"]}
    cs = [_key_to_col.get(k, COL["fodas"]) for k in labs]

    jd = []; js = []
    for k in labs:
        md, sc = _per_device_qos(runs[k])
        jd.append(jain(md)); js.append(jain([s + 1e-9 for s in sc]))
    xs = np.arange(len(labs)); w = 0.32
    ax[0].bar(xs - w / 2, jd, w, label="Jain over per-device mean delay", color="#4292c6")
    ax[0].bar(xs + w / 2, js, w, label="Jain over per-device success rate", color="#9ecae1")
    for x, (a, b) in enumerate(zip(jd, js)):
        ax[0].text(x - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax[0].text(x + w / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(xs, labs)
    ax[0].set(ylabel="Jain fairness index $J$", ylim=(max(0, min(jd) - 0.05), 1.02),
              title="(a) Service fairness across 200 IoT devices")
    ax[0].legend(fontsize=8, loc="lower right")

    for k, c in zip(labs, cs):
        md, _ = _per_device_qos(runs[k])
        ax[1].plot(np.arange(len(md)), np.sort(md), label=k, color=c, lw=1.7)
    ax[1].set(xlabel="IoT device (sorted by its mean delay)",
              ylabel="per-device mean delay (ms)",
              title="(b) Per-device delay profile (flatter = fairer)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Why coordinated task distribution: per-device uploading/QoS fairness, "
                 "RL / random / FODAS / ReLIEF scheduling", y=1.03)
    _save(fig, "fig6_fairness_index")


def fig7_utilisation(runs):
    """Render only from event-audited utilization-study outputs.

    ``runs`` is retained for call-site compatibility. The former assigned-cycle
    approximation is intentionally unavailable because it clipped overload and
    did not measure CPU occupation.
    """
    summary_path = Path(FIG_DIR) / "fig7_seed_load_policy.csv"
    per_uav_path = Path(FIG_DIR) / "fig7_per_uav.csv"
    if not summary_path.exists() or not per_uav_path.exists():
        print(
            "[FIG7] skipped: run event_training.py and utilisation_study.py "
            "to create event-audited utilization data",
            flush=True)
        return
    from utilisation_study import render_figure7
    with summary_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    with per_uav_path.open(newline="") as fh:
        per_uav = list(csv.DictReader(fh))
    render_figure7(rows, per_uav, Path(FIG_DIR))


# ==============================================================================
# Driver
# ==============================================================================

def _render_all_figures(runs) -> None:
    A1, B1, B2, C1, D1, E1 = runs
    print("[FIGURES]")
    fig1_total_delay(A1, C1, D1, E1)
    fig2_comm_overhead(A1, C1, D1, E1)
    mob = {"event_driven": A1, "random": B2, "static": B1}
    fig3_propagation(mob)
    fig4_energy(mob)
    fig5_edge_energy(C1, A1, D1, E1)
    fig6_fairness({"fair (attention+PPO)": A1, "random": C1,
                   "FODAS": D1, "ReLIEF": E1})
    fig7_utilisation({"attention+PPO": A1, "random": C1,
                      "FODAS": D1, "ReLIEF": E1})


def _checkpoint_signature() -> Dict:
    return {"seed": SEED, "ppo_episodes": N_PRETRAIN,
            "evaluation_tasks_per_run": N_EVAL_TASKS,
            "feature_version": 2}


def _save_training_checkpoint(encoder, actor, critic, optimizer,
                              completed_ppo: int, bc: Dict) -> None:
    _write_torch_atomic(TASK_TRAINING_CHECKPOINT, {
        "signature": _checkpoint_signature(),
        "completed_ppo_episodes": completed_ppo,
        "behavior_cloning": bc,
        "encoder": encoder.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "running_norm": {
            "mu": encoder.running_norm.mu,
            "sigma": encoder.running_norm.sigma,
            "first": encoder.running_norm._first,
        },
    })


def _load_training_checkpoint(encoder, actor, critic, optimizer):
    if not os.path.exists(TASK_TRAINING_CHECKPOINT):
        raise ValueError(
            f"--resume requested but training checkpoint is missing: "
            f"{TASK_TRAINING_CHECKPOINT}")
    try:
        saved = torch.load(TASK_TRAINING_CHECKPOINT, map_location="cpu",
                           weights_only=False)
    except TypeError:  # compatibility with older PyTorch releases
        saved = torch.load(TASK_TRAINING_CHECKPOINT, map_location="cpu")
    if saved.get("signature") != _checkpoint_signature():
        raise ValueError(
            "resume configuration differs from the task-distribution checkpoint")
    encoder.load_state_dict(saved["encoder"])
    actor.load_state_dict(saved["actor"])
    critic.load_state_dict(saved["critic"])
    optimizer.load_state_dict(saved["optimizer"])
    running_norm = saved["running_norm"]
    encoder.running_norm.mu = running_norm["mu"].clone()
    encoder.running_norm.sigma = running_norm["sigma"].clone()
    encoder.running_norm._first = bool(running_norm["first"])
    return (int(saved.get("completed_ppo_episodes", 0)),
            dict(saved.get("behavior_cloning", {})))


def _load_run_checkpoint() -> Dict[str, Dict]:
    if not os.path.exists(TASK_RUNS_CHECKPOINT):
        return {}
    with open(TASK_RUNS_CHECKPOINT, "rb") as fh:
        saved = pickle.load(fh)
    if saved.get("signature") != _checkpoint_signature():
        raise ValueError(
            "resume configuration differs from the evaluation-run checkpoint")
    return dict(saved.get("runs", {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw all figures from the completed run cache")
    parser.add_argument("--resume", action="store_true",
                        help="resume PPO training or evaluation from its last checkpoint")
    parser.add_argument("--pause-after-ppo", type=int,
                        help="checkpoint and stop after this total PPO episode number")
    parser.add_argument("--pause-after-run", type=int,
                        help="checkpoint and stop after this total evaluation run number")
    args = parser.parse_args()
    if args.plot_only and args.resume:
        parser.error("--plot-only and --resume cannot be used together")
    if (args.pause_after_ppo is not None and
            not 1 <= args.pause_after_ppo <= N_PRETRAIN):
        parser.error(f"--pause-after-ppo must be between 1 and {N_PRETRAIN}")
    if (args.pause_after_run is not None and
            not 1 <= args.pause_after_run <= 6):
        parser.error("--pause-after-run must be between 1 and 6")

    print("=" * 70)
    print("  EXPERIMENT HARNESS — paper figures")
    print("=" * 70)

    cache = os.path.join(FIG_DIR, "_runs_cache.pkl")
    if args.plot_only:
        if not os.path.exists(cache):
            parser.error(f"--plot-only requested but cache is missing: {cache}")
        with open(cache, "rb") as fh:
            runs = pickle.load(fh)
        print(f"[CACHE] loaded runs from {cache}")
        _render_all_figures(runs)
        print(f"\nAll figures in {FIG_DIR}")
        return

    global _STOP_AFTER_STAGE
    _STOP_AFTER_STAGE = False
    previous_sigint = signal.signal(signal.SIGINT, _request_task_pause)
    run_started = time.monotonic()
    encoder, actor, critic = AttentionEncoder(), Actor(), Critic()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(actor.parameters()) +
        list(critic.parameters()), lr=LEARNING_RATE)
    _set_mode("event_driven")

    try:
        if args.resume:
            try:
                completed_ppo, bc = _load_training_checkpoint(
                    encoder, actor, critic, optimizer)
                completed_runs = _load_run_checkpoint()
            except ValueError as exc:
                parser.error(str(exc))
            print(f"[RESUME] PPO {completed_ppo}/{N_PRETRAIN}; evaluation "
                  f"{len(completed_runs)}/6", flush=True)
        else:
            completed_ppo = 0
            completed_runs = {}
            _write_pickle_atomic(TASK_RUNS_CHECKPOINT, {
                "signature": _checkpoint_signature(), "runs": {}})
            _write_task_progress(
                status="in_progress", phase="behavior_cloning",
                completed_ppo_episodes=0, completed_evaluation_runs=0,
                training_percent=0.0, evaluation_percent=0.0,
                elapsed_this_run_s=0.0,
                estimated_phase_remaining_s=None)
            print("[BC] FODAS behavior-cloning warm start ...", flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                bc = behavior_clone_fodas(encoder, actor)
            _save_training_checkpoint(
                encoder, actor, critic, optimizer, completed_ppo, bc)
            print(f"  [BC] samples={int(bc['bc_samples'])} "
                  f"batches={int(bc['bc_batches'])} "
                  f"loss={bc['bc_loss']:.3f} | checkpoint saved", flush=True)
            if _STOP_AFTER_STAGE:
                _write_task_progress(
                    status="paused", phase="behavior_cloning_complete",
                    completed_ppo_episodes=0, completed_evaluation_runs=0,
                    training_percent=0.0, evaluation_percent=0.0,
                    elapsed_this_run_s=time.monotonic() - run_started,
                    estimated_phase_remaining_s=None)
                print(f"[PAUSED] Resume with --resume. Progress: "
                      f"{TASK_PROGRESS_PATH}", flush=True)
                return

        print(f"[PRETRAIN] {N_PRETRAIN} offline PPO fine-tuning episodes; "
              f"starting at {completed_ppo + 1 if completed_ppo < N_PRETRAIN else N_PRETRAIN}/"
              f"{N_PRETRAIN}", flush=True)
        training_started = time.monotonic()
        training_done_this_run = 0
        for i in range(completed_ppo, N_PRETRAIN):
            episode_started = time.monotonic()
            set_global_seed(SEED + 1000 + i)
            with contextlib.redirect_stdout(io.StringIO()):
                st = simulation.run_episode(
                    i + 1, actor, critic, BrokerSelector(), encoder,
                    optimizer, train=True)
            completed_ppo = i + 1
            training_done_this_run += 1
            elapsed_training = time.monotonic() - training_started
            mean_episode_s = elapsed_training / training_done_this_run
            eta = mean_episode_s * (N_PRETRAIN - completed_ppo)
            _save_training_checkpoint(
                encoder, actor, critic, optimizer, completed_ppo, bc)
            _write_task_progress(
                status=("in_progress" if completed_ppo < N_PRETRAIN
                        else "training_complete"),
                phase="ppo_training",
                completed_ppo_episodes=completed_ppo,
                completed_evaluation_runs=len(completed_runs),
                training_percent=100.0 * completed_ppo / N_PRETRAIN,
                evaluation_percent=100.0 * len(completed_runs) / 6.0,
                last_episode_duration_s=time.monotonic() - episode_started,
                elapsed_this_run_s=time.monotonic() - run_started,
                estimated_phase_remaining_s=eta)
            print(
                f"  [PT {completed_ppo:02d}/{N_PRETRAIN:02d}] "
                f"drop={st['drop_rate']*100:.2f}% loss={st['total_loss']:.3f} | "
                f"elapsed {_format_duration(elapsed_training)} | "
                f"ETA {_format_duration(eta)} | checkpoint saved",
                flush=True)
            requested_pause = (
                _STOP_AFTER_STAGE or
                (args.pause_after_ppo is not None and
                 completed_ppo >= args.pause_after_ppo))
            if requested_pause:
                _write_task_progress(
                    status="paused", phase="ppo_training",
                    completed_ppo_episodes=completed_ppo,
                    completed_evaluation_runs=len(completed_runs),
                    training_percent=100.0 * completed_ppo / N_PRETRAIN,
                    evaluation_percent=100.0 * len(completed_runs) / 6.0,
                    elapsed_this_run_s=time.monotonic() - run_started,
                    estimated_phase_remaining_s=eta)
                print(f"[PAUSED] Safely stopped after PPO episode "
                      f"{completed_ppo}/{N_PRETRAIN}. Resume with --resume. "
                      f"Progress: {TASK_PROGRESS_PATH}", flush=True)
                return

        nets = (encoder, actor)
        eval_specs = [
            ("A1", "optimal-head/event/rl", "event_driven", "spotis", "rl", True),
            ("B1", "static/spotis/rl", "static", "spotis", "rl", True),
            ("B2", "random-move/spotis/rl", "random", "spotis", "rl", True),
            ("C1", "headless/random", "event_driven", "none", "random", False),
            ("D1", "fodas/edf-heuristic", "event_driven", "none", "fodas", False),
            ("E1", "relief/q-learning", "event_driven", "none", "relief", False),
        ]
        evaluation_started = time.monotonic()
        evaluations_done_this_run = 0
        for index, (key, label, move_mode, head_mode, dispatch,
                    coordinated) in enumerate(eval_specs, start=1):
            if key in completed_runs:
                print(f"[EVAL {index}/6] {label} already checkpointed; skipping",
                      flush=True)
                continue
            _write_task_progress(
                status="in_progress", phase="evaluation",
                current_evaluation_run=index,
                current_evaluation_label=label,
                completed_ppo_episodes=completed_ppo,
                completed_evaluation_runs=len(completed_runs),
                training_percent=100.0,
                evaluation_percent=100.0 * len(completed_runs) / 6.0,
                elapsed_this_run_s=time.monotonic() - run_started,
                estimated_phase_remaining_s=None)
            print(f"[EVAL {index}/6] START {label}; {N_EVAL_TASKS} tasks",
                  flush=True)
            eval_started = time.monotonic()
            completed_runs[key] = run_exp(
                label, move_mode, head_mode, dispatch, coordinated, nets,
                N_EVAL_TASKS)
            evaluations_done_this_run += 1
            elapsed_evaluation = time.monotonic() - evaluation_started
            mean_eval_s = elapsed_evaluation / evaluations_done_this_run
            eta = mean_eval_s * (6 - len(completed_runs))
            _write_pickle_atomic(TASK_RUNS_CHECKPOINT, {
                "signature": _checkpoint_signature(),
                "runs": completed_runs,
            })
            _write_task_progress(
                status=("in_progress" if len(completed_runs) < 6
                        else "evaluation_complete"),
                phase="evaluation",
                last_completed_evaluation_run=index,
                last_completed_evaluation_label=label,
                completed_ppo_episodes=completed_ppo,
                completed_evaluation_runs=len(completed_runs),
                training_percent=100.0,
                evaluation_percent=100.0 * len(completed_runs) / 6.0,
                last_evaluation_duration_s=time.monotonic() - eval_started,
                elapsed_this_run_s=time.monotonic() - run_started,
                estimated_phase_remaining_s=eta)
            print(f"[EVAL {index}/6] COMPLETE {label} | "
                  f"evaluation elapsed {_format_duration(elapsed_evaluation)} | "
                  f"ETA {_format_duration(eta)} | checkpoint saved", flush=True)
            requested_pause = (
                _STOP_AFTER_STAGE or
                (args.pause_after_run is not None and
                 len(completed_runs) >= args.pause_after_run))
            if requested_pause and len(completed_runs) < 6:
                _write_task_progress(
                    status="paused", phase="evaluation",
                    completed_ppo_episodes=completed_ppo,
                    completed_evaluation_runs=len(completed_runs),
                    training_percent=100.0,
                    evaluation_percent=100.0 * len(completed_runs) / 6.0,
                    elapsed_this_run_s=time.monotonic() - run_started,
                    estimated_phase_remaining_s=eta)
                print(f"[PAUSED] Safely stopped after evaluation run "
                      f"{len(completed_runs)}/6. Resume with --resume. "
                      f"Progress: {TASK_PROGRESS_PATH}", flush=True)
                return

        runs = tuple(completed_runs[key] for key, *_ in eval_specs)
        _write_pickle_atomic(cache, runs)
        _write_task_progress(
            status="in_progress", phase="figure_rendering",
            completed_ppo_episodes=N_PRETRAIN,
            completed_evaluation_runs=6,
            training_percent=100.0, evaluation_percent=100.0,
            elapsed_this_run_s=time.monotonic() - run_started,
            estimated_phase_remaining_s=0.0)
        _render_all_figures(runs)
        _write_task_progress(
            status="complete", phase="complete",
            completed_ppo_episodes=N_PRETRAIN,
            completed_evaluation_runs=6,
            training_percent=100.0, evaluation_percent=100.0,
            elapsed_this_run_s=time.monotonic() - run_started,
            estimated_phase_remaining_s=0.0)
        print(f"\nAll figures in {FIG_DIR}")
        print(f"Progress file: {TASK_PROGRESS_PATH}")
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    main()
