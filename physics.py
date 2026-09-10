"""
physics.py
==========
The analytical models that turn raw state into the quantities every decision
stage consumes:

  * Delay model        (SECTION 8)  — channel gain, data rate, transmission,
                                      execution, queue, broker overhead, total delay.
  * Energy model       (SECTION 9)  — hover/aero power, processing/comm/idle energy,
                                      and the helpers that drain a node's battery.
  * Capability models  (SECTION 10) — processing capability, workload imbalance,
                                      and reliability (base + primary/backup + system).

The trailing _demo_* helpers are illustrative only; they are never called at
import and exist as runnable usage examples (they read the global FOG_SWARM /
IOT_DEVICES / TASK_STREAM built in world.py).
"""

import math
import time
from typing import List, Tuple, Dict

from config import *
from models import Task, FogNode, IoTDevice
from world import FOG_SWARM, IOT_DEVICES, TASK_STREAM

# SECTION 8 — DELAY MODEL  (§2.1)
#
# Total delay for task i served by node j:
#   D_i = T_broker + T_u + T_Q + T_e
#
#   T_broker  — broker pipeline overhead (MEREC + Kalman + KL + optional SPOTIS + PPO infer)
#   T_u       — uplink transmission delay  (Shannon capacity, IoT -> UAV)
#   T_Q       — queue wait at the node under EDF scheduling
#   T_e       — execution (compute) time at the node's CPU
# ==============================================================================

def slant_distance_m(node: "FogNode", gx: float, gy: float, gz: float = 0.0) -> float:
    """
    3-D slant range from a UAV to a ground point (FU-Serve Eq. (10)):

      d = sqrt( (x - gx)^2 + (y - gy)^2 + (z - gz)^2 )

    IoT devices sit at gz = 0; UAV altitude z is fixed per node (0 in 'static'
    mode, so static reproduces the original 2-D geometry exactly).  Every link
    quantity downstream (channel gain, data rate, delay, comm energy, reliability)
    consumes this distance, which is how movement propagates through the pipeline
    (UAV_Mobility_Design.pdf Fig. 4).  Guard: minimum 1 m.
    """
    return max(math.sqrt((node.x - gx) ** 2 + (node.y - gy) ** 2 + (node.z - gz) ** 2), 1.0)


def uav_uav_distance_m(a: "FogNode", b: "FogNode") -> float:
    """3-D distance between two UAVs (fog-head relay leg). Guard: minimum 1 m."""
    return max(math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2), 1.0)


def channel_gain(dist_m: float) -> float:
    """
    UAV air-to-ground LoS channel gain: g = upsilon_1 * d^(-upsilon_2).

    Units: dimensionless (ratio).
    upsilon_1 = (lambda/4*pi)^2 (free-space gain at 1 m, carrier 2 GHz),
    upsilon_2 = 2.3 (LoS air-to-ground path-loss exponent).  See §2.1 for the
    link-budget rationale.  Guard: minimum distance 1 m prevents division-by-zero.
    """
    dist_m = max(dist_m, 1.0)                           # enforce minimum 1 m separation
    return CHANNEL_UPSILON1 * (dist_m ** (-CHANNEL_UPSILON2))  # g = upsilon_1 * d^(-2)


def data_rate_bps(dist_m: float, p_tx_w: float) -> float:
    """
    Uplink Shannon capacity: TR = B * log2(1 + g * P_tx / N_0).

    B       — channel bandwidth, Hz
    g       — channel gain (dimensionless)
    P_tx    — IoT transmit power, W
    N_0     — thermal noise power, W
    Returns: data rate in bits per second (bps).

    The SNR term g*P_tx/N_0 captures both path loss and transmit power [Zeng19 §II].
    """
    g   = channel_gain(dist_m)                          # dimensionless channel gain
    snr = g * p_tx_w / NOISE_W                          # received SNR (dimensionless)
    return BANDWIDTH_HZ * math.log2(1.0 + snr)          # Shannon rate, bps


def transmission_delay_ms(size_kb: float, TR_bps: float) -> float:
    """
    Uplink transmission delay: T_u = (size_kb * 1024 * 8) / TR_bps.

    size_kb * 1024   — payload in bytes
    * 8              — convert bytes -> bits
    / TR_bps         — divide by channel rate to get time in seconds
    * 1000           — convert seconds -> ms

    Returns: T_u in milliseconds.
    """
    payload_bits = size_kb * 1024.0 * 8.0               # bits
    return (payload_bits / TR_bps) * 1000.0              # ms


def execution_delay_ms(cycles: float, fr_avg_hz: float) -> float:
    """
    CPU execution time: T_e = cycles / fr_avg_hz.

    cycles     — total CPU cycles required for this task (precomputed in Task.generate)
    fr_avg_hz  — node's mean CPU frequency, Hz
    * 1000     — convert seconds -> ms

    Returns: T_e in milliseconds.
    """
    return (cycles / fr_avg_hz) * 1000.0                 # ms


def queue_delay_ms(node: "FogNode") -> float:
    """
    Average queue wait under dual-queue EDF scheduling.

    Scheduling policy (non-preemptive EDF):
      - backup queue has absolute priority over primary queue
        (a backup task activates only when the primary fails; it must execute first)
      - within each queue tasks are ordered Earliest-Deadline-First (EDF)
      - the unified execution order is: backup queue (EDF) followed by primary queue (EDF)

    Wait formula (steady-state work-conserving approximation):
      Let the unified order have n tasks with cycle counts l_1 .. l_n.
      The k-th task in the order must wait for all tasks ahead of it to finish:
        wait_k = sum_{j=1}^{k-1} l_j / fr_avg_hz
      Average wait across all n tasks:
        T_Q_avg = (1 / (n * fr_avg_hz)) * sum_{k=1}^{n-1} (n - k) * l_k
      (The (n-k) factor counts how many later tasks each task l_k blocks.)

    Returns 0 if the queue has 0 or 1 tasks (no waiting possible).
    Returns: T_Q_avg in milliseconds.
    """
    state = getattr(node, "cpu_state", None)
    engine = getattr(state, "_engine", None) if state is not None else None
    if engine is not None:
        return 1000.0 * engine.projected_edf_delay_s(
            node.node_id, deadline_s=float("inf"), backup=False)

    fr_hz = node.fr_avg_hz()                             # CPU frequency, Hz

    # Sort each sub-queue by absolute deadline (EDF = smallest deadline first).
    # queue_primary / queue_backup entries are (Task, deadline_abs_ms) tuples.
    backup_sorted  = sorted(node.queue_backup,  key=lambda t: t[1])  # EDF on backup
    primary_sorted = sorted(node.queue_primary, key=lambda t: t[1])  # EDF on primary

    # Unified order: backup tasks execute before primary tasks (non-preemptive priority).
    order = backup_sorted + primary_sorted               # list of (Task, deadline_abs_ms)
    n     = len(order)

    if n <= 1:
        return 0.0                                       # 0 or 1 task => no queue wait

    # Extract cycle counts from the ordered task list.
    cycles_list = [entry[0].cycles for entry in order]  # l_k for k = 0..n-1

    # Accumulate: task at position k waits for sum of cycles of tasks 0..k-1.
    # Equivalently: each task l_k contributes l_k * (n - 1 - k) to the total wait numerator.
    # (We use 0-based indexing: position k has n-1-k tasks after it, so blocks them.)
    total_wait_cycles = 0.0
    for k in range(n - 1):                               # last task blocks nobody
        total_wait_cycles += cycles_list[k] * (n - 1 - k)

    # Average wait = total_wait_cycles / (n * fr_hz), then convert s -> ms.
    return (total_wait_cycles / (n * fr_hz)) * 1000.0   # ms


def broker_overhead_ms(spotis_ran: bool) -> float:
    """
    Total broker pipeline overhead per task.

    Components (all in ms):
      T_MEREC   — MEREC weight computation  (event-triggered)
      T_KALMAN  — Kalman smoother update    (event-triggered)
      T_KL      — KL divergence gate test   (event-triggered)
      T_SPOTIS  — SPOTIS full re-rank       (only when KL gate fires => spotis_ran=True)
      T_PPO_INFER — PPO actor forward pass  (once per task)

    Returns: total overhead in milliseconds.
    """
    overhead = T_PPO_INFER_MS
    if spotis_ran:
        overhead += T_MEREC_MS + T_KALMAN_MS + T_KL_MS + T_SPOTIS_MS
    return overhead                                      # ms


def total_delay_ms(
    task:       "Task",
    node:       "FogNode",
    dist_m:     float,
    p_tx_w:     float,
    spotis_ran: bool,
) -> Dict[str, float]:
    """
    Full end-to-end delay for task i served by node j.

    D_i = T_broker + T_u + T_Q + T_e    (all terms in ms)

    Returns a dict with keys:
      'broker_ms', 'tx_ms', 'queue_ms', 'exec_ms', 'total_ms'
    so callers can print every sub-term with its label and unit.
    """
    TR_bps    = data_rate_bps(dist_m, p_tx_w)           # uplink rate, bps

    broker_ms = broker_overhead_ms(spotis_ran)           # ms — pipeline decision overhead
    tx_ms     = transmission_delay_ms(task.size_kb, TR_bps)  # ms — uplink transmission
    queue_ms  = queue_delay_ms(node)                     # ms — EDF queue wait at node
    exec_ms   = execution_delay_ms(task.cycles, node.fr_avg_hz())  # ms — CPU execution

    total_ms  = broker_ms + tx_ms + queue_ms + exec_ms  # ms — end-to-end delay

    return {
        "broker_ms": broker_ms,
        "tx_ms":     tx_ms,
        "queue_ms":  queue_ms,
        "exec_ms":   exec_ms,
        "total_ms":  total_ms,
    }


# ------------------------------------------------------------------------------
# §8 — DEBUG DEMO  (set condition to True temporarily to inspect delay breakdown)
# ------------------------------------------------------------------------------
def _demo_delay_model() -> None:
    """
    Print the delay breakdown for a 150 KB task on a Tier-2 node at 1000 m.
    Not called in normal execution; toggle the `if False` guard to run.
    """
    import dataclasses

    # Construct a synthetic Tier-2 node (fr_avg = 2.30 GHz) with an empty queue.
    demo_node = FogNode(
        node_id=99, tier=2, fr_avg_ghz=2.30,
        MP_tot_gb=4.0, MS_tot_gb=6.0, MP_occ_gb=0.0, MS_occ_gb=0.0,
        E_initial_j=198_000.0, E_res_j=198_000.0,
        lambda_fail=0.015, mu_fail=0.005, varsigma=4, C_mips=15_000,
        x=0.0, y=0.0,
    )

    # Construct a synthetic 150 KB task (medium-small, within small threshold).
    demo_task = Task(
        task_id=0, size_kb=150.0,
        cycles=150.0 * 1024 * CYCLES_PER_BYTE,
        deadline_ms=deadline_for_size_ms(150.0),
        arrival_s=0.0, is_small=True,
    )

    dist_m  = 1000.0         # 1 km IoT-to-UAV separation
    p_tx_w  = P_TX_LOW_W     # low-power sensor

    breakdown = total_delay_ms(
        task=demo_task, node=demo_node,
        dist_m=dist_m, p_tx_w=p_tx_w,
        spotis_ran=True,         # assume SPOTIS ran this tick
    )

    TR = data_rate_bps(dist_m, p_tx_w)
    g  = channel_gain(dist_m)

    print("\n[DEMO §8] Delay breakdown — 150 KB task, Tier-2 node, 1000 m distance")
    print(f"  Channel gain g          : {g:.4e}  (dimensionless)")
    print(f"  SNR                     : {g * p_tx_w / NOISE_W:.2f}  (dimensionless)")
    print(f"  Data rate TR            : {TR/1e6:.3f}  Mbps")
    print(f"  Broker overhead T_b     : {breakdown['broker_ms']:.3f}  ms")
    print(f"  Transmission delay T_u  : {breakdown['tx_ms']:.3f}  ms")
    print(f"  Queue delay T_Q         : {breakdown['queue_ms']:.3f}  ms  (empty queue)")
    print(f"  Execution delay T_e     : {breakdown['exec_ms']:.3f}  ms")
    print(f"  TOTAL delay D_i         : {breakdown['total_ms']:.3f}  ms")
    print(f"  Deadline                : {demo_task.deadline_ms:.1f}  ms")
    print(f"  Feasible (D < deadline) : {breakdown['total_ms'] < demo_task.deadline_ms}")



# ==============================================================================
# SECTION 9 — ENERGY MODEL  (§2.2)
#
# Per-tick energy budget for one UAV:
#   E_tick = P_HOVER * dt            (hover propulsion — always on while airborne)
#          + P_PROC  * t_exec        (compute power while processing assigned tasks)
#          + 2*P_COMM * t_comm       (receive + forward at broker link)
#          + P_IDLE  * t_idle        (avionics baseline — charged separately if needed)
#
# All drain functions clamp E_res_j >= 0 so SOC property stays in [0, 1].
# ==============================================================================

def hover_power_w() -> float:
    """
    Return total rotary-wing hover power: P_HOVER = P_0 + P_i = 870.52 W [Zeng19 Eq.(3)].

    Endurance note:
      Tier 1 (E_initial = 234 000 J): 234000 / 870.52 ≈ 269 s  (~4.5 min pure hover)
      Tier 2 (E_initial = 198 000 J): 198000 / 870.52 ≈ 227 s  (~3.8 min)
      Tier 3 (E_initial = 162 000 J): 162000 / 870.52 ≈ 186 s  (~3.1 min)

    Over one 50 s episode hover alone drains:
      Tier 1: 870.52 * 50 / 234000 ≈ 18.6% SOC
      Tier 2: 870.52 * 50 / 198000 ≈ 22.0% SOC
      Tier 3: 870.52 * 50 / 162000 ≈ 26.9% SOC

    This means energy-aware routing has real consequence within a single episode — a UAV
    can lose 19–27% SOC from hover alone before any task processing is counted.
    The RL agent must therefore account for energy even on short missions [Zeng19].
    """
    return P_HOVER   # W; constant in hover (V=0) operating state


def aero_power_w(V_ms: float) -> float:
    """
    Full rotary-wing aerodynamic power at forward speed V m/s [Zeng19 Eq.(3)].

      P(V) = P_0 * (1 + 3*V^2 / U_tip^2)                     (blade profile term)
           + P_i * sqrt( sqrt(1 + V^4/(4*v_0^4)) - V^2/(2*v_0^2) )  (induced term)
           + 0.5 * d_0 * rho * A * V^3                         (parasitic drag term)

    Operating state is hover (V = 0), where this reduces to:
      P(0) = P_0 * 1 + P_i * 1 + 0 = P_HOVER  ✓

    Called every control tick via drain_flight_energy with the node's current
    speed (world.step_mobility): hover and flight share this single curve, so a
    'static' run (V = 0 everywhere) charges exactly P_HOVER per tick — the
    original energy model — while moving modes ride the induced-power dip
    (P(12 m/s) ≈ 779 W < P_HOVER; UAV_Mobility_Design.pdf Fig. 1).
    """
    # Blade profile power grows with V^2 (faster rotation needed to maintain lift).
    blade = P_BLADE * (1.0 + 3.0 * V_ms**2 / U_TIP**2)

    # Induced power: decreases with speed as inflow angle changes.
    ratio   = V_ms**2 / (2.0 * MEAN_ROTOR_VELOCITY**2)              # V^2 / (2*v_0^2)
    induced = P_INDUCED * math.sqrt(math.sqrt(1.0 + ratio**2) - ratio)

    # Parasitic (fuselage drag) power grows as V^3.
    parasitic = 0.5 * DRAG_RATIO * AIR_DENSITY * ROTOR_DISC_AREA * V_ms**3

    return blade + induced + parasitic   # total aerodynamic power, W


def processing_energy_j(node: FogNode, cycles: float) -> float:
    """
    Energy consumed by the on-board CPU executing a task.

      E_proc = P_PROC * (cycles / fr_avg_hz)

    P_PROC   — processing power draw, W  (constant; 10 W typical embedded compute [Mao17])
    cycles   — CPU cycles needed for this task
    / fr_hz  — gives execution time in seconds
    Returns: Joules.
    """
    t_exec_s = cycles / node.fr_avg_hz()   # execution duration, s
    return P_PROC * t_exec_s               # J = W * s


def comm_energy_j(node: FogNode, size_kb: float, TR_bps: float) -> float:
    """
    Energy consumed by radio communication for one task payload.

      E_comm = 2 * P_COMM * (size_kb * 1024 * 8 / TR_bps)

    Factor 2: accounts for both the uplink RECEIVE leg (IoT -> UAV fog node)
    and the forward/relay leg (fog node -> broker or cloud), matching the
    dual-hop broker architecture described in §2.1.
    P_COMM   — transceiver active power, W  (2 W; typical low-power SDR [Zeng19])
    Returns: Joules.
    """
    payload_bits = size_kb * 1024.0 * 8.0          # bits
    t_comm_s     = payload_bits / TR_bps            # transmission duration, s
    return 2.0 * P_COMM * t_comm_s                  # J; factor 2 for receive + forward


def idle_energy_j(t_idle_s: float) -> float:
    """
    Avionics baseline energy during idle time (no task, no active radio).

      E_idle = P_IDLE * t_idle_s

    P_IDLE   — microcontroller + sensor bus standby power, W  (3 W)
    Returns: Joules.
    """
    return P_IDLE * t_idle_s   # J


def drain_hover_energy(node: FogNode, dt_s: float) -> None:
    """
    Subtract propulsion energy for one control tick from node's residual energy.

      delta_E = P_HOVER * dt_s

    Called once per UAV per control tick (dt_s = CONTROL_TICK_S = 0.1 s) in the main loop.
    Clamps E_res_j at 0 — a UAV that runs out of battery stays at SOC=0 (grounded).
    Mutates node in-place; returns nothing (SOC readable via node.soc property).
    """
    node.E_res_j = max(0.0, node.E_res_j - P_HOVER * dt_s)   # J; clamp at 0


def drain_flight_energy(node: FogNode, dt_s: float) -> None:
    """
    Subtract propulsion energy for one control tick at the node's CURRENT speed:

      delta_E = P(V) * dt_s,   P(V) = aero_power_w(node.speed_ms)  [Zeng19 Eq. (3)]

    Generalises drain_hover_energy: at V = 0 this charges exactly P_HOVER * dt_s
    (same curve, no discontinuity — UAV_Mobility_Design.pdf §1), so 'static' runs
    reproduce the original hover-only energy budget.  node.speed_ms is set each
    tick by world.step_mobility.  Clamps E_res_j at 0; mutates node in place.
    """
    node.E_res_j = max(0.0, node.E_res_j - aero_power_w(node.speed_ms) * dt_s)


def apply_task_energy(node: FogNode, cycles: float, size_kb: float, TR_bps: float) -> None:
    """
    Subtract compute + communication energy incurred by processing one task.

      delta_E = E_proc + E_comm

    Called once per task routed to this node (in addition to per-tick hover drain).
    Clamps E_res_j at 0.
    Mutates node in-place.
    """
    e_proc = processing_energy_j(node, cycles)              # J — CPU execution cost
    e_comm = comm_energy_j(node, size_kb, TR_bps)           # J — radio cost (rx + forward)
    node.E_res_j = max(0.0, node.E_res_j - e_proc - e_comm) # J; clamp at 0


# ==============================================================================
# SECTION 10 — CAPABILITY, WORKLOAD, AND RELIABILITY MODELS  (§2.3 – §2.5)
# ==============================================================================

# ------------------------------------------------------------------------------
# §2.3  Memory efficiency (Stage-1 criterion) and computational efficiency (tiering)
# ------------------------------------------------------------------------------

def task_memory_footprint_gb(size_kb: float) -> Tuple[float, float]:
    """
    Memory a single task holds while resident in a node's queue.

      ram_gb     = max(MEM_RAM_FLOOR_GB,     size_kb * MEM_RAM_GB_PER_KB)
      storage_gb = max(MEM_STORAGE_FLOOR_GB, size_kb * MEM_STORAGE_GB_PER_KB)

    A queued task stages its (decompressed) working set in RAM and persists its
    payload + intermediate results to storage; both scale with payload size, with
    a small per-task floor for runtime/container overhead.

    Returns: (ram_gb, storage_gb).
    """
    ram_gb     = max(MEM_RAM_FLOOR_GB,     size_kb * MEM_RAM_GB_PER_KB)
    storage_gb = max(MEM_STORAGE_FLOOR_GB, size_kb * MEM_STORAGE_GB_PER_KB)
    return ram_gb, storage_gb


def update_memory_occupancy(fog_nodes: List[FogNode]) -> None:
    """
    Recompute MP_occ_gb / MS_occ_gb for every node from the tasks currently
    resident in its primary + backup queues.

    This is what makes memory efficiency a LIVE, load-coupled criterion: as a node
    accumulates queued tasks its free memory falls, and when the per-tick queue
    pruning (run_episode §5 cleanup) evicts deadline-passed tasks its free memory
    recovers.  Occupancy is clamped to the node's physical capacity so free memory
    never goes negative.

    Called once per control tick (in the Stage-1 block) BEFORE memory_efficiency.
    Mutates each node in-place; returns nothing.
    """
    for node in fog_nodes:
        state = getattr(node, "cpu_state", None)
        engine = getattr(state, "_engine", None) if state is not None else None
        if engine is not None:
            node.MP_occ_gb, node.MS_occ_gb = engine.memory_occupancy(node.node_id)
            continue
        ram = 0.0
        sto = 0.0
        for (task, _deadline) in node.queue_primary:
            r, s = task_memory_footprint_gb(task.size_kb)
            ram += r
            sto += s
        for (task, _deadline) in node.queue_backup:
            r, s = task_memory_footprint_gb(task.size_kb)
            ram += r
            sto += s
        node.MP_occ_gb = min(ram, node.MP_tot_gb)   # cannot exceed physical RAM
        node.MS_occ_gb = min(sto, node.MS_tot_gb)   # cannot exceed physical storage


def memory_efficiency(fog_nodes: List[FogNode]) -> Dict[int, float]:
    """
    Compute the Stage-1 memory-efficiency criterion ME in [0, 1] for every node.

      ME_j = clip( [ CAP_ALPHA*(MP_tot - MP_occ) + CAP_BETA*(MS_tot - MS_occ) ]
                   / ME_NORM_GB ,  0, 1 )

    ME captures BOTH capacity and occupancy:
      * capacity  — MP_tot / MS_tot vary per node, so ME discriminates across nodes
                    even with empty queues (a larger-memory drone scores higher);
      * occupancy — MP_occ / MS_occ come from update_memory_occupancy, so ME falls
                    as a node fills and rises as its queue drains.

    Higher ME = more free memory = better (beneficial criterion).  Normalising by
    ME_NORM_GB (max free memory of the strongest node) keeps ME in [0, 1], so the
    SPOTIS col-1 bound stays [0, 1].  Replaces the former processing_capability.

    Returns: dict mapping node_id -> ME in [0, 1].
    """
    result: Dict[int, float] = {}
    for node in fog_nodes:
        free_gb = (CAP_ALPHA * (node.MP_tot_gb - node.MP_occ_gb)
                   + CAP_BETA * (node.MS_tot_gb - node.MS_occ_gb))
        result[node.node_id] = min(1.0, max(0.0, free_gb / ME_NORM_GB))
    return result


def computational_efficiency(fog_nodes: List[FogNode]) -> Dict[int, float]:
    """
    Computational efficiency CE_eff per node — the quantity used to FORM tiers
    (see config.computational_efficiency_score and world.build_fog_swarm).

      CE_eff_j = fr_hz_j / (varsigma_j * C_mips_j * 1e6)   (higher = better)

    Exposed here for inspection / verification only; it is NOT a Stage-1 criterion.
    Returns: dict mapping node_id -> CE_eff.
    """
    return {
        node.node_id: computational_efficiency_score(
            node.fr_avg_ghz, node.varsigma, node.C_mips
        )
        for node in fog_nodes
    }


# ------------------------------------------------------------------------------
# §2.4  Workload model
# ------------------------------------------------------------------------------

def node_workload_cycles(node: FogNode) -> float:
    """
    Total pending CPU cycles queued at a node (primary + backup queues combined).

      W_j = sum of task.cycles for all tasks in queue_primary and queue_backup

    This per-node W_j is used as an input FEATURE to the attention encoder — it
    tells the RL agent how busy each node currently is.
    (Distinct from the scalar WL used in the reward; see system_workload_imbalance.)
    """
    state = getattr(node, "cpu_state", None)
    engine = getattr(state, "_engine", None) if state is not None else None
    if engine is not None:
        return engine.remaining_workload(node.node_id)
    total = sum(task.cycles for task, _deadline in node.queue_primary)
    total += sum(task.cycles for task, _deadline in node.queue_backup)
    return total


def system_workload_imbalance(fog_nodes: List[FogNode]) -> Tuple[Dict[int, float], float]:
    """
    Compute per-node workload and the scalar imbalance metric WL.

      W_j     = node_workload_cycles(node_j)                      (cycles per node)
      W_bar   = mean(W_j) over all nodes                          (mean cycles)
      WL      = sum_j |W_j - W_bar|                               (total absolute deviation)

    Two distinct objects are returned so callers use the right one for the right purpose:
      per_node_dict  — fed as a feature into the self-attention encoder for each node
      WL (scalar)    — fed into the PPO reward function as the workload-imbalance term

    Keeping them separate avoids accidentally summing a per-node vector into a scalar reward.
    Returns: (per_node_dict mapping node_id -> W_j cycles, WL scalar).
    """
    per_node: Dict[int, float] = {n.node_id: node_workload_cycles(n) for n in fog_nodes}

    w_bar = sum(per_node.values()) / len(fog_nodes)   # mean workload across swarm, cycles

    wl = sum(abs(w - w_bar) for w in per_node.values())   # total absolute deviation, cycles

    return per_node, wl                                    # (dict, scalar)


# ------------------------------------------------------------------------------
# §2.5  Reliability model
# ------------------------------------------------------------------------------

def base_reliability(node: FogNode, cycles: float, size_kb: float, TR_bps: float) -> float:
    """
    Single-node task-completion probability R0 under independent compute and link failures.

      R0 = exp( -lambda_fail * t_exec  -  mu_fail * t_comm )

    Where:
      t_exec  = cycles / fr_avg_hz          (compute duration, s)
      t_comm  = size_kb*1024*8 / TR_bps     (transmission duration, s)

    This is the survival probability of two independent Poisson failure processes
    running in parallel: one for compute failures (rate lambda_fail) and one for
    link failures (rate mu_fail).  The exponents add because the processes are
    independent, and survival requires BOTH to survive.

    Returns R0 in (0, 1].
    """
    t_exec_s = cycles / node.fr_avg_hz()                     # s
    t_comm_s = (size_kb * 1024.0 * 8.0) / TR_bps            # s

    exponent = (- node.lambda_fail * t_exec_s               # compute-failure contribution
                - node.mu_fail    * t_comm_s)                # link-failure contribution
    return math.exp(exponent)                                 # R0 in (0, 1]


def head_operational_availability(node: FogNode, horizon_s: float) -> float:
    """Probability that an elected fog head remains available for a horizon.

    ``base_reliability`` exposes a node to failure only for one task's
    millisecond-scale execution and transmission time. A fog head is instead an
    active coordinator throughout its tenure. Under the same independent
    exponential compute and radio failure processes, its interval availability
    is ``exp(-(lambda_fail + mu_fail) * horizon_s)``.

    The horizon must be reported with the value; the fog-head study uses the
    configured 10-second interval.
    """
    if horizon_s < 0.0:
        raise ValueError("availability horizon must be non-negative")
    return math.exp(-(node.lambda_fail + node.mu_fail) * horizon_s)


def primary_backup_reliability(R0_p: float, R0_b: float) -> float:
    """
    Combined reliability of a primary + backup UAV pair serving the same task.

      R = 1 - (1 - R0_p) * (1 - R0_b)

    This is the "at-least-one-survives" formula for two statistically independent
    nodes — the task succeeds if either the primary OR the backup completes it.
    Primary and backup MUST be different nodes (independence assumption holds only
    when they are distinct physical UAVs on separate links and CPUs).

    Returns R in [0, 1]; always >= max(R0_p, R0_b).
    """
    return 1.0 - (1.0 - R0_p) * (1.0 - R0_b)   # combined survival probability


def mean_system_reliability(reliabilities: List[float]) -> float:
    """
    Arithmetic mean of per-task reliability values over an episode.

      R_mean = (1/N) * sum_i R_i

    SECONDARY operator-facing metric: the average per-task success probability.
    NOTE: this is NOT ReLIEF Eq.(12).  Eq.(12) defines system reliability as the
    PRODUCT prod_i R_i ("ALL tasks succeed" semantics); the mean is forgiving of
    a single catastrophic failure that the product correctly flags.  The faithful
    Eq.(12) aggregate is log10_system_reliability() below, accumulated in log
    space so it never underflows.  Both are reported; the log-domain product is
    the headline system-reliability figure.

    Returns the mean, or 0.0 for an empty list.
    """
    if not reliabilities:
        return 0.0
    return sum(reliabilities) / len(reliabilities)   # arithmetic mean in [0, 1]


def log10_system_reliability(reliabilities: List[float]) -> float:
    """
    ReLIEF Eq.(12) system reliability, accumulated exactly in log space:

      log10(R_sys) = log10( prod_i R_i ) = sum_i log10(R_i)

    Summing logs is mathematically identical to the paper's product but cannot
    underflow: over 2,500 tasks at R_i ~ 0.95 the raw product is ~4.7e-56
    (numerically useless), while the log form is simply ~-55.3 and stays
    informative — it decreases linearly, not exponentially, as tasks accrue.
    This mirrors how ReLIEF itself reports Eq.(12): Fig. 6 plots reliability as
    a "number of nines", i.e. a log transform of the product, never the raw value.

    Each R_i is clamped below at RELIABILITY_LOG_CLAMP (1e-12) because dropped
    tasks record R_i = 0.0 exactly and log10(0) = -inf; the clamp converts each
    outright failure into a finite -12 contribution (see config.py).

    Returns sum_i log10(R_i) <= 0.0, or 0.0 for an empty list (empty product = 1).
    """
    if not reliabilities:
        return 0.0
    return sum(math.log10(max(r, RELIABILITY_LOG_CLAMP)) for r in reliabilities)


def geometric_mean_reliability(reliabilities: List[float]) -> float:
    """
    Per-task-normalized form of Eq.(12):  R_geo = (prod_i R_i)^(1/N)
                                                = 10^( log10(R_sys) / N ).

    Stays in [0, 1] and is comparable across episodes with different task counts,
    while PRESERVING the product's AND semantics: one catastrophic R_i drags the
    geometric mean down far harder than it dents the arithmetic mean.  Computed
    via the log-domain sum, so it inherits the same underflow immunity.

    Returns R_geo in [0, 1], or 0.0 for an empty list.
    """
    if not reliabilities:
        return 0.0
    return 10.0 ** (log10_system_reliability(reliabilities) / len(reliabilities))


def system_reliability_nines(log10_R: float) -> float:
    """
    Convert log10(R_sys) into the "number of nines" that ReLIEF Fig. 6 uses on
    its y-axis:  nines = -log10(1 - R_sys),  R_sys = 10^log10_R.

      R_sys = 0.9    -> 1 nine        R_sys = 0.999 -> 3 nines
      R_sys ~ 0      -> ~0 nines (an unreliable system)

    Uses expm1 for a numerically stable (1 - 10^x) when log10_R is within
    rounding of 0 (a near-perfect episode), and caps the result at 15 nines —
    beyond double-precision resolution of (1 - R), and far above any physically
    meaningful reliability claim.

    Returns nines >= 0.0.
    """
    if log10_R >= 0.0:
        return 15.0                                       # perfect product: cap, don't return inf
    one_minus_R = -math.expm1(log10_R * math.log(10.0))   # 1 - 10^log10_R, stable near 0
    if one_minus_R <= 1e-15:
        return 15.0
    return max(0.0, -math.log10(one_minus_R))


# ------------------------------------------------------------------------------
# §10 — DEBUG DEMO
# ------------------------------------------------------------------------------
def _demo_capability_reliability() -> None:
    """
    Print PC for all nodes and R0/R for a sample primary-backup pair.
    Not called in normal execution; toggle the `if False` guard to run.
    """
    update_memory_occupancy(FOG_SWARM)            # reflect any queued tasks (none at startup)
    me_dict = memory_efficiency(FOG_SWARM)
    ce_dict = computational_efficiency(FOG_SWARM)

    print("\n[DEMO §10a] Memory efficiency ME (Stage-1) and CE_eff (tiering) per node:")
    for node in FOG_SWARM:
        print(
            f"  Node {node.node_id:02d} Tier-{node.tier} "
            f"fr={node.fr_avg_ghz:.2f} GHz  "
            f"ME={me_dict[node.node_id]:.3f}  CE_eff={ce_dict[node.node_id]:.4f}"
        )

    # Sample primary = first Tier-1 node, backup = first Tier-3 node.
    primary = next(n for n in FOG_SWARM if n.tier == 1)
    backup  = next(n for n in FOG_SWARM if n.tier == 3)

    # Use a 150 KB task at 1000 m with low-power IoT for demo.
    cycles  = 150.0 * 1024 * CYCLES_PER_BYTE
    size_kb = 150.0
    TR_bps  = data_rate_bps(1000.0, P_TX_LOW_W)

    R0_p = base_reliability(primary, cycles, size_kb, TR_bps)
    R0_b = base_reliability(backup,  cycles, size_kb, TR_bps)
    R    = primary_backup_reliability(R0_p, R0_b)

    print(f"\n[DEMO §10b] Reliability — 150 KB task at 1000 m:")
    print(f"  Primary  Node {primary.node_id:02d} Tier-{primary.tier}: R0_p = {R0_p:.6f}")
    print(f"  Backup   Node {backup.node_id:02d}  Tier-{backup.tier}: R0_b = {R0_b:.6f}")
    print(f"  Combined R (at-least-one)     : R   = {R:.6f}")
    print(f"  Meets RELIABILITY_FLOOR {RELIABILITY_FLOOR}: {R >= RELIABILITY_FLOOR}")

    # Workload demo (all queues empty at startup).
    per_node, wl = system_workload_imbalance(FOG_SWARM)
    print(f"\n[DEMO §10c] Workload (empty queues): WL = {wl:.1f} cycles, "
          f"all W_j = {set(per_node.values())}")
