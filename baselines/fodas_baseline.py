"""
baselines/fodas_baseline.py
============================
FODAS (Fog-Oriented Dynamic Adaptive Scheduling) deterministic heuristic baseline.

This module implements the *heuristic_fodas* variant of the FODAS scheduler:

  EDF global queue  →  deadline-feasible candidate filtering  →
  completion-time + energy ranking  →  (f_p, f_b) selection

It is consumed exclusively through ``select_node_for_task()``, which is called
inside the ``dispatch == "fodas"`` branch of ``run_exp()`` in experiments.py.

The module is intentionally standalone:
  * It imports only standard-library, NumPy, and project
    constants/models/physics — NOT neural.py, broker.py, or simulation.py.
  * It does NOT run its own episode loop, executor, or metrics writer.
    Execution, queuing, energy accounting, and figure generation all remain
    inside run_exp() / experiments.py so every policy is measured through the
    same pipeline and results are directly comparable.

Paper reference:
    FODAS: A Novel Reinforcement Learning Approach for Efficient Task Scheduling
    in Fog Computing Network.  (See FODAS_A_Novel_...pdf in repo root.)

Deviations from the paper are documented in FODAS_BASELINE_DEVIATIONS_LOG.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project imports (constants + models + physics only — per reuse policy)
# ---------------------------------------------------------------------------
from config import (
    # Scenario constants
    SOC_PRIMARY_MIN, SOC_BACKUP_MIN,
    MAX_QUEUE_DEPTH,
    # Comm / energy constants
    P_COMM,
)
from models import Task, FogNode, IoTDevice
from physics import (
    slant_distance_m,
    data_rate_bps,
    transmission_delay_ms,
    execution_delay_ms,
    queue_delay_ms,
)

# Sentinel returned when no feasible node is found (mirrors neural._NO_NODE = -1)
_NO_NODE: int = -1


# ===========================================================================
# Configuration dataclass
# ===========================================================================

@dataclass
class FODASConfig:
    """
    Tunable knobs for the heuristic FODAS scheduler.

    All defaults reproduce the paper's priority ordering:
      1. Meet deadline first (deadline_first = True).
      2. Among deadline-feasible nodes, minimise estimated completion time.
      3. Break ties on energy cost (lower is better).
    """
    # If True, deadline-feasible candidates are always preferred over
    # non-feasible ones regardless of completion time.
    deadline_first: bool = True

    # Safety margin subtracted from remaining deadline budget when deciding
    # feasibility.  Gives headroom for queue jitter not captured by the
    # static completion-time estimate.
    feasibility_margin_ms: float = 10.0

    # Weight for energy cost in the combined objective.
    # objective = w_time * norm_completion_time + w_energy * norm_energy_cost
    w_time: float = 0.7
    w_energy: float = 0.3


# ===========================================================================
# Candidate dataclass
# ===========================================================================

@dataclass
class FODASCandidate:
    """
    Per-node evaluation result for one (task, node) pair.

    Fields produced by estimate_candidate() and consumed by rank_candidates().
    """
    node_idx: int           # index into the fog list passed to select_node_for_task
    node: FogNode           # reference to the FogNode object

    # Delay sub-terms (ms)
    t_up_ms: float          # uplink transmission delay  IoT → UAV
    t_queue_ms: float       # current queue wait at this node
    t_exec_ms: float        # CPU execution time for this task
    total_ms: float         # estimated end-to-end completion time  (sum above)

    # Energy cost (J) — IoT uplink + UAV receive
    energy_j: float

    # Whether the node can deliver the task before its deadline
    # (total_ms <= remaining_deadline_ms - feasibility_margin)
    meets_deadline: bool

    # Whether the node passes ALL hard-safety filters
    # (SOC ≥ floor AND queue not full AND node is not depleted)
    hard_safe: bool


# ===========================================================================
# Queue / admission helpers
# ===========================================================================

def absolute_deadline_ms(task: Task) -> float:
    """
    Absolute deadline in ms-from-episode-start for a task.

      deadline_abs = arrival_s * 1000 + deadline_ms
    """
    return task.arrival_s * 1000.0 + task.deadline_ms


def admit_task(task: Task, sim_time_s: float) -> bool:
    """
    Task admission check.

    Reject any task whose absolute deadline has already passed at the
    current simulation time.  This mirrors the paper's admission gate
    (§III-B: tasks with d_i < t_now are discarded before scheduling).

    Args:
        task:       The task to check.
        sim_time_s: Current simulator time, seconds.

    Returns True if the task should be scheduled, False if it should be dropped.
    """
    deadline_abs = absolute_deadline_ms(task)
    arrival_abs  = task.arrival_s * 1000.0
    now_ms       = sim_time_s * 1000.0
    # Guard: task must also have a positive time-to-deadline remaining.
    return deadline_abs > arrival_abs and deadline_abs > now_ms


def edf_sort(tasks: List[Task]) -> List[Task]:
    """
    Sort a list of tasks by Earliest Deadline First (ascending absolute deadline).

    Verifies post-condition: resulting deadlines are non-decreasing.

    Args:
        tasks: Unsorted list of Task objects.

    Returns a new list sorted by absolute_deadline_ms ascending.
    """
    sorted_tasks = sorted(tasks, key=absolute_deadline_ms)
    # Sanity check: deadlines are non-decreasing after sort
    assert all(
        absolute_deadline_ms(sorted_tasks[i]) <= absolute_deadline_ms(sorted_tasks[i + 1])
        for i in range(len(sorted_tasks) - 1)
    ), "edf_sort postcondition violated: deadlines not non-decreasing"
    return sorted_tasks


# ===========================================================================
# Candidate evaluation
# ===========================================================================

def estimate_candidate(
    task: Task,
    iot_dev: IoTDevice,
    node: FogNode,
    node_idx: int,
    sim_time_s: float,
    cfg: FODASConfig,
) -> FODASCandidate:
    """
    Compute the estimated completion time and energy cost of assigning ``task``
    to ``node``, as seen from IoT device ``iot_dev``.

    Delay model (same physics as run_exp's critical path):
        T_up    = transmission_delay_ms(task.size_kb, data_rate_bps(dist, iot_dev.p_tx_w))
        T_queue = queue_delay_ms(node)      ← steady-state EDF wait
        T_exec  = execution_delay_ms(task.cycles, node.fr_avg_hz())
        total   = T_up + T_queue + T_exec  (no broker overhead: FODAS is headless)

    Energy model (IoT uplink + UAV receive, in Joules):
        E = iot_dev.p_tx_w * (T_up / 1000)   +   P_COMM * (T_up / 1000)

    Feasibility:
        meets_deadline = total <= remaining_deadline - feasibility_margin

    Hard safety (both must hold for the node to be usable as primary):
        node.soc >= SOC_PRIMARY_MIN
        len(queue_primary) + len(queue_backup) < MAX_QUEUE_DEPTH
    """
    # Geometric distance and uplink rate
    dist_m  = slant_distance_m(node, iot_dev.x, iot_dev.y)
    tr_bps  = data_rate_bps(dist_m, iot_dev.p_tx_w)

    # Delay sub-terms
    t_up_ms    = transmission_delay_ms(task.size_kb, tr_bps)
    t_queue_ms = queue_delay_ms(node)
    t_exec_ms  = execution_delay_ms(task.cycles, node.fr_avg_hz())
    total_ms   = t_up_ms + t_queue_ms + t_exec_ms

    # Energy (IoT uplink TX + UAV receive)
    t_up_s   = t_up_ms / 1000.0
    energy_j = iot_dev.p_tx_w * t_up_s + P_COMM * t_up_s

    # Remaining deadline budget
    now_ms           = sim_time_s * 1000.0
    deadline_abs_ms  = absolute_deadline_ms(task)
    remaining_ms     = deadline_abs_ms - now_ms

    meets_deadline = total_ms <= max(0.0, remaining_ms - cfg.feasibility_margin_ms)

    # Hard-safety gates
    queue_depth = len(node.queue_primary) + len(node.queue_backup)
    hard_safe   = (node.soc >= SOC_PRIMARY_MIN
                   and queue_depth < MAX_QUEUE_DEPTH)

    return FODASCandidate(
        node_idx=node_idx,
        node=node,
        t_up_ms=t_up_ms,
        t_queue_ms=t_queue_ms,
        t_exec_ms=t_exec_ms,
        total_ms=total_ms,
        energy_j=energy_j,
        meets_deadline=meets_deadline,
        hard_safe=hard_safe,
    )


def candidate_is_hard_safe(cand: FODASCandidate) -> bool:
    """
    Return True only if the candidate passes all hard-safety constraints.

    Hard constraints (no soft trade-off allowed):
      1. SOC ≥ SOC_PRIMARY_MIN  — node has enough battery to serve as primary.
      2. Queue not full         — captured inside estimate_candidate via hard_safe field.
    """
    return cand.hard_safe


def rank_candidates(
    candidates: List[FODASCandidate],
    cfg: FODASConfig,
) -> List[FODASCandidate]:
    """
    Rank safe candidates according to the FODAS priority ordering:

    1. If cfg.deadline_first: deadline-feasible candidates come before
       non-feasible ones, regardless of completion time.
    2. Within each feasibility tier, rank by a weighted objective:
         score = w_time * norm_total_ms + w_energy * norm_energy_j
       (lower is better; min-max normalised within the candidate set).
    3. Ties broken by total_ms ascending.

    Only hard-safe candidates are accepted.

    Returns a sorted list (best first).  Empty list if no safe candidates.
    """
    safe = [c for c in candidates if candidate_is_hard_safe(c)]
    if not safe:
        return []

    # Min-max normalise total_ms and energy_j within the safe pool
    times   = [c.total_ms  for c in safe]
    energies = [c.energy_j for c in safe]
    t_min, t_max   = min(times),   max(times)
    e_min, e_max   = min(energies), max(energies)
    t_range = t_max - t_min if t_max > t_min else 1.0
    e_range = e_max - e_min if e_max > e_min else 1.0

    def _score(c: FODASCandidate) -> Tuple:
        # Primary sort key: deadline feasibility (0 = feasible = better if deadline_first)
        tier = 0 if (c.meets_deadline or not cfg.deadline_first) else 1
        # Secondary sort key: weighted normalised objective
        norm_t = (c.total_ms  - t_min) / t_range
        norm_e = (c.energy_j  - e_min) / e_range
        obj    = cfg.w_time * norm_t + cfg.w_energy * norm_e
        # Tertiary: raw total_ms as tie-breaker
        return (tier, obj, c.total_ms)

    return sorted(safe, key=_score)


# ===========================================================================
# Main scheduler entry point  (called by run_exp in experiments.py)
# ===========================================================================

def select_node_for_task(
    task: Task,
    iot_dev: IoTDevice,
    fog_nodes: List[FogNode],
    sim_time_s: float,
    cfg: Optional[FODASConfig] = None,
) -> Tuple[int, int]:
    """
    FODAS heuristic node selector.  Called once per task by run_exp().

    Algorithm:
        1. Admission check: if the task's absolute deadline has already passed,
           return (_NO_NODE, _NO_NODE) — task is dropped before scheduling.
        2. Evaluate each fog node as a candidate (completion time + energy).
        3. Filter to hard-safe candidates (SOC ≥ floor, queue not full).
        4. Rank by FODAS priority: deadline-feasible first, then by weighted
           completion-time / energy objective.
        5. Select the best ranked node as primary (f_p).
        6. Select the second-best as backup (f_b), subject to:
              node.soc >= SOC_BACKUP_MIN  (backup SOC floor is lower)
           If no second candidate qualifies, f_b = _NO_NODE.

    Returns:
        (f_p, f_b) — indices into fog_nodes list.
        Both equal _NO_NODE if no feasible primary exists.
    """
    if cfg is None:
        cfg = FODASConfig()

    # Step 1: admission gate
    if not admit_task(task, sim_time_s):
        return _NO_NODE, _NO_NODE

    # Step 2: evaluate all candidate nodes
    candidates = [
        estimate_candidate(task, iot_dev, node, idx, sim_time_s, cfg)
        for idx, node in enumerate(fog_nodes)
    ]

    # Step 3 + 4: filter hard-unsafe, then rank
    ranked = rank_candidates(candidates, cfg)

    if not ranked:
        # No node is safe: task cannot be scheduled
        return _NO_NODE, _NO_NODE

    # Step 5: primary selection
    primary = ranked[0]
    f_p = primary.node_idx

    # Sanity check: selected primary must not violate hard constraints
    assert candidate_is_hard_safe(primary), (
        f"FODAS bug: selected primary node {f_p} is not hard-safe"
    )

    # Step 6: backup selection — second ranked candidate that also meets the
    # (lower) SOC_BACKUP_MIN threshold.
    f_b = _NO_NODE
    for cand in ranked[1:]:
        if cand.node.soc >= SOC_BACKUP_MIN:
            f_b = cand.node_idx
            break

    return f_p, f_b
