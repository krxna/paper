"""
baselines/relief_baseline.py
=============================
ReLIEF tabular Q-learning primary-backup task-assignment baseline.

This module implements ReLIEF as the paper describes it: a broker-side
Q-learning agent that, for each arriving task, selects a (primary, backup)
fog-node pair. State = discretized (system reliability, workload balance).
Reward follows paper Eq.(21); Q-update follows Eq.(1); action selection
follows Bellman's principle of optimality Eq.(2). The backup is *delayed*:
it only runs when the primary's result is not received in time (this
project's existing run_exp() delayed-backup path already implements that
semantics — see experiments.py's dispatch=="relief" branch).

It is consumed exclusively through:
  * ``pretrain_agent()``  — builds and cold-start-trains a ReLIEFAgent in a
    throwaway simulated world (paper Phase 1).
  * ``ReLIEFAgent.select_pair()`` / ``ReLIEFAgent.observe()`` — called once
    per task inside the ``dispatch == "relief"`` branch of ``run_exp()`` in
    experiments.py (paper Phase 2, runtime).

The module is intentionally standalone:
  * It imports only standard-library, NumPy, and project
    constants/models/world/physics — NOT neural.py, broker.py, or
    simulation.py.
  * It does NOT run its own episode loop, executor, or metrics writer at
    evaluation time. Execution, queuing, energy accounting, and figure
    generation all remain inside run_exp() / experiments.py so every policy
    is measured through the same pipeline and results are directly
    comparable. (pretrain_agent() runs its OWN lightweight throwaway loop
    only to fill the Q-table before evaluation — this mirrors the paper's
    cold-start fix and produces no metrics or figures.)

Paper reference:
    ReLIEF: A Reinforcement-Learning-Based Real-Time Task Assignment
    Strategy in Emerging Fault-Tolerant Fog Computing. IEEE IoT-J, vol. 10,
    no. 12, 2023. (See ReLIEF_A_Reinforcement-Learning-Based_...pdf in repo
    root.)

Deviations from the paper are documented in RELIEF_BASELINE_DEVIATIONS_LOG.md.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project imports (constants + models + world + physics only — per reuse policy)
# ---------------------------------------------------------------------------
from config import (
    SOC_PRIMARY_MIN, SOC_BACKUP_MIN,
    MAX_QUEUE_DEPTH,
    TASK_SIZE_MIN_KB, TASK_SIZE_MAX_KB,
    CYCLES_PER_BYTE_MIN, CYCLES_PER_BYTE_MAX,
    RELIABILITY_LOG_CLAMP,
    SEED, set_global_seed,
)
from models import Task, FogNode, IoTDevice
from execution_engine import exact_queue_depth
from world import (
    build_fog_swarm, build_iot_devices, generate_task_stream,
    assign_mobility, step_mobility,
)
from physics import (
    slant_distance_m, data_rate_bps, uav_uav_distance_m,
    transmission_delay_ms, execution_delay_ms, queue_delay_ms,
    processing_energy_j,
    base_reliability, primary_backup_reliability,
    system_reliability_nines,
    system_workload_imbalance,
)

# Sentinel returned when no feasible node is found (mirrors neural._NO_NODE = -1)
_NO_NODE: int = -1

# Nominal per-node workload target (cycles): a node carrying half a full
# queue (MAX_QUEUE_DEPTH/2) of nominal-size tasks is "balanced". Fixed
# rather than derived from the live (possibly-empty) queue state, so the
# Eq.(21) WL_k denominator never collapses toward zero early in an episode.
_NOM_KB = 0.5 * (TASK_SIZE_MIN_KB + TASK_SIZE_MAX_KB)
_NOM_CPB = 0.5 * (CYCLES_PER_BYTE_MIN + CYCLES_PER_BYTE_MAX)
_NOM_CYCLES = _NOM_KB * 1024.0 * _NOM_CPB
WL_TARGET_CYCLES: float = _NOM_CYCLES * (MAX_QUEUE_DEPTH / 2.0)


# ===========================================================================
# Configuration dataclass
# ===========================================================================

@dataclass
class ReLIEFConfig:
    """
    Tunable knobs for the ReLIEF Q-learning agent.

    alpha/gamma: standard Q-learning step size and discount (Eq.1).
    epsilon:     linearly decayed from eps_start to eps_end over
                 eps_decay_steps observed transitions (paper does not fix
                 a schedule; this is the project deviation logged in
                 RELIEF_BASELINE_DEVIATIONS_LOG.md).
    rho1/rho2/rho3: reward weights from paper Table IV (WL / D / R terms
                 of Eq.21).
    n_rel_buckets/n_wl_buckets: state discretization resolution (paper
                 §IV-B1: "actual state values are mapped to discrete
                 values" — bucket counts are a project choice).
    rel_nines_cap: clip system-reliability-in-nines to this range before
                 bucketing (nines grows unboundedly as tasks accrue).
    target_reliability: R_K in Eq.(21) — the reliability level the reward
                 is scored against.
    pretrain_episodes / pretrain_tasks: size of the cold-start warm-up
                 (paper Phase 1) run inside pretrain_agent().
    """
    alpha: float = 0.1
    gamma: float = 0.9
    eps_start: float = 0.3
    eps_end: float = 0.01
    eps_decay_steps: int = 20_000

    rho1: float = 0.36   # workload-balance weight (paper Table IV)
    rho2: float = 0.27   # delay weight
    rho3: float = 0.29   # reliability weight

    n_rel_buckets: int = 5
    n_wl_buckets: int = 5
    rel_nines_cap: float = 6.0
    target_reliability: float = 0.99
    n_backup_candidates: int = 5   # nearest-K backup pruning; see _feasible_actions

    # Decay for the exponential moving average of log10(R_i) that tracks
    # "current" system reliability (see update_reliability_log_ema below).
    # Project deviation: the paper's Eq.(12) product is a whole-episode
    # aggregate metric, not a per-step state signal; an unbounded running
    # product collapses toward log10(R)->-inf (0 nines) after only a few
    # dozen tasks and would make the reliability half of the state useless
    # for the rest of the episode. The EMA keeps a bounded, responsive
    # "recent reliability" signal instead, at decay ~50-task memory.
    rel_ema_decay: float = 0.98

    pretrain_episodes: int = 5
    pretrain_tasks: int = 4000


# ===========================================================================
# State discretization
# ===========================================================================

def discretize_state(rel_nines: float, wl_cycles: float, wl_scale: float,
                      cfg: ReLIEFConfig) -> Tuple[int, int]:
    """
    Map continuous (system reliability, workload imbalance) onto a small
    discrete state s_k = (r_bucket, wl_bucket), per paper §IV-B1.

    rel_nines: system_reliability_nines() of the running episode reliability
               product — higher is better, clipped to [0, cfg.rel_nines_cap].
    wl_cycles: system_workload_imbalance() scalar WL (total abs deviation,
               cycles) — lower is better.
    wl_scale:  a nominal WL scale (e.g. mean per-node workload) used to
               normalise wl_cycles into [0, 1] before bucketing.
    """
    r = max(0.0, min(cfg.rel_nines_cap, rel_nines))
    r_bucket = min(cfg.n_rel_buckets - 1,
                   int(r / cfg.rel_nines_cap * cfg.n_rel_buckets))

    wl_norm = wl_cycles / wl_scale if wl_scale > 0 else 0.0
    wl_norm = max(0.0, min(1.0, wl_norm))
    wl_bucket = min(cfg.n_wl_buckets - 1, int(wl_norm * cfg.n_wl_buckets))

    return r_bucket, wl_bucket


def update_reliability_log_ema(prev_log10_ema: float, r_task: float,
                                cfg: ReLIEFConfig) -> float:
    """
    Exponential moving average of log10(R_i) across observed tasks — the
    "current system reliability" signal fed into discretize_state() via
    system_reliability_nines(). Bounded and responsive, unlike an unbounded
    running log10-sum (which represents the whole-episode product and
    monotonically collapses to 0 nines after a few dozen tasks — see
    ReLIEFConfig.rel_ema_decay).

    R = 10**ema approximates the geometric-mean reliability of the most
    recent ~1/(1-decay) tasks.
    """
    log_r = math.log10(max(r_task, RELIABILITY_LOG_CLAMP))
    return cfg.rel_ema_decay * prev_log10_ema + (1.0 - cfg.rel_ema_decay) * log_r


# ===========================================================================
# Q-learning agent
# ===========================================================================

@dataclass
class ReLIEFAgent:
    """
    Tabular Q-learning broker: state = (reliability bucket, workload
    bucket); action = (primary_idx, backup_idx) fog-node pair.

    q is a plain dict keyed by (state, action) -> float, lazily initialised
    to 0.0 (optimistic-neutral init). Deterministic under a fixed seed
    because action sampling uses the module-level `random` RNG that
    set_global_seed() seeds.
    """
    cfg: ReLIEFConfig
    q: Dict[Tuple, float] = field(default_factory=lambda: defaultdict(float))
    steps: int = 0

    def _epsilon(self) -> float:
        t = min(1.0, self.steps / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + t * (self.cfg.eps_end - self.cfg.eps_start)

    @staticmethod
    def _feasible_nodes(fog: List[FogNode], soc_floor: float) -> List[int]:
        return [i for i, n in enumerate(fog)
                if n.soc >= soc_floor
                and n.available
                and exact_queue_depth(n) < MAX_QUEUE_DEPTH]

    def _feasible_actions(self, fog: List[FogNode]) -> List[Tuple[int, int]]:
        """
        Feasible (primary, backup) pairs, plus (primary, _NO_NODE).

        Backup candidates per primary are pruned to the cfg.n_backup_candidates
        nearest feasible nodes (by inter-UAV distance, the same metric the
        harness uses for backup-forwarding delay in run_exp). Project
        deviation: the paper's action space A is already every (primary,
        backup) node pair, but its own experiments use 5-15 fog nodes; this
        project's N_FOG=20 makes the full cross-product (~380 pairs) too
        large for a tabular Q-table to cover within a practical pretrain
        budget. Pruning to nearby backups is the same physically-motivated
        top-K restriction the main methodology's own candidate filtering
        applies (neural.filter_candidates) — logged in
        RELIEF_BASELINE_DEVIATIONS_LOG.md.
        """
        primaries = self._feasible_nodes(fog, SOC_PRIMARY_MIN)
        backups = self._feasible_nodes(fog, SOC_BACKUP_MIN)
        actions: List[Tuple[int, int]] = []
        for p in primaries:
            actions.append((p, _NO_NODE))
            cands = [b for b in backups if b != p]
            cands.sort(key=lambda b: uav_uav_distance_m(fog[p], fog[b]))
            for b in cands[: self.cfg.n_backup_candidates]:
                actions.append((p, b))
        return actions

    def select_pair(self, fog: List[FogNode], task: Task, dev: IoTDevice,
                     state: Tuple[int, int], greedy: bool = True) -> Tuple[int, int]:
        """
        Bellman-optimal action selection, Eq.(2): pi(s) = argmax_a Q(s,a).

        When greedy=False (pretrain), epsilon-greedy exploration is used so
        the Q-table covers the action space before evaluation runs greedy.
        Ties among equal-Q feasible actions are broken uniformly at random.
        """
        actions = self._feasible_actions(fog)
        if not actions:
            return _NO_NODE, _NO_NODE

        if not greedy and random.random() < self._epsilon():
            return random.choice(actions)

        best_q = max(self.q[(state, a)] for a in actions)
        best = [a for a in actions if self.q[(state, a)] == best_q]
        return random.choice(best)

    def reward(self, wl_after: float, wl_scale: float,
               d_ms: float, deadline_ms: float,
               r_task: float) -> float:
        """
        Reward Eq.(21): r = rho1*(delta_WL/WL_k) + rho2*(delta_D/D_ik)
                          + rho3*(delta_R/R_K)

        Project-scale targets (logged in the deviations log):
          WL_k  = wl_scale        — nominal per-node workload (cycles)
          D_ik  = deadline_ms     — this task's own deadline (paper
                                    subscript i on D matches per-task
                                    normalisation)
          R_K   = cfg.target_reliability

        delta_WL = wl_scale - wl_after      (positive: WL below nominal load
                                              -> well balanced -> rewarded)
        delta_D  = deadline_ms - d_ms       (positive: finished under
                                              deadline -> rewarded)
        delta_R  = r_task - target_reliability (positive: task reliability
                                              exceeds the target -> rewarded)
        """
        delta_wl = (wl_scale - wl_after) / wl_scale if wl_scale > 0 else 0.0
        delta_d = (deadline_ms - d_ms) / deadline_ms if deadline_ms > 0 else 0.0
        delta_r = (r_task - self.cfg.target_reliability) / self.cfg.target_reliability

        return (self.cfg.rho1 * delta_wl
                + self.cfg.rho2 * delta_d
                + self.cfg.rho3 * delta_r)

    def observe(self, state: Tuple[int, int], action: Tuple[int, int],
                reward_value: float, next_state: Tuple[int, int],
                fog: List[FogNode]) -> None:
        """
        Q-update Eq.(1):
          Q'(s,a) = Q(s,a) + alpha*(r + gamma*max_a' Q(s',a') - Q(s,a))
        """
        next_actions = self._feasible_actions(fog)
        max_next_q = max((self.q[(next_state, a)] for a in next_actions), default=0.0)

        key = (state, action)
        self.q[key] += self.cfg.alpha * (
            reward_value + self.cfg.gamma * max_next_q - self.q[key]
        )
        self.steps += 1


# ===========================================================================
# Candidate delay/reliability estimate (mirrors fodas_baseline.estimate_candidate)
# ===========================================================================

def _estimate_pair(task: Task, dev: IoTDevice, fog: List[FogNode],
                    f_p: int, f_b: int) -> Tuple[float, float]:
    """
    Estimate (completion_time_ms, pair_reliability) for assigning `task`
    to primary fog[f_p] with optional backup fog[f_b], from IoT device dev.

    Delay model matches run_exp's critical path (no broker overhead —
    ReLIEF runs headless, like FODAS): T_up + T_queue + T_exec.
    Reliability follows paper Eq.(7)-(11): base_reliability() gives R0 per
    node, primary_backup_reliability() combines primary + backup into the
    primary-backup pair reliability R_i.
    """
    prim = fog[f_p]
    dist_m = slant_distance_m(prim, dev.x, dev.y)
    tr_bps = data_rate_bps(dist_m, dev.p_tx_w)
    t_up_ms = transmission_delay_ms(task.size_kb, tr_bps)
    t_queue_ms = queue_delay_ms(prim)
    t_exec_ms = execution_delay_ms(task.cycles, prim.fr_avg_hz())
    total_ms = t_up_ms + t_queue_ms + t_exec_ms

    r0_p = base_reliability(prim, task.cycles, task.size_kb, tr_bps)
    if f_b != _NO_NODE:
        bak = fog[f_b]
        dist_b = slant_distance_m(bak, dev.x, dev.y)
        tr_b = data_rate_bps(dist_b, dev.p_tx_w)
        r0_b = base_reliability(bak, task.cycles, task.size_kb, tr_b)
        r_pair = primary_backup_reliability(r0_p, r0_b)
    else:
        r_pair = r0_p

    return total_ms, r_pair


# ===========================================================================
# Cold-start pretraining (paper Phase 1)
# ===========================================================================

def pretrain_agent(
    seed: int = SEED,
    cfg: Optional[ReLIEFConfig] = None,
    *,
    world_state: Optional[
        Tuple[List[FogNode], List[IoTDevice], List[List[float]]]
    ] = None,
) -> ReLIEFAgent:
    """
    Fill the Q-table in a throwaway simulated world before evaluation, to
    avoid ReLIEF's cold-start issue (paper §IV-B, "the model's outputs
    become less reliable and more unpredictable with time" if launched with
    an untrained table). Runs cfg.pretrain_episodes independent episodes of
    cfg.pretrain_tasks tasks each, epsilon-greedy exploring.

    The fog swarm is built ONCE with `seed` — the same seed run_exp() uses
    (set_global_seed(SEED) then build_fog_swarm()) — and reused unchanged
    across every pretrain episode, only resetting queues/battery between
    episodes. This matters: node indices are not physically meaningful
    across different random worlds (build_fog_swarm() randomly assigns
    hardware quality per index), so a Q-table keyed on (state, node-index
    pair) trained on one world and evaluated on another world would be
    learning noise. Only the task stream and IoT device draws vary between
    pretrain episodes (via reseeding per episode without rebuilding fog).

    No plotting, no metrics output, no interaction with run_exp() — this
    function's only side effect is returning a warmed-up ReLIEFAgent.
    Deterministic under `seed`.
    """
    if cfg is None:
        cfg = ReLIEFConfig()

    agent = ReLIEFAgent(cfg=cfg)
    # EMA of log10(R_i) — see update_reliability_log_ema(). Starts at 0.0
    # (R=1, optimistic prior) and settles within ~1/(1-decay) tasks.
    log10_ema = 0.0

    if world_state is None:
        set_global_seed(seed)
        fog = build_fog_swarm()
        iot, centres = build_iot_devices()
        assign_mobility(fog, centres)
    else:
        fog, iot, centres = world_state

    for ep in range(cfg.pretrain_episodes):
        set_global_seed(seed + 1000 + ep)   # vary traffic only, not the swarm
        for n in fog:                       # fresh battery + empty queues each episode
            n.E_res_j = n.E_initial_j
            n.queue_primary = []
            n.queue_backup = []
        tasks = generate_task_stream(cfg.pretrain_tasks)

        control_tick_s = 0.1
        sim_t, next_idx = 0.0, 0
        n_tasks = len(tasks)

        while next_idx < n_tasks:
            sim_t += control_tick_s
            tick_end_ms = sim_t * 1000.0
            step_mobility(fog, centres, control_tick_s)

            tick_tasks: List[Task] = []
            while next_idx < n_tasks and tasks[next_idx].arrival_s * 1000.0 < tick_end_ms:
                tick_tasks.append(tasks[next_idx]); next_idx += 1

            for task in tick_tasks:
                dev = iot[random.randint(0, len(iot) - 1)]

                _per_node, wl_before = system_workload_imbalance(fog)
                nines_before = system_reliability_nines(log10_ema)
                state = discretize_state(nines_before, wl_before, WL_TARGET_CYCLES, cfg)

                f_p, f_b = agent.select_pair(fog, task, dev, state, greedy=False)
                if f_p == _NO_NODE:
                    continue

                d_ms, r_pair = _estimate_pair(task, dev, fog, f_p, f_b)
                log10_ema = update_reliability_log_ema(log10_ema, r_pair, cfg)

                deadline_abs = task.arrival_s * 1000.0 + task.deadline_ms
                prim = fog[f_p]
                prim.queue_primary.append((task, deadline_abs))
                prim.E_res_j = max(0.0, prim.E_res_j - processing_energy_j(prim, task.cycles))
                if f_b != _NO_NODE:
                    bak = fog[f_b]
                    bak.queue_backup.append((task, deadline_abs))

                _per_node, wl_after = system_workload_imbalance(fog)
                nines_after = system_reliability_nines(log10_ema)
                next_state = discretize_state(nines_after, wl_after, WL_TARGET_CYCLES, cfg)

                r = agent.reward(wl_after, WL_TARGET_CYCLES, d_ms, task.deadline_ms, r_pair)
                agent.observe(state, (f_p, f_b), r, next_state, fog)

            cutoff = sim_t * 1000.0
            for n in fog:
                n.queue_primary = [(t, d) for (t, d) in n.queue_primary if d >= cutoff]
                n.queue_backup = [(t, d) for (t, d) in n.queue_backup if d >= cutoff]

    return agent
