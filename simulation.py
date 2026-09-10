"""
simulation.py
=============
The runtime that ties Stage 1 and Stage 2 together:

  * Executor       (SECTION 15) — functional_survival, dispatch_task (the
                                  delayed-backup protocol), and episode_drop_rate.
  * Episode runner (SECTION 16) — run_episode, the 100 ms control-tick main loop
                                  that elects a broker, routes each arriving task,
                                  drains energy, collects experience, and (when
                                  training) applies the PPO update.
"""

import math
import time
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch

from config import *
from models import Task, FogNode, IoTDevice
from world import (
    FOG_SWARM, IOT_DEVICES, IOT_CENTRES, TASK_STREAM,
    build_fog_swarm, build_iot_devices, generate_task_stream,
    assign_mobility, step_mobility,
)
from physics import *
from broker import *
from neural import *
# Names starting with "_" are not pulled in by "import *", so import them explicitly.
from neural import _NO_NODE, build_candidate_features, filter_candidates

# SECTION 15 — EXECUTOR  (§5)
#
# The executor translates routing decisions (f_p, f_b) into realized outcomes.
# It enforces two scheduling invariants that match §2.4 queue_delay_ms:
#   1. Backup queue has ABSOLUTE NON-PREEMPTIVE PRIORITY over primary queue.
#      A backup task that activates jumps to the front of f_b's execution order,
#      ahead of any waiting primary tasks.
#   2. Within each queue, tasks are EDF-ordered (earliest absolute deadline first).
#
# The DELAYED BACKUP DISPATCH rule is the key efficiency mechanism:
#   - The backup slot is HELD (reserved) but NOT committed until the primary is
#     confirmed to have failed or will miss its deadline.
#   - This avoids burning f_b's energy and queue capacity on every task; in practice
#     the primary succeeds > 99% of the time (high R0), so most backups are dropped.
# ==============================================================================

def functional_survival(node: FogNode, exposure_time_s: float) -> bool:
    """
    Draw a binary survival outcome for a node over a given exposure window.

    Survival probability follows from the same independent Poisson failure model
    used to compute base_reliability (§2.5):

      P_survive = exp( -(lambda_fail + mu_fail) * exposure_time_s )

    lambda_fail — per-second compute-failure rate
    mu_fail     — per-second link-failure rate

    Both failure processes run simultaneously, so their rates ADD in the exponent.
    This couples the theoretical reliability R0 (§2.5) to the realized binary
    outcome: E[functional_survival] == R0 in expectation, ensuring the reliability
    metric is not just a design score but a reflection of actual success frequency.

    Returns True if the node survives (task can be completed), False otherwise.
    """
    combined_rate  = node.lambda_fail + node.mu_fail       # total failure rate, 1/s
    p_survive      = math.exp(-combined_rate * exposure_time_s)   # survival probability in (0,1]
    return random.random() < p_survive                     # Bernoulli draw


def dispatch_task(
    task:        Task,
    f_p:         int,               # primary node index into fog_nodes list
    f_b:         int,               # backup node index (_NO_NODE if no backup)
    fog_nodes:   List[FogNode],
    dist_p_m:    float,             # IoT-to-primary distance, m
    dist_b_m:    float,             # IoT-to-backup distance, m (ignored if f_b == _NO_NODE)
    p_tx_w:      float,             # IoT transmit power, W
    spotis_ran:  bool,              # whether SPOTIS ran this tick (affects broker_overhead_ms)
) -> Dict:
    """
    Execute the delayed-backup dispatch protocol for one task.

    Protocol (mirrors the spec §5 broker rule):

      Step 1 — Primary dispatch:
        Compute D_primary (full pipeline delay via total_delay_ms).
        Push task onto f_p.queue_primary at its EDF position.
        Draw primary survival over the execution exposure window.

      Step 2 — Backup decision (delayed):
        If D_primary <= deadline AND primary survives:
          => Primary succeeded on time. Delete the held backup slot (never dispatch f_b).
        Else:
          => Primary missed deadline OR failed. Dispatch f_b if available.
             Check whether remaining time budget covers f_b's expected path:
               T_remaining = deadline - D_primary (clamped >= 0)
               Required for f_b: T_u(f_b) + T_Q_avg(f_b) + T_e(f_b)  [no broker overhead again]
             If T_remaining >= required: dispatch f_b, draw its survival.
             Else: backup arrives too late; task is counted as failed.

      Step 3 — Success determination:
        success = 1 if primary delivered on time AND survived, OR backup delivered on time AND survived.

      Step 4 — Energy accounting:
        Apply task energy to each node that ACTUALLY executed (primary always; backup only if dispatched).
        Hover energy is drained per-tick by drain_hover_energy (not here).

    Queue management note:
      Backup tasks are pushed onto queue_BACKUP (not queue_primary), giving them absolute
      non-preemptive priority. Both queues are EDF-ordered; queue_delay_ms enforces this
      in the delay model (see §2.4). We do not re-sort here — the queue is maintained in
      insertion order and sorted only when queue_delay_ms is called.

    Returns a dict:
      success      : int  0 or 1
      used_backup  : bool True if backup was actually dispatched
      D_primary_ms : float end-to-end delay of primary path, ms
      D_backup_ms  : float end-to-end delay of backup path, ms (None if not dispatched)
      R_i          : float realized combined reliability (R0_p alone, or primary_backup_reliability)
      primary_id   : int node_id of primary
      backup_id    : int node_id of backup (or _NO_NODE)
      dropped      : bool True if task failed (no successful path within deadline)
    """
    primary = fog_nodes[f_p]

    TR_p = data_rate_bps(dist_p_m, p_tx_w)   # uplink rate for primary path, bps

    # ---- Step 1: Primary path ----
    delay_p = total_delay_ms(
        task       = task,
        node       = primary,
        dist_m     = dist_p_m,
        p_tx_w     = p_tx_w,
        spotis_ran = spotis_ran,
    )
    D_primary_ms = delay_p['total_ms']           # ms — full end-to-end primary delay

    # Push task onto primary's EDF queue (appended; sorted by deadline_ms on lookup).
    deadline_abs = task.arrival_s * 1000.0 + task.deadline_ms   # absolute deadline, ms from epoch
    primary.queue_primary.append((task, deadline_abs))

    # Exposure time = how long the primary is occupied (queue wait + execution).
    exposure_p_s = (delay_p['queue_ms'] + delay_p['exec_ms']) / 1000.0   # s

    # Draw primary survival over its exposure window.
    p_survived = functional_survival(primary, exposure_p_s)

    # Reliability score for the primary-only path.
    R0_p = base_reliability(primary, task.cycles, task.size_kb, TR_p)

    # Apply energy consumed by primary (regardless of survival: energy is already spent).
    apply_task_energy(primary, task.cycles, task.size_kb, TR_p)

    # ---- Step 2: Delayed backup decision ----
    used_backup  = False
    D_backup_ms  = None
    b_survived   = False
    R0_b         = 0.0   # default: no backup contribution to reliability

    primary_on_time = D_primary_ms <= task.deadline_ms

    if primary_on_time and p_survived:
        # Primary succeeded on schedule — delete held backup, save f_b's resources.
        pass   # backup never dispatched; this is the common case (~>99% of tasks)

    elif f_b != _NO_NODE:
        # Primary failed or will miss deadline — try backup.
        backup  = fog_nodes[f_b]
        TR_b    = data_rate_bps(dist_b_m, p_tx_w)

        # Time budget remaining for the backup path.
        # T_remaining = slack after primary's delay (may be negative = already past deadline).
        T_remaining_ms = max(0.0, task.deadline_ms - D_primary_ms)

        # Expected backup path cost (transmission + queue + execution; NO broker overhead,
        # since the broker decision was already paid by the primary dispatch).
        T_u_b  = transmission_delay_ms(task.size_kb, TR_b)         # ms
        T_Q_b  = queue_delay_ms(backup)                             # ms
        T_e_b  = execution_delay_ms(task.cycles, backup.fr_avg_hz()) # ms
        T_backup_needed_ms = T_u_b + T_Q_b + T_e_b                  # ms

        if T_remaining_ms >= T_backup_needed_ms:
            # Backup can still meet the deadline — dispatch it now.
            used_backup = True

            # Push onto backup queue (ABSOLUTE PRIORITY over primary queue in EDF scheduling).
            backup.queue_backup.append((task, deadline_abs))

            delay_b = total_delay_ms(
                task       = task,
                node       = backup,
                dist_m     = dist_b_m,
                p_tx_w     = p_tx_w,
                spotis_ran = False,   # broker overhead NOT charged again for backup leg
            )
            D_backup_ms = delay_b['total_ms']   # ms

            exposure_b_s = (delay_b['queue_ms'] + delay_b['exec_ms']) / 1000.0
            b_survived   = functional_survival(backup, exposure_b_s)

            R0_b = base_reliability(backup, task.cycles, task.size_kb, TR_b)

            # Apply energy to backup node for its actual execution.
            apply_task_energy(backup, task.cycles, task.size_kb, TR_b)

    # ---- Step 3: Success ----
    primary_success = p_survived and primary_on_time
    backup_success  = used_backup and b_survived and (D_backup_ms is not None) and (D_backup_ms <= task.deadline_ms)
    success         = 1 if (primary_success or backup_success) else 0

    # Combined reliability for this task (at-least-one formulation from §2.5).
    R_i = primary_backup_reliability(R0_p, R0_b) if used_backup else R0_p

    return {
        "success":      success,
        "used_backup":  used_backup,
        "D_primary_ms": round(D_primary_ms, 3),
        "D_backup_ms":  round(D_backup_ms, 3) if D_backup_ms is not None else None,
        "R_i":          round(R_i, 6),
        "primary_id":   primary.node_id,
        "backup_id":    f_b if f_b != _NO_NODE else _NO_NODE,
        "dropped":      success == 0,
    }


def episode_drop_rate(outcomes: List[Dict]) -> float:
    """
    Fraction of dispatched tasks that failed (missed deadline or both nodes failed).

      drop_rate = 1 - mean(success_i)   over all tasks with a dispatch decision

    Tasks with f_p == _NO_NODE (no feasible primary at all) are also counted as
    dropped; they are added to outcomes with success=0 by the main loop before
    calling this function.

    Returns value in [0, 1].
    """
    if not outcomes:
        return 0.0
    return 1.0 - sum(o['success'] for o in outcomes) / len(outcomes)


# ==============================================================================
# SECTION 16 — EPISODE RUNNER  (§5 main loop)
# ==============================================================================

def _legacy_run_episode(
    episode_idx: int,
    actor:       Actor,
    critic:      Critic,
    broker:      BrokerSelector,
    encoder:     AttentionEncoder,
    optimizer:   torch.optim.Optimizer,
    train:       bool = True,
) -> Dict:
    """
    Run one complete simulation episode: TASKS_PER_EPISODE tasks over ~500 control ticks.

    Episode anatomy:
      - Fresh world (fog_nodes, iot, tasks) each episode so UAVs start fully charged
        and IoT cluster positions are re-sampled; this prevents overfitting to one topology.
      - The BrokerSelector (broker) carries its Kalman state IN from the caller; the caller
        decides whether to reset it between episodes.
      - The encoder, actor, and critic are shared across episodes (weights accumulate).

    Control flow per tick:
      (1) Drain hover energy for every UAV (P_HOVER * CONTROL_TICK_S J per node).
      (2) Stage 1: compute per_node_metrics, run broker.select_head (MEREC→Kalman→KL→SPOTIS).
      (3) Collect tasks whose Poisson arrival_s falls in [tick_start, tick_end).
      (4) For each task: encode → select_action → dispatch → reward → buffer.
      (5) Queue cleanup: evict tasks whose deadline has passed (they can't succeed now).
      (6) Apply print policy (PRINT_EVERY throttle + force-print exceptions).

    After all tasks are processed:
      - If train=True: compute GAE advantages, run PPO update.
      - Compute and return episode summary statistics.
    """

    # ----------------------------------------------------------------
    # (A) Setup — fresh world per episode
    # ----------------------------------------------------------------
    fog_nodes = build_fog_swarm()              # 20 UAVs, full charge, new random positions
    iot, iot_centres = build_iot_devices()     # 200 IoT devices, 4-cluster Gaussian layout
    tasks     = generate_task_stream(TASKS_PER_EPISODE)  # 2500 Poisson-arrival tasks

    # Mobility init (UAV_Mobility_Design.pdf): hotspot/ring-slot assignment and
    # altitude layers ('static' keeps z = 0 and never moves — original behaviour).
    assign_mobility(fog_nodes, iot_centres)

    # Set module training modes.
    # encoder.train() enables RunningNorm EMA updates during rollout collection.
    # All modules are in train mode so gradient computation works later in ppo_update.
    if train:
        encoder.train(); actor.train(); critic.train()
    else:
        encoder.eval();  actor.eval();  critic.eval()

    # Population-weighted mean transmit power: used as a representative p_tx
    # when computing MCDM metrics (avoids picking a specific IoT group).
    mean_p_tx: float = (N_LOW_POWER_IOT * P_TX_LOW_W + N_HIGH_POWER_IOT * P_TX_HIGH_W) / N_IOT

    def _refresh_geometry() -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Per-node mean 3-D slant range to all IoT devices and the representative
        data rate derived from it.  Called once at setup and again every tick in
        moving modes (PDF Fig. 2 step 2 — positions changed, so every distance-
        derived quantity must be refreshed).  'static' mode computes it once,
        matching the original episode-level baseline exactly (z = 0 there).
        """
        md = {
            n.node_id: max(float(np.mean([slant_distance_m(n, d.x, d.y) for d in iot])), 1.0)
            for n in fog_nodes
        }
        tr = {n.node_id: data_rate_bps(md[n.node_id], mean_p_tx) for n in fog_nodes}
        return md, tr

    mean_iot_dist, tr_repr = _refresh_geometry()

    # Nominal task for MCDM representative metrics (midpoint of payload range).
    _nom_kb: float = 0.5 * (TASK_SIZE_MIN_KB + TASK_SIZE_MAX_KB)   # 430 KB
    nominal_task = Task(
        task_id=-1, size_kb=_nom_kb,
        cycles=_nom_kb * 1024.0 * CYCLES_PER_BYTE,
        deadline_ms=deadline_for_size_ms(_nom_kb), arrival_s=0.0, is_small=False,
    )

    # ----------------------------------------------------------------
    # (B) Initialize previous-step metrics for reward deltas
    #     At episode start: empty queues (WL=0), nominal delay and reliability.
    # ----------------------------------------------------------------
    _nom_d_vals = []
    _nom_r_vals = []
    for node in fog_nodes:
        TR = tr_repr[node.node_id]
        _nom_d_vals.append(
            total_delay_ms(nominal_task, node, mean_iot_dist[node.node_id], mean_p_tx, False)['total_ms']
        )
        _nom_r_vals.append(base_reliability(node, nominal_task.cycles, nominal_task.size_kb, TR))

    prev_wl = 0.0                                     # cycles; empty queues
    prev_d  = float(np.mean(_nom_d_vals))             # ms; mean nominal delay across swarm
    prev_r  = float(np.mean(_nom_r_vals))             # dimensionless; mean nominal R0

    # ----------------------------------------------------------------
    # (C) PPO trajectory buffer — one slot per arriving task
    # ----------------------------------------------------------------
    buf_H:        List[torch.Tensor] = []   # per-node embeddings at decision time, detached
    buf_c:        List[torch.Tensor] = []   # global context vectors, detached
    buf_soc:      List[torch.Tensor] = []   # SOC vectors for mask re-application in PPO
    buf_cand_masks: List[torch.Tensor] = [] # physics top-K candidate masks (for PPO log_prob_of)
    buf_fp:       List[int]          = []   # primary node indices
    buf_fb:       List[int]          = []   # backup node indices (_NO_NODE if none)
    buf_logp_old: List[float]        = []   # log pi_old(a|s) at collection time (no grad)
    buf_rewards:  List[float]        = []   # realized scalar rewards
    buf_values:   List[float]        = []   # V_phi(c) estimates at collection time
    buf_dones:    List[bool]         = []   # True only for the last task of the episode

    # ----------------------------------------------------------------
    # (D) Episode accumulators
    # ----------------------------------------------------------------
    outcomes_all: List[Dict]  = []   # one dispatch outcome dict per task
    latencies_ms: List[float] = []   # per-task routing wall-clock time, ms
    wl_samples:   List[float] = []   # per-tick workload-imbalance WL (ReLIEF Eq. 17)

    # ----------------------------------------------------------------
    # (E) Main simulation clock
    # ----------------------------------------------------------------
    sim_time_s    = 0.0    # simulated elapsed time, seconds
    tick          = 0      # control-tick counter (1-based)
    next_task_idx = 0      # index of the next unconsumed task in `tasks`

    # ================================================================
    # MAIN LOOP — one iteration per 100 ms control tick
    # ================================================================
    while next_task_idx < TASKS_PER_EPISODE:

        tick       += 1
        sim_time_s += CONTROL_TICK_S
        tick_start_ms = (sim_time_s - CONTROL_TICK_S) * 1000.0   # ms — window start
        tick_end_ms   =  sim_time_s                  * 1000.0     # ms — window end

        # ---- (1) Mobility + propulsion energy drain for all 20 UAVs ----
        # Step 1 of the tick (UAV_Mobility_Design.pdf Fig. 2): drift hotspots, move
        # each UAV per MOVE_MODE, then charge Zeng19 P(V) * dt at its current speed
        # (static: V=0 => exactly the original P_HOVER * dt = 87.05 J per node).
        step_mobility(fog_nodes, iot_centres, CONTROL_TICK_S)
        for node in fog_nodes:
            drain_flight_energy(node, CONTROL_TICK_S)  # mutates node.E_res_j

        # ---- (1b) Recompute geometry after movement (PDF Fig. 2 step 2) ----
        # New positions => new slant ranges => new data rates for Stage 1/Stage 2.
        if MOVE_MODE != "static":
            mean_iot_dist, tr_repr = _refresh_geometry()

        # ---- (2) Stage 1: MCDM broker selection ----
        t_stage1 = time.perf_counter()

        update_memory_occupancy(fog_nodes)             # refresh MP_occ/MS_occ from live queues
        me_dict               = memory_efficiency(fog_nodes)
        wl_per_node, _        = system_workload_imbalance(fog_nodes)

        per_node_metrics: Dict[int, Dict] = {}
        for node in fog_nodes:
            dist = mean_iot_dist[node.node_id]
            TR   = tr_repr[node.node_id]
            per_node_metrics[node.node_id] = {
                'me':   me_dict[node.node_id],
                'r0':   base_reliability(node, nominal_task.cycles, nominal_task.size_kb, TR),
                'd_ms': total_delay_ms(nominal_task, node, dist, mean_p_tx, False)['total_ms'],
                'wl':   wl_per_node[node.node_id],
            }

        stage1 = broker.select_head(
            fog_nodes, per_node_metrics, episode_idx, tick)
        stage1_wall_ms = (time.perf_counter() - t_stage1) * 1000.0   # ms

        # ---- (3) Collect tasks whose Poisson arrival falls in this tick's window ----
        tick_tasks: List[Task] = []
        while (next_task_idx < TASKS_PER_EPISODE and
               tasks[next_task_idx].arrival_s * 1000.0 < tick_end_ms):
            tick_tasks.append(tasks[next_task_idx])
            next_task_idx += 1

        # ---- (4) Stage 2: route each arriving task ----
        tick_outcomes: List[Dict]   = []
        tick_reward_parts: Optional[Dict] = None   # last task's reward breakdown (for print)
        sample_task: Optional[Task] = None         # "interesting" task to highlight in print
        sample_fp = _NO_NODE
        sample_fb = _NO_NODE

        # Representative data rate per node for the encoder feature build.
        # Recomputed here so it reflects current node positions (static within episode).
        tr_for_encoder: Dict[int, float] = tr_repr   # reuse episode-level baseline

        for task_idx_in_tick, task in enumerate(tick_tasks):

            # Assign a random IoT source device: determines distance and p_tx.
            iot_dev = iot[random.randint(0, N_IOT - 1)]

            def _dist_to(node_list_idx: int) -> float:
                """3-D slant range from the assigned IoT device to fog_nodes[idx]. Min 1 m."""
                return slant_distance_m(fog_nodes[node_list_idx], iot_dev.x, iot_dev.y)

            # Pre-compute memory efficiency dict (used in candidate features + MCDM; already done
            # for Stage 1 above, but we need it per-task here too; reuse me_dict from Stage 1).
            # me_dict is computed in Stage 1 above; reuse it here.

            # Build task-aware, source-aware candidate features (shape N_FOG x 16).
            x_feat = build_candidate_features(fog_nodes, task, iot_dev, me_dict)   # (N_FOG, 16)

            # Physics-guided top-K candidate filtering.
            cand_mask, h_scores = filter_candidates(fog_nodes, task, iot_dev)

            t_stage2 = time.perf_counter()

            # Encoder: project features to contextual embeddings.
            # no_grad during rollout — gradients not needed until ppo_update.
            with torch.no_grad():
                H, c = encoder(x_feat, update_stats=train)   # H (N,128), c (128,)

            # Critic baseline estimate (scalar).
            with torch.no_grad():
                V_phi = float(critic(c).item())

            soc_vec = torch.tensor([n.soc for n in fog_nodes], dtype=torch.float32)  # (N,)

            # Pre-compute primary-candidate metrics for conditional backup decision.
            # Use the candidate with highest heuristic score as a proxy for what the
            # actor will likely choose (avoids a full forward pass before the action).
            # The actual task_r0 and slack_ratio are computed after primary is selected.
            # Initialise to trigger backup (conservative); will update after primary selection.
            _task_r0_proxy   = 1.0   # will be updated after primary is selected
            _slack_ratio_proxy = 1.0

            # Actor: select primary and backup.
            with torch.no_grad():
                f_p, f_b, logp_p, logp_b, ent_p, ent_b = actor.select_action(
                    H, c, soc_vec,
                    greedy=(not train),            # greedy = deterministic at eval time
                    candidate_mask=cand_mask,      # physics top-K mask
                    heuristic_scores=h_scores,     # for eval epsilon-greedy
                    task_r0_primary=_task_r0_proxy,
                    task_slack_ratio=_slack_ratio_proxy,
                )

            # If a valid primary was selected, recompute actual r0 and slack_ratio and
            # re-run select_action with the correct backup decision logic (training only).
            if f_p != _NO_NODE and train:
                _d_to_p  = _dist_to(f_p)
                _TR_p    = max(data_rate_bps(_d_to_p, iot_dev.p_tx_w), 1.0)
                _r0_p    = base_reliability(fog_nodes[f_p], task.cycles, task.size_kb, _TR_p)
                _qd_p    = queue_delay_ms(fog_nodes[f_p])   # outer-scope import via from physics import *
                _ul_p    = transmission_delay_ms(task.size_kb, _TR_p)
                _ex_p    = execution_delay_ms(task.cycles, fog_nodes[f_p].fr_avg_hz())
                _pred_p  = _ul_p + _qd_p + _ex_p
                _sr_p    = (task.deadline_ms - _pred_p) / max(task.deadline_ms, 1.0)
                # Re-run action selection with actual task metrics for correct backup decision.
                with torch.no_grad():
                    f_p, f_b, logp_p, logp_b, ent_p, ent_b = actor.select_action(
                        H, c, soc_vec,
                        greedy=False,
                        candidate_mask=cand_mask,
                        heuristic_scores=h_scores,
                        task_r0_primary=_r0_p,
                        task_slack_ratio=_sr_p,
                    )

            stage2_wall_ms = (time.perf_counter() - t_stage2) * 1000.0

            # Decision latency: MCDM cost amortized over tick's tasks + per-task Stage-2 cost.
            per_task_mcdm_ms = stage1_wall_ms / max(len(tick_tasks), 1)
            decision_lat_ms  = per_task_mcdm_ms + stage2_wall_ms
            latencies_ms.append(decision_lat_ms)

            # ---- Dispatch or drop ----
            if f_p == _NO_NODE:
                # All nodes below SOC_PRIMARY_MIN: task is undispatchable → drop.
                outcome = {
                    'success': 0, 'used_backup': False,
                    'D_primary_ms': 0.0, 'D_backup_ms': None,
                    'R_i': 0.0, 'primary_id': _NO_NODE, 'backup_id': _NO_NODE,
                    'dropped': True,
                }
                reward   = float(-CRITICAL_PENALTY)   # hard penalty for battery exhaustion
                r_parts  = {'dWL_contrib': 0.0, 'dD_contrib': 0.0,
                             'dR_contrib': 0.0, 'energy_contrib': 0.0,
                             'r_k': reward, 'rth_penalty': True}
                logp_old_val = 0.0    # dummy; ppo_update skips _NO_NODE steps

            else:
                d_iot_p = _dist_to(f_p)
                d_iot_b = _dist_to(f_b) if f_b != _NO_NODE else 0.0
                TQ_ms_pre = queue_delay_ms(fog_nodes[f_p])

                outcome = dispatch_task(
                    task       = task,
                    f_p        = f_p,
                    f_b        = f_b,
                    fog_nodes  = fog_nodes,
                    dist_p_m   = d_iot_p,
                    dist_b_m   = d_iot_b,
                    p_tx_w     = iot_dev.p_tx_w,
                    spotis_ran = stage1['run_spotis'],
                )

                # Recompute workload imbalance AFTER dispatch (queue grew).
                _, wl_curr = system_workload_imbalance(fog_nodes)

                # Energy consumption prediction for the reward energy term.
                TR_p  = data_rate_bps(d_iot_p, iot_dev.p_tx_w)
                TQ_ms_post = queue_delay_ms(fog_nodes[f_p])
                E_con = predicted_consume_energy_j(fog_nodes[f_p], task, TR_p, TQ_ms_pre)

                # Swarm queue depth statistics for hotspot penalty.
                all_depths = [
                    (n.cpu_state._engine.queue_depth(n.node_id)
                     if getattr(n, "cpu_state", None) is not None else
                     len(n.queue_primary) + len(n.queue_backup))
                    for n in fog_nodes]
                swarm_mean_depth   = float(np.mean(all_depths))
                swarm_max_depth    = float(max(all_depths))

                reward, r_parts = compute_reward(
                    task                  = task,
                    primary_node          = fog_nodes[f_p],
                    outcome               = outcome,
                    E_consume_j           = E_con,
                    queue_delay_primary_ms = TQ_ms_pre,
                    swarm_mean_queue_depth = swarm_mean_depth,
                    swarm_max_queue_depth  = swarm_max_depth,
                    next_queue_delay_primary_ms = TQ_ms_post,
                )

                # Advance previous-step metrics (retained for compatibility with print block).
                prev_wl = wl_curr
                prev_d  = outcome['D_primary_ms']
                prev_r  = outcome['R_i']

                # Stored log-prob: joint action (primary + backup).
                logp_p_f = logp_p.item() if logp_p is not None else 0.0
                logp_b_f = (logp_b.item()
                             if (logp_b is not None and f_b != _NO_NODE) else 0.0)
                logp_old_val = logp_p_f + logp_b_f

            # ---- Store trajectory step ----
            buf_H.append(H.detach())            # detached: gradients not needed until PPO update
            buf_c.append(c.detach())
            buf_soc.append(soc_vec)
            buf_cand_masks.append(cand_mask)    # store physics mask for PPO log_prob_of
            buf_fp.append(f_p)
            buf_fb.append(f_b if f_b is not None else _NO_NODE)
            buf_logp_old.append(logp_old_val)
            buf_rewards.append(float(reward))
            buf_values.append(V_phi)
            buf_dones.append(False)             # will fix the last entry after the loop

            tick_outcomes.append(outcome)
            outcomes_all.append(outcome)
            tick_reward_parts = r_parts         # keep the last task's breakdown for print

            # For the print block: prefer to display a dropped task (most informative).
            if sample_task is None or outcome['dropped']:
                sample_task = task
                sample_fp   = f_p
                sample_fb   = f_b if f_b is not None else _NO_NODE

        # Mark the very last task of the episode as terminal (for GAE bootstrap).
        if buf_dones and next_task_idx >= TASKS_PER_EPISODE:
            buf_dones[-1] = True

        # ---- (5) Queue cleanup ----
        # Remove tasks whose absolute deadline has already passed; they cannot succeed
        # and would only inflate queue-delay estimates for future tasks.
        for node in fog_nodes:
            cutoff_ms = sim_time_s * 1000.0
            node.queue_primary = [(t, d) for (t, d) in node.queue_primary if d >= cutoff_ms]
            node.queue_backup  = [(t, d) for (t, d) in node.queue_backup  if d >= cutoff_ms]

        # Sample the workload-imbalance metric WL (ReLIEF Eq. 17) once per tick.
        _, wl_tick = system_workload_imbalance(fog_nodes)
        wl_samples.append(wl_tick)

        # ---- (6) Print policy ----
        n_tick_drops   = sum(1 for o in tick_outcomes if o['dropped'])
        n_tick_success = sum(1 for o in tick_outcomes if o['success'])
        is_last_tick   = (next_task_idx >= TASKS_PER_EPISODE)
        force_print    = (tick == 1 or is_last_tick or stage1['head_changed'] or n_tick_drops > 0)
        should_print   = force_print or (tick % PRINT_EVERY == 0)

        if should_print:
            socs       = [n.soc for n in fog_nodes]
            min_soc_i  = int(np.argmin(socs))
            head       = stage1['head_node']
            z, w       = stage1['z'], stage1['w']

            print(f"\n[Ep {episode_idx} | Tick {tick} | t={sim_time_s*1000:.1f} ms]")
            print(f"  Energy  : meanSOC={100*float(np.mean(socs)):.1f}%  "
                  f"minSOC={100*socs[min_soc_i]:.2f}%  (UAV{min_soc_i} at {100*socs[min_soc_i]:.2f}%)")
            print(f"  MEREC   : z=[{', '.join(f'{v:.3f}' for v in z)}]")
            print(f"  Kalman  : w=[{', '.join(f'{v:.3f}' for v in w)}]")
            print(f"  KL-gate : KL={stage1['kl_value']:.4f} -> run_SPOTIS={stage1['run_spotis']}")

            if stage1['run_spotis']:
                fover = [n.node_id for n in stage1['ranking'][:8]]  # first 8 in failover order
                print(f"  SPOTIS  : head=UAV{head.node_id:02d} (tier{head.tier})  "
                      f"failover={fover}")
            else:
                print(f"  SPOTIS  : head unchanged  (UAV{head.node_id:02d})")

            if tick_tasks and sample_task is not None:
                fp_s = (f"UAV{sample_fp:02d} (SOC {fog_nodes[sample_fp].soc*100:.1f}%)"
                        if sample_fp != _NO_NODE else "NONE")
                fb_s = (f"UAV{sample_fb:02d} (SOC {fog_nodes[sample_fb].soc*100:.1f}%)"
                        if sample_fb != _NO_NODE else "none")
                print(f"  Routing : routed {len(tick_tasks)} tasks;  "
                      f"sample task#{sample_task.task_id} "
                      f"s={sample_task.size_kb:.1f} KB  "
                      f"-> primary {fp_s},  backup {fb_s}")
                if tick_reward_parts:
                    rp = tick_reward_parts
                    print(f"  Reward  : r={rp['r_k']:.3g}  "
                          f"[base={rp.get('base', 0):.3g}  slack={rp.get('slack_contrib', 0):.3g}  "
                          f"rel={rp.get('rel_contrib', 0):.3g}  E={rp.get('energy_contrib', 0):.3g}  "
                          f"q={rp.get('queue_contrib', 0):.3g}]")
            else:
                print(f"  Routing : no tasks this tick")

            print(f"  Exec    : successes={n_tick_success}/{len(tick_tasks)}  "
                  f"drops={n_tick_drops}")

            if VERBOSE:
                for t_v, o_v in zip(tick_tasks, tick_outcomes):
                    print(f"    [VERBOSE] task#{t_v.task_id}  "
                          f"fp={o_v['primary_id']}  fb={o_v['backup_id']}  "
                          f"success={o_v['success']}  D_p={o_v['D_primary_ms']:.1f} ms")

    # ================================================================
    # POST-EPISODE: GAE advantage computation + PPO update
    # ================================================================
    losses = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'total_loss': 0.0}

    if train and buf_rewards:
        T = len(buf_rewards)

        # Bootstrap next-state values: shift buffer by 1; terminal step gets 0.
        next_vals = buf_values[1:] + [0.0]   # V(s_{k+1}); V(s_T+1) = 0 (episode ends)

        advantages, vtargets = compute_gae(
            rewards     = buf_rewards,
            values      = buf_values,
            next_values = next_vals,
            dones       = buf_dones,
        )

        ppo_batch = {
            'H':          buf_H,
            'c':          buf_c,
            'soc':        buf_soc,
            'cand_masks': buf_cand_masks,
            'f_p':        buf_fp,
            'f_b':        buf_fb,
            'logp_old':   buf_logp_old,
            'advantages': advantages,
            'vtargets':   vtargets,
        }

        actor.train(); critic.train()    # ensure train mode before backward passes
        losses = ppo_update(actor, critic, optimizer, ppo_batch)

    # ================================================================
    # EPISODE STATISTICS
    # ================================================================
    drop_rate     = episode_drop_rate(outcomes_all)
    r_values      = [o['R_i'] for o in outcomes_all]
    mean_rel      = mean_system_reliability(r_values)          # secondary: avg per-task R_i
    log10_R       = log10_system_reliability(r_values)         # ReLIEF Eq.(12), log domain (exact)
    geo_rel       = geometric_mean_reliability(r_values)       # Eq.(12) normalized per task
    rel_nines     = system_reliability_nines(log10_R)          # Fig. 6-comparable display form
    valid_delays  = [o['D_primary_ms'] for o in outcomes_all
                     if not o['dropped'] and o['D_primary_ms'] > 0.0]
    mean_delay_ms = float(np.mean(valid_delays)) if valid_delays else 0.0

    socs_final   = [n.soc for n in fog_nodes]
    soc_spread   = float(max(socs_final) - min(socs_final))   # range of SOC at episode end
    mean_soc     = float(np.mean(socs_final))

    lat_arr      = np.array(latencies_ms) if latencies_ms else np.array([0.0])
    mean_lat_ms  = float(lat_arr.mean())
    p95_lat_ms   = float(np.percentile(lat_arr, 95))
    budget_ms    = 1000.0 / TASK_ARRIVAL_RATE   # 20 ms inter-arrival budget

    mean_wl      = float(np.mean(wl_samples)) if wl_samples else 0.0   # ReLIEF Eq. 17

    return {
        'episode':          episode_idx,
        'drop_rate':        round(drop_rate, 4),
        'mean_reliability': round(mean_rel, 6),
        'log10_system_reliability': round(log10_R, 4),   # ReLIEF Eq.(12): sum_i log10 R_i
        'geo_mean_reliability':     round(geo_rel, 6),   # (prod_i R_i)^(1/N)
        'reliability_nines':        round(rel_nines, 3), # -log10(1 - prod_i R_i)
        'mean_delay_ms':    round(mean_delay_ms, 3),
        'workload_imbalance': round(mean_wl, 1),
        'soc_spread':       round(soc_spread, 4),
        'mean_soc':         round(mean_soc, 4),
        'mean_lat_ms':      round(mean_lat_ms, 4),
        'p95_lat_ms':       round(p95_lat_ms, 4),
        'budget_ms':        budget_ms,
        'n_tasks':          len(outcomes_all),
        **losses,
    }


# ==============================================================================


def run_episode(
    episode_idx: int,
    actor: Actor,
    critic: Critic,
    broker: BrokerSelector,
    encoder: AttentionEncoder,
    optimizer: torch.optim.Optimizer,
    train: bool = True,
    controller_profile: Optional[Dict] = None,
) -> Dict:
    """Compatibility wrapper around :func:`scenario.run_scenario`.

    ``broker`` remains in the signature for existing callers; the shared
    Attention+PPO adapter owns the per-scenario persistent broker state.
    """
    from controller_profile import require_profile
    from execution_engine import EXECUTION_MODEL_VERSION
    from scenario import ScenarioConfig, make_policy, run_scenario

    profile = controller_profile or require_profile()
    seed = SEED + 1000 + max(episode_idx - 1, 0)
    duration = TASKS_PER_EPISODE / TASK_ARRIVAL_RATE
    if train:
        encoder.train(); actor.train(); critic.train()
    else:
        encoder.eval(); actor.eval(); critic.eval()
    adapter = make_policy(
        "Attention+PPO", seed, (encoder, actor),
        critic=critic if train else None)
    result = run_scenario(
        ScenarioConfig(
            seed=seed, target_load=0.96, policy="Attention+PPO",
            controller_placement="equal_controller",
            warmup_s=0.0, measurement_s=duration, training=train),
        adapter, profile)

    losses = {
        "policy_loss": 0.0, "value_loss": 0.0,
        "entropy": 0.0, "total_loss": 0.0}
    if train and result.rollout:
        values = [row["value"] for row in result.rollout]
        rewards = [row["reward"] for row in result.rollout]
        dones = [False] * len(values)
        dones[-1] = True
        advantages, targets = compute_gae(
            rewards, values, values[1:] + [0.0], dones)
        losses = ppo_update(actor, critic, optimizer, {
            "H": [row["H"] for row in result.rollout],
            "c": [row["c"] for row in result.rollout],
            "soc": [row["soc"] for row in result.rollout],
            "cand_masks": [row["cand_mask"] for row in result.rollout],
            "f_p": [row["f_p"] for row in result.rollout],
            "f_b": [row["f_b"] for row in result.rollout],
            "logp_old": [row["logp_old"] for row in result.rollout],
            "advantages": advantages,
            "vtargets": targets,
        })
    latencies = [
        float(row["latency_ms"]) for row in result.task_records
        if math.isfinite(float(row.get("latency_ms", math.nan)))]
    success = max(min(result.success_rate, 1.0), 0.0)
    log10_reliability = (
        len(result.task_records) * math.log10(max(success, 1e-12))
        if result.task_records else 0.0)
    return {
        "episode": episode_idx,
        "drop_rate": result.drop_rate,
        "mean_reliability": success,
        "log10_system_reliability": log10_reliability,
        "geo_mean_reliability": success,
        "reliability_nines": -math.log10(max(1.0 - success, 1e-12)),
        "mean_delay_ms": float(np.mean(latencies)) if latencies else 0.0,
        "workload_imbalance": 0.0,
        "soc_spread": 0.0,
        "mean_soc": 0.0,
        "mean_lat_ms": 0.0,
        "p95_lat_ms": 0.0,
        "budget_ms": 1000.0 / TASK_ARRIVAL_RATE,
        "n_tasks": len(result.task_records),
        "execution_model_version": EXECUTION_MODEL_VERSION,
        **losses,
    }
