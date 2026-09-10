"""
neural.py
=========
Stage 2 of the control loop: the learned routing policy.

  * State encoder   (SECTION 12) — RunningNorm + self-attention AttentionEncoder
                                   and build_candidate_features.
  * Actor & Critic  (SECTION 13) — primary/backup action heads and the value head.
  * Learning        (SECTION 14) — reward shaping, GAE advantage, and the PPO update.

The trailing _demo_* helpers are illustrative, never-called examples.
"""

import time
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from models import Task, FogNode, IoTDevice
from world import FOG_SWARM, IOT_DEVICES, TASK_STREAM
from physics import *

# SECTION 12 — STAGE 2: STATE ENCODER  (§4.1)
#
# The broker feeds each arriving task to an attention-based encoder that reads
# the current state of all N_FOG=20 nodes and produces:
#   H   — per-node embeddings  shape (N_FOG, D_MODEL=128)
#   c   — global context vector shape (D_MODEL,) = mean over node embeddings
#
# The actor and critic (Section 13) then use H and c to select PRIMARY + BACKUP.
#
# Per-candidate raw feature vector x_j (18 scalars) — task-aware, source-aware:
#   [0]  soc_j                  — state of charge (0..1)
#   [1]  queue_cycles_norm      — pending cycles, normalised by WL_MAX_CYCLES
#   [2]  queue_delay_ms_norm    — queue wait, normalised by DEADLINE_LARGE_MS
#   [3]  queue_depth_norm       — queue depth, normalised by MAX_QUEUE_DEPTH
#   [4]  reliability_j          — task-specific R0 for actual task + IoT source
#   [5]  dist_norm              — IoT-to-UAV distance, normalised by GRID_M
#   [6]  uplink_delay_norm      — T_u for this task, normalised by DEADLINE_LARGE_MS
#   [7]  exec_delay_norm        — T_e for this task, normalised by DEADLINE_LARGE_MS
#   [8]  pred_total_delay_norm  — predicted total delay, normalised by DEADLINE_LARGE_MS
#   [9]  deadline_slack_norm    — (deadline - pred_delay) / deadline  in [-inf, 1]
#   [10] norm_slack_clipped     — clipped deadline_slack_norm into [-1, 1]
#   [11] pred_energy_norm       — predicted energy cost, normalised by HW_E_STRONG_J
#   [12] marginal_queue_norm    — extra queue damage after assigning (new_q - cur_q) / WL_MAX_CYCLES
#   [13] projected_util         — projected workload fraction of WL_MAX_CYCLES after assign
#   [14] tier_norm              — tier / 2.0  (so tier 1 -> 0.5, tier 2 -> 1.0)
#   [15] mem_efficiency_j       — memory efficiency ME in [0, 1]
#   [16] fodas_feasible         — FODAS 10 ms-margin deadline-feasibility flag
#   [17] fodas_objective_norm   — FODAS 0.7*time + 0.3*energy objective, min-max normalized
#
# This replaces the former 5-feature "representative average" vector with a
# per-task, per-IoT-source feature that lets the actor choose the best executor
# for the ACTUAL task and ACTUAL IoT source rather than a generic "good UAV".
# ==============================================================================

_N_NODE_FEATURES: int = 18   # length of per-candidate feature vector x_j


class RunningNorm:
    """
    Online EMA (Exponential Moving Average) feature normalizer.

    Maintains per-feature mean mu and std sigma, updated each training step.
    Inference (eval) uses the frozen statistics from training.

    Normalization: x_tilde = (x - mu) / (sigma + 1e-5)

    The 1e-5 floor prevents division by zero when a feature has near-zero variance
    (e.g., all nodes at full charge early in the episode).

    Why EMA rather than batch statistics?
      - The feature distribution shifts as the episode progresses (SOC drains,
        queues grow) — a fixed mean/std from episode start would go stale.
      - EMA with momentum=0.99 gives a ~100-step memory, tracking slow drift
        without amplifying noise from individual outlier ticks.

    Usage:
      norm.update(x_batch)  — call during training steps to update statistics
      norm.normalize(x_batch) — call at inference and training to normalize
    """

    def __init__(self, n_features: int, momentum: float = 0.99) -> None:
        self.momentum  = momentum                              # EMA decay; 0.99 => ~100-step memory
        self.mu        = torch.zeros(n_features)               # running mean, per feature
        self.sigma     = torch.ones(n_features)                # running std,  per feature
        self.training  = True                                  # False = freeze stats at inference
        self._first    = True                                  # flag: initialize from first batch

    def update(self, x: torch.Tensor) -> None:
        """Update EMA statistics from a batch of feature rows, shape (N, n_features)."""
        if not self.training:
            return                                             # stats frozen during inference

        batch_mean = x.mean(dim=0).detach()                   # per-feature mean over N nodes
        batch_std  = x.std(dim=0).detach().clamp(min=1e-5)    # per-feature std; clamp avoids zero

        if self._first:
            # Warm-start: use batch statistics directly on first call.
            self.mu    = batch_mean
            self.sigma = batch_std
            self._first = False
        else:
            # EMA update: mu <- momentum*mu + (1-momentum)*batch_mean
            self.mu    = self.momentum * self.mu    + (1.0 - self.momentum) * batch_mean
            self.sigma = self.momentum * self.sigma + (1.0 - self.momentum) * batch_std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Return (x - mu) / (sigma + 1e-5), shape unchanged."""
        return (x - self.mu) / (self.sigma + 1e-5)            # 1e-5 floor: avoids /0


def expected_wait_ms(node: FogNode, task: Task) -> float:
    """Exact EDF insertion wait when the shared engine is present."""
    state = getattr(node, "cpu_state", None)
    engine = getattr(state, "_engine", None) if state is not None else None
    if engine is None:
        return queue_delay_ms(node)
    deadline_s = task.arrival_s + task.deadline_ms / 1000.0
    return 1000.0 * engine.projected_edf_delay_s(
        node.node_id, deadline_s=deadline_s)


class AttentionEncoder(nn.Module):
    """
    Transformer encoder that maps a set of N fog-node feature vectors to embeddings.

    Architecture [Vas17]:
      1. Linear embedding: x_j (5-dim) -> h_j (D_MODEL=128-dim)
      2. N_ENC_LAYERS=3 Transformer encoder layers, each implementing:
           h = LayerNorm( h + MultiHeadAttention(h, h, h) )   (self-attention + residual)
           h = LayerNorm( h + FFN(h) )                        (feed-forward + residual)
         with N_HEADS=8 attention heads (d_k = D_MODEL/N_HEADS = 128/8 = 16 per head)
         and FFN hidden width D_FF=256 with ReLU activation.

    Implementation: uses nn.TransformerEncoderLayer(batch_first=True, norm_first=False)
    which applies LayerNorm AFTER the residual (post-norm, as in the original [Vas17]).
    batch_first=True means input shape is (batch, seq, features) = (1, N, D_MODEL).

    NO positional encoding: UAV nodes form an unordered set; position in the list is
    arbitrary and must not bias the attention scores.

    forward input:  x_batch  shape (N, 5)   — raw or pre-normalized feature matrix
    forward output: H        shape (N, 128)  — per-node contextual embeddings
                    c        shape (128,)    — global context = mean over node embeddings
    """

    def __init__(self) -> None:
        super().__init__()

        # Linear projection: maps raw 18-dim candidate features into the D_MODEL embedding space.
        self.embed = nn.Linear(_N_NODE_FEATURES, D_MODEL)     # 18 -> 128

        # Stack of N_ENC_LAYERS=3 Transformer encoder layers.
        # Each layer: MultiHeadSelfAttention + LayerNorm + FFN(ReLU) + LayerNorm [Vas17].
        # batch_first=True: input/output tensors have shape (batch, seq_len, d_model).
        # dim_feedforward=D_FF=256: hidden width of the 2-layer FFN inside each layer [Vas17].
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = D_MODEL,     # 128 — embedding/attention dimension [Vas17]
            nhead           = N_HEADS,     # 8   — attention heads; d_k = 128/8 = 16 [Vas17]
            dim_feedforward = D_FF,        # 256 — FFN hidden width (~2x D_MODEL) [Vas17]
            dropout         = 0.0,         # no dropout: small swarm (N=20), sim environment
            activation      = 'relu',      # ReLU as in [Vas17] (GELU also valid but unnecessary)
            batch_first     = True,        # (batch, seq, d_model) layout; cleaner for our use
            norm_first      = False,       # post-norm: LayerNorm after residual, matching [Vas17]
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer  = encoder_layer,
            num_layers     = N_ENC_LAYERS,   # 3 stacked layers [Vas17]; sufficient for N=20 nodes
            enable_nested_tensor = False,    # disable nested-tensor optimization (not needed for N=20)
        )

        # EMA feature normalizer — keeps all 5 feature scales comparable before embedding.
        self.running_norm = RunningNorm(n_features=_N_NODE_FEATURES, momentum=0.99)

    def forward(
        self,
        x_batch: torch.Tensor,   # shape (N, 5); N = number of fog nodes
        update_stats: bool = False,   # True during training steps only
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode fog-node features into contextual embeddings.

        Args:
            x_batch     : raw per-node feature matrix, shape (N, 5)
            update_stats: if True, update EMA normalizer statistics (training only)

        Returns:
            H : per-node embeddings, shape (N, D_MODEL=128)
            c : global context vector, shape (D_MODEL,) = mean over rows of H
        """
        # Update running statistics during training (noop during inference).
        if update_stats:
            self.running_norm.update(x_batch)

        x_norm = self.running_norm.normalize(x_batch)          # (N, 5); zero-mean, unit-std per feature

        h = self.embed(x_norm)                                  # (N, 128); linear projection

        # TransformerEncoder expects (batch, seq, d_model); add batch dim of 1.
        h = h.unsqueeze(0)                                      # (1, N, 128)
        h = self.encoder(h)                                     # (1, N, 128); attention applied
        h = h.squeeze(0)                                        # (N, 128); remove batch dim

        c = h.mean(dim=0)                                       # (128,); global context = mean pooling

        return h, c   # H: per-node embeddings; c: global context


def build_candidate_features(
    fog_nodes:  List[FogNode],
    task:       "Task",
    iot_dev:    "IoTDevice",
    me_dict:    Optional[Dict[int, float]] = None,
) -> torch.Tensor:
    """
    Build the task-aware, source-aware per-candidate feature matrix.

    Shape: (N_FOG, 18) — one row per fog node; 18 features per candidate.

    Feature indices match _N_NODE_FEATURES layout (see header).

    Args:
        fog_nodes: current fog swarm
        task:      the task to be dispatched
        iot_dev:   the IoT device that generated the task (determines distance & p_tx)
        me_dict:   optional pre-computed memory-efficiency dict; computed internally if None

    Unlike the former build_node_features (which used a nominal representative task),
    this builder uses the actual task parameters and actual IoT-to-UAV geometry,
    so the actor learns to prefer the best executor for THIS task and THIS source.
    """
    from physics import (
        slant_distance_m, data_rate_bps, transmission_delay_ms,
        execution_delay_ms, queue_delay_ms, base_reliability,
        memory_efficiency, node_workload_cycles, task_memory_footprint_gb,
    )
    import math

    if me_dict is None:
        me_dict = memory_efficiency(fog_nodes)

    # Normalisation denominators — constants from config.
    _WL_MAX   = max(WL_MAX_CYCLES, 1.0)
    _D_MAX    = max(DEADLINE_LARGE_MS, 1.0)
    _E_MAX    = max(HW_E_STRONG_J, 1.0)
    _Q_MAX    = max(MAX_QUEUE_DEPTH, 1)

    raw_rows = []
    fodas_times = []
    fodas_energies = []
    fodas_feasible = []

    for node in fog_nodes:
        # ---- distance & link ----
        dist_m = slant_distance_m(node, iot_dev.x, iot_dev.y)
        TR_bps = max(data_rate_bps(dist_m, iot_dev.p_tx_w), 1.0)

        # ---- queue state ----
        queue_cycles   = node_workload_cycles(node)
        state = getattr(node, "cpu_state", None)
        engine = getattr(state, "_engine", None) if state is not None else None
        queue_depth = (engine.queue_depth(node.node_id) if engine is not None
                       else len(node.queue_primary) + len(node.queue_backup))
        pessimistic_queue_delay_ms = queue_delay_ms(node)
        queue_delay_ms_val = expected_wait_ms(node, task)

        # ---- task-specific delays ----
        uplink_ms = transmission_delay_ms(task.size_kb, TR_bps)
        exec_ms   = execution_delay_ms(task.cycles, node.fr_avg_hz())
        # Predicted total includes uplink + queue + exec (no broker overhead for comparison purposes)
        pred_total_ms = uplink_ms + queue_delay_ms_val + exec_ms

        # ---- deadline slack ----
        deadline_slack_ms = task.deadline_ms - pred_total_ms
        norm_slack        = deadline_slack_ms / max(task.deadline_ms, 1.0)  # in (-inf, 1]
        norm_slack_clipped = float(np.clip(norm_slack, -1.0, 1.0))

        # ---- task-specific reliability ----
        r0 = base_reliability(node, task.cycles, task.size_kb, TR_bps)

        # ---- predicted energy cost ----
        from physics import comm_energy_j
        Te_s       = task.cycles / node.fr_avg_hz()
        TQ_s       = queue_delay_ms_val / 1000.0
        E_queue    = (P_HOVER + P_IDLE) * TQ_s
        E_exec     = (P_HOVER + P_PROC) * Te_s
        E_comm     = comm_energy_j(node, task.size_kb, TR_bps)
        pred_energy = E_queue + E_exec + E_comm

        # ---- marginal queue externality ----
        # Approximate extra queue damage of adding this task on top of current load.
        new_cycles = queue_cycles + task.cycles
        marginal_queue_norm = float(np.clip(
            (new_cycles - queue_cycles) / _WL_MAX, 0.0, 1.0
        ))

        # ---- projected utilization ----
        projected_util = float(np.clip(new_cycles / _WL_MAX, 0.0, 1.0))

        # ---- memory efficiency ----
        me_j = me_dict.get(node.node_id, 0.5)

        # FODAS anchor signals: same 10 ms feasibility margin and 0.7/0.3
        # completion-time/energy trade-off used by baselines.fodas_baseline.
        remaining_ms = task.arrival_s * 1000.0 + task.deadline_ms - task.arrival_s * 1000.0
        fodas_total_ms = uplink_ms + pessimistic_queue_delay_ms + exec_ms
        fodas_energy = ((P_HOVER + P_IDLE)
                        * pessimistic_queue_delay_ms / 1000.0
                        + E_exec + E_comm)
        meets_fodas_margin = fodas_total_ms <= max(0.0, remaining_ms - 10.0)
        fodas_times.append(fodas_total_ms)
        fodas_energies.append(fodas_energy)
        fodas_feasible.append(1.0 if meets_fodas_margin else 0.0)

        raw_rows.append([
            node.soc,                                                  # [0]
            float(np.clip(queue_cycles / _WL_MAX, 0.0, 1.0)),         # [1] queue_cycles_norm
            float(np.clip(queue_delay_ms_val / _D_MAX, 0.0, 1.0)),    # [2] queue_delay_norm
            float(np.clip(queue_depth / _Q_MAX, 0.0, 1.0)),           # [3] queue_depth_norm
            r0,                                                         # [4] reliability
            float(np.clip(dist_m / GRID_M, 0.0, 2.0)),                # [5] dist_norm
            float(np.clip(uplink_ms / _D_MAX, 0.0, 1.0)),             # [6] uplink_delay_norm
            float(np.clip(exec_ms / _D_MAX, 0.0, 1.0)),               # [7] exec_delay_norm
            float(np.clip(pred_total_ms / _D_MAX, 0.0, 2.0)),         # [8] pred_total_delay_norm
            norm_slack,                                                 # [9] deadline_slack_norm
            norm_slack_clipped,                                        # [10] norm_slack_clipped
            float(np.clip(pred_energy / _E_MAX, 0.0, 1.0)),           # [11] pred_energy_norm
            marginal_queue_norm,                                       # [12] marginal_queue_norm
            projected_util,                                            # [13] projected_util
            float(node.tier) / 2.0,                                   # [14] tier_norm (1->0.5, 2->1.0)
            me_j,                                                      # [15] mem_efficiency
        ])

    t_min, t_max = min(fodas_times), max(fodas_times)
    e_min, e_max = min(fodas_energies), max(fodas_energies)
    t_range = t_max - t_min if t_max > t_min else 1.0
    e_range = e_max - e_min if e_max > e_min else 1.0

    rows = []
    for row, total_ms, energy_j, feasible in zip(raw_rows, fodas_times, fodas_energies, fodas_feasible):
        norm_t = (total_ms - t_min) / t_range
        norm_e = (energy_j - e_min) / e_range
        fodas_obj = float(np.clip(0.7 * norm_t + 0.3 * norm_e, 0.0, 1.0))
        rows.append(row + [feasible, fodas_obj])

    return torch.tensor(rows, dtype=torch.float32)   # shape (N_FOG, 18)


def filter_candidates(
    fog_nodes: List[FogNode],
    task:      "Task",
    iot_dev:   "IoTDevice",
    feat:      Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Physics-guided candidate filtering for Stage 2 dispatch.

    Returns:
        candidate_mask   : BoolTensor (N,) — True = candidate is admissible for primary selection
        heuristic_scores : FloatTensor (N,) — higher = better (used for epsilon-greedy sampling)

    Hard rejection rules (candidate excluded from primary selection):
        - SOC < SOC_PRIMARY_MIN
        - queue depth >= MAX_QUEUE_DEPTH

    Admission rule:
        - predicted_total_delay <= task.deadline_ms - ADMISSION_MARGIN_MS

    Top-k selection: only the top-k scoring candidates by heuristic
    score remain admissible.  This keeps the actor's effective action space small without
    removing the backup candidates.

    Heuristic score per candidate:
        score = 2.0*slack_ratio + 1.0*reliability - 0.75*utilization
                - 0.50*queue_norm - 0.50*energy_norm
                - W_LINK_ENERGY*link_energy_norm
    """
    from physics import (
        slant_distance_m, data_rate_bps, transmission_delay_ms,
        execution_delay_ms, queue_delay_ms, base_reliability,
        node_workload_cycles,
    )

    N = len(fog_nodes)
    scores = np.full(N, -1e9, dtype=float)   # heuristic score; low default = excluded
    link_j = np.zeros(N, dtype=float)
    hard_ok = np.zeros(N, dtype=bool)        # True = passes hard constraints
    deadline_ok = np.zeros(N, dtype=bool)    # True = predicted delay <= deadline

    _WL_MAX = max(WL_MAX_CYCLES, 1.0)
    _E_MAX  = max(HW_E_STRONG_J, 1.0)

    for j, node in enumerate(fog_nodes):
        soc         = node.soc
        state = getattr(node, "cpu_state", None)
        engine = getattr(state, "_engine", None) if state is not None else None
        queue_depth = (engine.queue_depth(node.node_id) if engine is not None
                       else len(node.queue_primary) + len(node.queue_backup))

        # Hard constraints
        if soc < SOC_PRIMARY_MIN:
            continue
        if queue_depth >= MAX_QUEUE_DEPTH:
            continue
        hard_ok[j] = True

        dist_m = slant_distance_m(node, iot_dev.x, iot_dev.y)
        TR_bps = max(data_rate_bps(dist_m, iot_dev.p_tx_w), 1.0)

        q_delay_ms  = expected_wait_ms(node, task)
        uplink_ms   = transmission_delay_ms(task.size_kb, TR_bps)
        link_j[j]   = iot_dev.p_tx_w * uplink_ms / 1000.0
        exec_ms     = execution_delay_ms(task.cycles, node.fr_avg_hz())
        pred_total  = uplink_ms + q_delay_ms + exec_ms

        slack_ms    = task.deadline_ms - pred_total
        slack_ratio = float(np.clip(slack_ms / max(task.deadline_ms, 1.0), -1.0, 1.0))
        deadline_ok[j] = slack_ms >= ADMISSION_MARGIN_MS

        reliability = base_reliability(node, task.cycles, task.size_kb, TR_bps)

        q_cycles    = node_workload_cycles(node)
        utilization = float(np.clip(q_cycles / _WL_MAX, 0.0, 1.0))
        queue_norm  = float(np.clip(q_delay_ms / max(DEADLINE_LARGE_MS, 1.0), 0.0, 1.0))

        Te_s        = task.cycles / node.fr_avg_hz()
        TQ_s        = q_delay_ms / 1000.0
        pred_energy = (P_HOVER + P_IDLE) * TQ_s + (P_HOVER + P_PROC) * Te_s
        energy_norm = float(np.clip(pred_energy / _E_MAX, 0.0, 1.0))

        scores[j] = (2.0 * slack_ratio
                     + 1.0 * reliability
                     - 0.75 * utilization
                     - 0.50 * queue_norm
                     - 0.50 * energy_norm)

    if hard_ok.any():
        valid_link_j = link_j[hard_ok]
        link_norm = ((valid_link_j - valid_link_j.min())
                     / max(float(np.ptp(valid_link_j)), 1e-9))
        scores[hard_ok] -= W_LINK_ENERGY * link_norm

    admissible = hard_ok & deadline_ok

    # Top-k selection among admissible candidates.
    if admissible.any():
        admissible_scores = np.where(admissible, scores, -1e10)
        topk = min(DISPATCH_TOP_K, int(admissible.sum()))
        topk_indices = np.argpartition(admissible_scores, -topk)[-topk:]
        topk_mask = np.zeros(N, dtype=bool)
        topk_mask[topk_indices] = True
        candidate_mask = topk_mask & admissible
    else:
        # No valid candidate at all — let the actor fall through to _NO_NODE.
        candidate_mask = np.zeros(N, dtype=bool)

    candidate_mask_t  = torch.tensor(candidate_mask, dtype=torch.bool)
    heuristic_scores_t = torch.tensor(scores, dtype=torch.float32)
    return candidate_mask_t, heuristic_scores_t


# ------------------------------------------------------------------------------
# §12 — DEBUG DEMO
# ------------------------------------------------------------------------------
def _demo_encoder() -> None:
    """
    Feed 20 candidate feature rows through AttentionEncoder; verify output shapes.
    Not called in normal execution; toggle the `if False` guard to run.
    """
    from world import TASK_STREAM
    from physics import memory_efficiency

    # Use first task in the stream and first IoT device as representative.
    demo_task = TASK_STREAM[0]
    demo_iot  = IOT_DEVICES[0]

    # Build real feature matrix from live swarm state.
    me_dict = memory_efficiency(FOG_SWARM)
    x = build_candidate_features(FOG_SWARM, demo_task, demo_iot, me_dict)  # shape (20, 16)

    encoder = AttentionEncoder()
    encoder.eval()                                         # freeze dropout / BN (none here, but good habit)

    with torch.no_grad():
        H, c = encoder(x, update_stats=False)

    print("\n[DEMO §12] AttentionEncoder output shapes")
    print(f"  Input  x      : {tuple(x.shape)}   (N_FOG={N_FOG}, features={_N_NODE_FEATURES})")
    print(f"  Output H      : {tuple(H.shape)}   (N_FOG={N_FOG}, D_MODEL={D_MODEL}) ✓" if H.shape == (N_FOG, D_MODEL) else f"  Output H WRONG: {tuple(H.shape)}")
    print(f"  Output c      : {tuple(c.shape)}   (D_MODEL={D_MODEL},) ✓" if c.shape == (D_MODEL,) else f"  Output c WRONG: {tuple(c.shape)}")
    print(f"  H mean        : {H.mean().item():.4f}  H std: {H.std().item():.4f}")
    print(f"  c mean        : {c.mean().item():.4f}  c std: {c.std().item():.4f}")
    print(f"  c == H.mean(0): {torch.allclose(c, H.mean(dim=0))}")




# ==============================================================================
# SECTION 13 — STAGE 2: ACTOR AND CRITIC  (§4.2 and §4.4)
# ==============================================================================

# Sentinel used when no feasible node exists under a mask (SOC too low / already primary).
# The executor interprets (f_p=None or f_b=None) as an undispatchable task and drops it,
# incrementing the episode drop counter.  This is the ONLY mechanism for task dropping —
# we never silently route to a constraint-violating node.
_NO_NODE: int = -1   # integer sentinel; stored in PPO buffer as -1 to flag drop

_CAND_FEATURE_EPS: float = 1e-8  # guard for feature normalisers


def _build_scoring_mlp(in_dim: int) -> nn.Sequential:
    """
    Build a 2-hidden-layer MLP that maps a feature vector to a single scalar score.

    Architecture: Linear(in_dim, D_MODEL) -> ReLU -> Linear(D_MODEL, D_MODEL) -> ReLU
                  -> Linear(D_MODEL, 1)
    Output is a raw logit (unbounded); callers apply softmax + masking.
    """
    return nn.Sequential(
        nn.Linear(in_dim, D_MODEL),    # first hidden layer; width D_MODEL=128
        nn.ReLU(),
        nn.Linear(D_MODEL, D_MODEL),   # second hidden layer; same width
        nn.ReLU(),
        nn.Linear(D_MODEL, 1),         # output: single logit score
    )


class Actor(nn.Module):
    """
    Two-head actor for hierarchical PRIMARY + BACKUP node selection.

    Primary head MLP_p:
      Input:  [h_j || c]          shape (2*D_MODEL = 256)
      Output: scalar score z_p^j  for each node j
      Selects f_p from softmax distribution over N nodes (SOC >= 0.30 required).

    Backup head MLP_b:
      Input:  [h_j || c || h_{f_p}]  shape (3*D_MODEL = 384)
      Output: scalar score z_b^j
      Selects f_b from softmax distribution over N nodes, given the chosen primary.
      Constraints: j != f_p  AND  SOC_j >= SOC_BACKUP_MIN (0.15).

    IMPORTANT — masking policy:
      We mask only on SOC floors and the identity constraint (backup != primary).
      We do NOT mask high-tier nodes as backup candidates.
      "Prefer lower-tier backups" is a LEARNED preference encoded via the tier
      FEATURE (index 4 of x_j) — the agent discovers this from reward signals.
      Hard-coding it as a mask would remove the agent's ability to adapt when
      a lower-tier node is unavailable or overloaded.

    Edge cases:
      - All nodes violate primary SOC mask: f_p = _NO_NODE (task is dropped by executor).
      - Valid primary found but all other nodes violate backup mask: f_b = _NO_NODE
        (single-node dispatch — executor may still attempt execution but records no redundancy).
    """

    def __init__(self) -> None:
        super().__init__()
        # Primary scoring head: input = [h_j (D_MODEL) || c (D_MODEL)] = 2*D_MODEL
        self.mlp_p = _build_scoring_mlp(in_dim=2 * D_MODEL)
        # Backup scoring head: input = [h_j (D_MODEL) || c (D_MODEL) || h_fp (D_MODEL)] = 3*D_MODEL
        self.mlp_b = _build_scoring_mlp(in_dim=3 * D_MODEL)

    # ------------------------------------------------------------------
    # Internal: compute logits for all N nodes given the input vectors
    # ------------------------------------------------------------------
    def _primary_logits(self, H: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """z_p^j = MLP_p([h_j || c]) for each j.  Returns shape (N,)."""
        N = H.shape[0]
        c_exp = c.unsqueeze(0).expand(N, -1)       # (N, D_MODEL): broadcast context to all rows
        inp   = torch.cat([H, c_exp], dim=-1)       # (N, 2*D_MODEL)
        return self.mlp_p(inp).squeeze(-1)          # (N,)

    def _backup_logits(
        self, H: torch.Tensor, c: torch.Tensor, h_fp: torch.Tensor
    ) -> torch.Tensor:
        """z_b^j = MLP_b([h_j || c || h_fp]) for each j.  Returns shape (N,)."""
        N = H.shape[0]
        c_exp   = c.unsqueeze(0).expand(N, -1)      # (N, D_MODEL)
        hfp_exp = h_fp.unsqueeze(0).expand(N, -1)   # (N, D_MODEL): primary embedding broadcast
        inp     = torch.cat([H, c_exp, hfp_exp], dim=-1)  # (N, 3*D_MODEL)
        return self.mlp_b(inp).squeeze(-1)           # (N,)

    # ------------------------------------------------------------------
    # Internal: build boolean feasibility masks as additive -inf tensors
    # ------------------------------------------------------------------
    @staticmethod
    def _primary_mask(soc_vector: torch.Tensor) -> torch.Tensor:
        """
        M_p[j] = -inf  if SOC_j < SOC_PRIMARY_MIN (0.30),  else 0.0.
        Adding to logits suppresses infeasible nodes before softmax.
        """
        mask = torch.zeros_like(soc_vector)
        mask[soc_vector < SOC_PRIMARY_MIN] = float('-inf')
        return mask   # shape (N,)

    @staticmethod
    def _backup_mask(soc_vector: torch.Tensor, f_p: int) -> torch.Tensor:
        """
        M_b[j] = -inf  if j == f_p  OR  SOC_j < SOC_BACKUP_MIN (0.15),  else 0.0.
        The identity constraint (j != f_p) ensures primary and backup are on distinct hardware.
        The 0.15 floor is the Return-To-Home threshold: below this a UAV must divert to base
        and cannot reliably execute or retransmit [spec §4].
        """
        mask = torch.zeros_like(soc_vector)
        mask[soc_vector < SOC_BACKUP_MIN] = float('-inf')   # SOC floor
        mask[f_p] = float('-inf')                           # cannot be own backup
        return mask   # shape (N,)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(
        self,
        H:               torch.Tensor,             # (N, D_MODEL) node embeddings from encoder
        c:               torch.Tensor,             # (D_MODEL,)   global context vector
        soc_vector:      torch.Tensor,             # (N,)         SOC_j for each node, in [0,1]
        greedy:          bool = False,             # True = argmax (deterministic); False = stochastic
        candidate_mask:  Optional[torch.Tensor] = None,  # (N,) bool — top-k admissible candidates
        heuristic_scores: Optional[torch.Tensor] = None, # (N,) float — for eval epsilon sampling
        task_r0_primary: float = 1.0,             # predicted reliability of selected primary
        task_slack_ratio: float = 1.0,            # (deadline - pred_delay) / deadline
    ) -> Tuple:
        """
        Select PRIMARY node f_p and (conditionally) BACKUP node f_b.

        Changes vs. the original version:
          1. Accepts a physics-guided candidate_mask that limits the primary action space
             to the top-K safe candidates (SOC >= floor, queue depth < MAX, deadline-feasible).
          2. At eval time (greedy=True) uses deterministic argmax unless EVAL_EPSILON is
             explicitly configured above zero.
          3. Backup parity is enabled by default: select a backup whenever a SOC-valid
             second node exists, matching FODAS's recovery-path policy.

        Returns a 6-tuple:
          (f_p, f_b, logp_p, logp_b, entropy_p, entropy_b)

          f_p, f_b   : int node indices (or _NO_NODE = -1 if infeasible / backup skipped)
          logp_p/b   : torch scalar log-probabilities (or None if infeasible)
          entropy_p/b: torch scalar distribution entropies (or None if infeasible)
        """
        N = H.shape[0]

        # -------- Step 1: Primary selection --------
        z_p      = self._primary_logits(H, c)                # (N,) raw scores
        mask_p   = self._primary_mask(soc_vector)            # (N,) 0 or -inf (SOC floor)

        # Apply physics-guided candidate mask on top of SOC mask.
        if candidate_mask is not None:
            # Exclude candidates not in the admissible top-K set.
            cand_additive = torch.zeros(N, dtype=torch.float32)
            cand_additive[~candidate_mask] = float('-inf')
            mask_p = mask_p + cand_additive

        logits_p = z_p + mask_p                              # masked logits

        # Edge case: every candidate is excluded -> task is undispatchable.
        if torch.all(torch.isinf(logits_p)):
            return _NO_NODE, _NO_NODE, None, None, None, None

        pi_p   = torch.softmax(logits_p, dim=0)             # (N,)
        dist_p = torch.distributions.Categorical(probs=pi_p)

        # Eval epsilon-greedy: EVAL_EPSILON fraction of the time, sample from top-K
        # using heuristic probabilities instead of taking the strict argmax.
        use_epsilon = (
            greedy
            and heuristic_scores is not None
            and candidate_mask is not None
            and random.random() < EVAL_EPSILON
        )
        if use_epsilon:
            # Sample among admissible candidates proportional to softmax of heuristic scores.
            h_scores_masked = heuristic_scores.clone().float()
            h_scores_masked[~candidate_mask] = float('-inf')
            pi_h = torch.softmax(h_scores_masked, dim=0)
            f_p  = int(torch.distributions.Categorical(probs=pi_h).sample().item())
        elif greedy:
            f_p  = int(logits_p.argmax().item())
        else:
            f_p  = int(dist_p.sample().item())

        logp_p = dist_p.log_prob(torch.tensor(f_p))         # scalar log-prob of selected primary
        ent_p  = dist_p.entropy()                            # scalar entropy of primary distribution

        # -------- Step 2: Conditional backup selection --------
        need_backup = (
            ALWAYS_SELECT_BACKUP
            or task_r0_primary < BACKUP_RELIABILITY_TRIGGER
            or task_slack_ratio < BACKUP_SLACK_TRIGGER_RATIO
        )

        if not need_backup:
            # Primary is strong enough — skip backup to save queue capacity and energy.
            return f_p, _NO_NODE, logp_p, None, ent_p, None

        h_fp     = H[f_p]                                    # (D_MODEL,) embedding of primary
        z_b      = self._backup_logits(H, c, h_fp)          # (N,) raw backup scores
        mask_b   = self._backup_mask(soc_vector, f_p)       # (N,) blocks f_p + low-SOC nodes
        logits_b = z_b + mask_b                              # masked logits

        # Edge case: no feasible backup available.
        if torch.all(torch.isinf(logits_b)):
            return f_p, _NO_NODE, logp_p, None, ent_p, None

        pi_b    = torch.softmax(logits_b, dim=0)             # (N,)
        dist_b  = torch.distributions.Categorical(probs=pi_b)

        f_b     = int(logits_b.argmax().item()) if greedy else int(dist_b.sample().item())
        logp_b  = dist_b.log_prob(torch.tensor(f_b))
        ent_b   = dist_b.entropy()

        return f_p, f_b, logp_p, logp_b, ent_p, ent_b

    # ------------------------------------------------------------------
    # Log-prob recomputation for PPO update
    # ------------------------------------------------------------------
    def log_prob_of(
        self,
        H:              torch.Tensor,            # (N, D_MODEL) — stored observation at action time
        c:              torch.Tensor,            # (D_MODEL,)
        soc_vector:     torch.Tensor,            # (N,) — stored SOC at action time
        f_p:            int,                     # stored primary action
        f_b:            int,                     # stored backup action (_NO_NODE if none)
        candidate_mask: Optional[torch.Tensor] = None,  # (N,) bool — stored top-K mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recompute log pi_theta(f_p, f_b | state) under the CURRENT policy weights.

        Called during PPO inner-epoch updates.  We re-apply the same masks with the
        stored soc_vector and candidate_mask so the probability ratio is computed
        under identical feasibility conditions as at collection time.

        Returns: (total_logp, total_entropy) as torch scalars.
          total_logp    = logp_p + logp_b    (or just logp_p if f_b == _NO_NODE)
          total_entropy = ent_p  + ent_b
        """
        N = H.shape[0]
        # Primary log-prob — re-apply candidate mask if provided.
        z_p      = self._primary_logits(H, c)
        mask_p   = self._primary_mask(soc_vector)
        if candidate_mask is not None:
            cand_additive = torch.zeros(N, dtype=torch.float32)
            cand_additive[~candidate_mask] = float('-inf')
            mask_p = mask_p + cand_additive
        logits_p = z_p + mask_p
        pi_p     = torch.softmax(logits_p, dim=0)
        dist_p   = torch.distributions.Categorical(probs=pi_p)
        logp_p   = dist_p.log_prob(torch.tensor(f_p))
        ent_p    = dist_p.entropy()

        if f_b == _NO_NODE:
            # No backup was selected; return primary terms only.
            return logp_p, ent_p

        # Backup log-prob
        h_fp     = H[f_p]
        z_b      = self._backup_logits(H, c, h_fp)
        logits_b = z_b + self._backup_mask(soc_vector, f_p)
        pi_b     = torch.softmax(logits_b, dim=0)
        dist_b   = torch.distributions.Categorical(probs=pi_b)
        logp_b   = dist_b.log_prob(torch.tensor(f_b))
        ent_b    = dist_b.entropy()

        return logp_p + logp_b, ent_p + ent_b   # PPO uses joint log-prob of the full action


class Critic(nn.Module):
    """
    State-value function V_phi(c) -> scalar.

    Takes the global context vector c (mean of all node embeddings) as input.
    The critic needs only the swarm-level summary c, not individual node embeddings H,
    because the value function estimates expected EPISODE return — a global quantity —
    not per-node quality.

    Architecture: MLP with 2 hidden layers, width D_MODEL, ReLU, scalar output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mlp_v = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),   # first hidden layer; input = context c (128-dim)
            nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL),   # second hidden layer
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),         # scalar value estimate V_phi
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value from global context c.

        Args:
            c: global context vector, shape (D_MODEL,) or (batch, D_MODEL)
        Returns:
            V: scalar state value, shape () or (batch,)
        """
        return self.mlp_v(c).squeeze(-1)   # remove trailing dim-1 to get scalar


# ------------------------------------------------------------------------------
# §13 — DEBUG DEMO
# ------------------------------------------------------------------------------
def _demo_actor_critic() -> None:
    """
    Run select_action on a real encoder pass; verify f_p != f_b and SOC constraints.
    Not called in normal execution; toggle the `if False` guard to run.
    """
    # Build real encoder output from current swarm state.
    x_feat = build_candidate_features(
        FOG_SWARM, TASK_STREAM[0], IOT_DEVICES[0]
    )

    encoder = AttentionEncoder()
    actor   = Actor()
    critic  = Critic()
    encoder.eval(); actor.eval(); critic.eval()

    with torch.no_grad():
        H, c = encoder(x_feat, update_stats=False)

    # Extract real SOC values from the swarm.
    soc_vec = torch.tensor([node.soc for node in FOG_SWARM], dtype=torch.float32)

    # Stochastic selection.
    with torch.no_grad():
        f_p, f_b, logp_p, logp_b, ent_p, ent_b = actor.select_action(H, c, soc_vec)

    # Critic value estimate.
    with torch.no_grad():
        V = critic(c)

    print("\n[DEMO §13] Actor / Critic — action selection on live swarm")
    print(f"  f_p (primary) : UAV#{f_p:02d}  SOC={soc_vec[f_p]:.3f}  (floor={SOC_PRIMARY_MIN})")
    if f_b != _NO_NODE:
        print(f"  f_b (backup)  : UAV#{f_b:02d}  SOC={soc_vec[f_b]:.3f}  (floor={SOC_BACKUP_MIN})")
        print(f"  f_p != f_b    : {f_p != f_b}")
        print(f"  SOC_p >= 0.30 : {soc_vec[f_p].item() >= SOC_PRIMARY_MIN}")
        print(f"  SOC_b >= 0.15 : {soc_vec[f_b].item() >= SOC_BACKUP_MIN}")
    else:
        print(f"  f_b (backup)  : NO_NODE (no feasible backup)")

    print(f"  logp_p        : {logp_p.item():.4f}  nats")
    if logp_b is not None:
        print(f"  logp_b        : {logp_b.item():.4f}  nats")
    print(f"  entropy_p     : {ent_p.item():.4f}")
    if ent_b is not None:
        print(f"  entropy_b     : {ent_b.item():.4f}")
    print(f"  V(c)          : {V.item():.4f}  (untrained; random init)")

    # Verify log_prob_of returns consistent logp under same policy.
    if f_b != _NO_NODE:
        with torch.no_grad():
            total_lp, total_ent = actor.log_prob_of(H, c, soc_vec, f_p, f_b)
        expected = (logp_p + logp_b).item()
        print(f"  log_prob_of check: {total_lp.item():.6f} == {expected:.6f} -> {abs(total_lp.item() - expected) < 1e-5}")




# ==============================================================================
# SECTION 14 — REWARD, GAE ADVANTAGE, AND PPO UPDATE  (§4.3 – §4.5)
# ==============================================================================

_REWARD_EPS: float = 1e-8   # guard for divide-by-zero in reward normalization denominators


# ------------------------------------------------------------------------------
# §4.3  Energy-consumption prediction for the reward term
# ------------------------------------------------------------------------------

def predicted_consume_energy_j(
    node:     FogNode,
    task:     Task,
    TR_bps:   float,
    TQ_avg_ms: float,
) -> float:
    """
    Predicted energy consumed by node j to serve one task, in Joules.

    Three phases:
      1. Queue wait   (node hovers + idles while task waits behind earlier tasks):
           E_queue = (P_HOVER + P_IDLE) * TQ_avg_s
      2. Execution    (node hovers + computes while running the task):
           E_exec  = (P_HOVER + P_PROC) * Te_s
      3. Communication (radio cost for receive + forward):
           E_comm  = comm_energy_j(node, task.size_kb, TR_bps)   (see §9)

    P_HOVER dominates in all three phases (870.52 W vs 10 W compute / 3 W idle)
    because the UAV must maintain altitude regardless of task state [Zeng19].
    """
    Te_s    = task.cycles / node.fr_avg_hz()           # execution time, s
    TQ_avg_s = TQ_avg_ms / 1000.0                      # queue wait, s

    E_queue = (P_HOVER + P_IDLE) * TQ_avg_s            # J — hover + idle during queue wait
    E_exec  = (P_HOVER + P_PROC) * Te_s                # J — hover + compute during execution
    E_comm  = comm_energy_j(node, task.size_kb, TR_bps) # J — radio (receive + forward)

    return E_queue + E_exec + E_comm                    # total predicted energy, J


# ------------------------------------------------------------------------------
# §4.3  Outcome-centred reward function  r_k
#
# Replaces the former delta-based reward that depended on WL/D/R deltas from the
# PREVIOUS task.  The delta formulation caused unstable gradients early in training
# (when WL_prev ~ 0, divisions blow up) and gave the agent a noisy signal that
# could not distinguish good routing from lucky task ordering.
#
# The new reward is DIRECT: it measures the per-task outcome in absolute terms
# and penalises queue damage and energy proportional to their actual magnitude.
# This makes the learning signal robust to episode position and task arrival order.
# ------------------------------------------------------------------------------

def compute_reward(
    task:         Task,
    primary_node: FogNode,
    outcome:      Dict,          # from dispatch_task: keys success, D_primary_ms, R_i
    E_consume_j:  float,         # predicted energy consumed by primary, J
    queue_delay_primary_ms: float,  # queue wait at primary before dispatch, ms
    swarm_mean_queue_depth: float,  # mean queue depth across swarm (hotspot detection)
    swarm_max_queue_depth:  float,  # max queue depth across swarm
    next_queue_delay_primary_ms: Optional[float] = None,  # queue wait after dispatch, ms
) -> Tuple[float, Dict[str, float]]:
    """
    Outcome-centred reward r_k for dispatching task k.

    Components:
      BASE:   +REWARD_SUCCESS (2.0) if delivered before deadline
            | +REWARD_MISS    (-3.0) if missed / dropped
      SLACK:  +REWARD_SLACK_W * latency_score
              latency_score = 1 - D_primary_ms/deadline, clipped to [-1, 1]
      REL:    +REWARD_RELIABILITY_W * R_i
      ENERGY: -REWARD_ENERGY_W * energy_norm
              energy_norm = E_consume_j / max(E_initial_j, 1)
      QUEUE:  REWARD_QUEUE_EXTERNALITY_W * (GAMMA*Phi(s') - Phi(s))
              Phi(s) = -queue_delay_norm. This potential-based form preserves
              the optimal policy while discouraging queue pile-ups.
      OVERLOAD: -REWARD_OVERLOAD_W * overload_indicator
              overload_indicator = 1 if primary queue is the single hotspot
                                   (swarm_max >> swarm_mean)

    HARD RTH PENALTY: overrides r_k with -CRITICAL_PENALTY if primary SOC < SOC_BACKUP_MIN.

    Returns: (r_k scalar, component_dict for printing)
    """
    D_ms        = outcome.get('D_primary_ms', task.deadline_ms)
    R_i         = outcome.get('R_i', 0.0)
    on_time     = (D_ms <= task.deadline_ms) and (outcome.get('success', 0) == 1)
    dropped     = outcome.get('dropped', False)

    # ---- Base outcome term ----
    base = REWARD_SUCCESS if (on_time and not dropped) else REWARD_MISS

    # ---- Dense normalized latency ----
    latency_score  = 1.0 - (D_ms / max(task.deadline_ms, _REWARD_EPS))
    latency_score  = float(np.clip(latency_score, -1.0, 1.0))
    contrib_slack  = REWARD_SLACK_W * latency_score

    # ---- Reliability ----
    contrib_rel    = REWARD_RELIABILITY_W * float(np.clip(R_i, 0.0, 1.0))

    # ---- Energy cost ----
    energy_norm    = float(np.clip(
        E_consume_j / max(primary_node.E_initial_j, _REWARD_EPS), 0.0, 1.0
    ))
    contrib_energy = -REWARD_ENERGY_W * energy_norm

    # ---- Potential-based queue shaping ----
    queue_norm     = float(np.clip(
        queue_delay_primary_ms / max(DEADLINE_LARGE_MS, _REWARD_EPS), 0.0, 1.0
    ))
    next_queue_norm = float(np.clip(
        (next_queue_delay_primary_ms if next_queue_delay_primary_ms is not None else queue_delay_primary_ms)
        / max(DEADLINE_LARGE_MS, _REWARD_EPS), 0.0, 1.0
    ))
    phi_s          = -queue_norm
    phi_next       = -next_queue_norm
    contrib_queue  = REWARD_QUEUE_EXTERNALITY_W * (GAMMA * phi_next - phi_s)

    # ---- Hotspot / overload penalty ----
    # If the max queue depth is >> mean (hotspot forming), penalise sending more to the overloaded node.
    overload_indicator = 0.0
    if swarm_mean_queue_depth > 0.0:
        hotspot_ratio = swarm_max_queue_depth / max(swarm_mean_queue_depth, 1.0)
        if hotspot_ratio >= 3.0:   # primary is carrying >= 3x mean load
            overload_indicator = float(np.clip((hotspot_ratio - 1.0) / 9.0, 0.0, 1.0))
    contrib_overload = -REWARD_OVERLOAD_W * overload_indicator

    r_k = base + contrib_slack + contrib_rel + contrib_energy + contrib_queue + contrib_overload

    # Hard RTH penalty: override completely if primary is below RTH floor.
    rth_violated = primary_node.soc < SOC_BACKUP_MIN
    if rth_violated:
        r_k = -CRITICAL_PENALTY

    components = {
        "base":             round(base,            4),
        "slack_contrib":    round(contrib_slack,   4),
        "rel_contrib":      round(contrib_rel,     4),
        "energy_contrib":   round(contrib_energy,  4),
        "queue_contrib":    round(contrib_queue,   4),
        "overload_contrib": round(contrib_overload, 4),
        "latency_score":    round(latency_score,   4),
        "r_k":              round(r_k,             4),
        "rth_penalty":      rth_violated,
    }
    return r_k, components


# ------------------------------------------------------------------------------
# §4.4  Generalized Advantage Estimation (GAE)  [Schul17]
# ------------------------------------------------------------------------------

def compute_gae(
    rewards:     List[float],
    values:      List[float],   # V(s_k) estimates from critic, as Python floats
    next_values: List[float],   # V(s_{k+1}); for terminal step use 0.0
    dones:       List[bool],    # True at episode boundary (next state is a new episode)
) -> Tuple[List[float], List[float]]:
    """
    Compute GAE advantages and value targets for a trajectory.

    TD residual (delta):
      delta_k = r_k + GAMMA * V(s_{k+1}) * (1 - done_k) - V(s_k)

    GAE advantage (backward recursion) [Schul17 Eq.(11)]:
      A_k = delta_k + (GAMMA * GAE_LAMBDA) * A_{k+1} * (1 - done_k)
      A_T = delta_T   (no future beyond trajectory end)

    Value target:
      y_k = A_k + V(s_k)   (advantage + baseline = MC-like return estimate)

    Advantage normalization (zero mean, unit std):
      Applied BEFORE the PPO update.  Without normalization, the magnitude of
      advantages depends on the reward scale, making the clipped-ratio bound
      PPO_CLIP=0.2 inconsistently tight or loose across training stages.
      Normalizing ensures the effective learning rate is stable [Schul17].

    Returns: (advantages list, value_targets list), both as Python floats.
    """
    T = len(rewards)
    advantages   = [0.0] * T
    value_targets = [0.0] * T

    gae = 0.0                                           # running GAE accumulator
    for k in reversed(range(T)):                        # backward pass over trajectory
        not_done = 0.0 if dones[k] else 1.0            # mask: zero out bootstrap at episode end
        delta = (rewards[k]
                 + GAMMA * next_values[k] * not_done   # bootstrapped next-state value
                 - values[k])                           # subtract current-state baseline

        gae = delta + GAMMA * GAE_LAMBDA * not_done * gae  # accumulate discounted advantage
        advantages[k]    = gae
        value_targets[k] = gae + values[k]             # target = advantage + baseline (stop-grad at train time)

    # Normalize advantages: zero mean, unit std across the trajectory.
    adv_mean = sum(advantages) / max(T, 1)
    adv_var  = sum((a - adv_mean) ** 2 for a in advantages) / max(T, 1)
    adv_std  = max(adv_var ** 0.5, _REWARD_EPS)        # clamp std to avoid /0 on constant trajectory
    advantages = [(a - adv_mean) / adv_std for a in advantages]

    return advantages, value_targets


# ------------------------------------------------------------------------------
# §4.5  PPO Update  [Schul17]
# ------------------------------------------------------------------------------

def ppo_update(
    actor:     Actor,
    critic:    Critic,
    optimizer: torch.optim.Optimizer,
    batch:     Dict,              # dict of lists collected during one episode (see below)
) -> Dict[str, float]:
    """
    Run PPO_INNER_EPOCHS passes over the stored trajectory batch and update the policy.

    Expected keys in `batch` (all lists of length T, one entry per dispatched task):
      'H'          : List[Tensor(N,D_MODEL)]  — encoder node embeddings at decision time
      'c'          : List[Tensor(D_MODEL,)]   — encoder context vectors
      'soc'        : List[Tensor(N,)]         — SOC vectors (for re-applying masks)
      'cand_masks' : List[Tensor(N,) bool]    — physics top-K candidate masks (may be all-True if not used)
      'f_p'        : List[int]                — primary action indices
      'f_b'        : List[int]                — backup action indices (_NO_NODE for drops)
      'logp_old'   : List[float]              — log-prob at collection time (no grad)
      'advantages' : List[float]              — GAE-normalized advantages (stop-grad)
      'vtargets'   : List[float]              — value regression targets (stop-grad)

    PPO clipped objective [Schul17 Eq.(7)]:
      r_t(theta) = exp(logp_new - logp_old)
      L_CLIP = mean( min(r_t * A,  clip(r_t, 1-eps, 1+eps) * A) )

    Value loss:
      L_VF = mean( (V_phi(c) - v_target)^2 )

    Entropy bonus (encourages exploration):
      H = mean( entropy_p + entropy_b )   [factorized joint policy]

    Combined loss (minimized, so policy and entropy terms are negated):
      loss = -L_CLIP + C1_VALUE * L_VF - C2_ENTROPY * H   [Schul17 Eq.(9)]

    Gradient clipping at max-norm 0.5 prevents destructively large updates,
    which can occur early in training when value estimates are poor [Schul17].

    Returns: dict with scalar losses for the episode summary printout.
    """
    T = len(batch['f_p'])                               # trajectory length (tasks in episode)

    # Pre-convert lists of scalars to tensors (no grad needed for targets).
    adv_tensor  = torch.tensor(batch['advantages'], dtype=torch.float32)  # (T,)
    vtgt_tensor = torch.tensor(batch['vtargets'],   dtype=torch.float32)  # (T,)
    logp_old    = torch.tensor(batch['logp_old'],   dtype=torch.float32)  # (T,); collected, no grad

    # Accumulators for loss reporting (averaged over inner epochs).
    total_policy_loss  = 0.0
    total_value_loss   = 0.0
    total_entropy      = 0.0

    for _epoch in range(PPO_INNER_EPOCHS):              # [Schul17]: multiple passes over same data
        policy_losses  = []
        value_losses   = []
        entropies      = []

        for k in range(T):
            H_k    = batch['H'][k]                      # (N, D_MODEL) — stored observation
            c_k    = batch['c'][k]                      # (D_MODEL,)
            soc_k  = batch['soc'][k]                    # (N,) — stored SOC for mask consistency
            f_p_k  = batch['f_p'][k]
            f_b_k  = batch['f_b'][k]
            # Retrieve stored candidate mask (None if not present in batch, e.g. old data).
            cand_masks_list = batch.get('cand_masks', None)
            cand_mask_k = cand_masks_list[k] if cand_masks_list is not None else None

            # Skip undispatchable tasks (both primary and backup were NO_NODE).
            if f_p_k == _NO_NODE:
                continue

            # Recompute log-prob and entropy under current policy weights.
            new_logp, ent = actor.log_prob_of(H_k, c_k, soc_k, f_p_k, f_b_k, cand_mask_k)

            # PPO probability ratio in log-space (avoids floating-point underflow).
            ratio = torch.exp(new_logp - logp_old[k])  # r_t(theta); close to 1.0 early in training

            A_k = adv_tensor[k]                         # normalized advantage (stop-grad)

            # Clipped surrogate objective (negated because we minimize).
            surr_unclipped = ratio * A_k
            surr_clipped   = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * A_k
            policy_loss    = -torch.min(surr_unclipped, surr_clipped)   # pessimistic bound

            # Value function regression loss.
            V_pred   = critic(c_k)                      # scalar estimate V_phi(s_k)
            vf_loss  = (V_pred - vtgt_tensor[k]) ** 2  # MSE against GAE target

            policy_losses.append(policy_loss)
            value_losses.append(vf_loss)
            entropies.append(ent)                       # factorized joint entropy

        if not policy_losses:                           # edge case: all tasks dropped this episode
            continue

        # Aggregate into scalar losses for this inner epoch.
        L_clip = torch.stack(policy_losses).mean()     # mean clipped policy loss
        L_vf   = torch.stack(value_losses).mean()      # mean value MSE
        H_ent  = torch.stack(entropies).mean()         # mean entropy (maximized via -C2 term)

        loss = L_clip + C1_VALUE * L_vf - C2_ENTROPY * H_ent   # [Schul17 Eq.(9)]

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: prevents large updates when critic is poorly calibrated early on.
        torch.nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()),
            max_norm=0.5,           # 0.5 is the standard PPO gradient clip magnitude [Schul17]
        )
        optimizer.step()

        total_policy_loss += L_clip.item()
        total_value_loss  += L_vf.item()
        total_entropy     += H_ent.item()

    n_epochs = max(PPO_INNER_EPOCHS, 1)                 # avoid /0 if all tasks dropped
    return {
        "policy_loss": total_policy_loss / n_epochs,    # mean over inner epochs
        "value_loss":  total_value_loss  / n_epochs,
        "entropy":     total_entropy     / n_epochs,
        "total_loss":  (total_policy_loss + C1_VALUE * total_value_loss
                        - C2_ENTROPY * total_entropy) / n_epochs,
    }


# ------------------------------------------------------------------------------
# §4.6  FODAS-anchored behavior-cloning warm start
# ------------------------------------------------------------------------------

def behavior_clone_fodas(
    encoder: AttentionEncoder,
    actor:   Actor,
    n_steps: int = BC_PRETRAIN_STEPS,
    batch_size: int = BC_BATCH_SIZE,
    epochs: int = BC_EPOCHS,
) -> Dict[str, float]:
    """
    Warm-start the actor by imitating the deterministic FODAS selector.

    The clone objective is deliberately narrow: cross-entropy on FODAS primary
    and backup choices under the same task-aware features and feasibility masks
    used by PPO. PPO still owns improvement beyond the FODAS anchor.
    """
    from world import build_fog_swarm, build_iot_devices, generate_task_stream, assign_mobility
    from baselines.fodas_baseline import select_node_for_task as fodas_select
    from execution_engine import BackupPlan, ExecutionEngine

    encoder.train()
    actor.train()
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(actor.parameters()),
        lr=BC_LEARNING_RATE,
    )

    losses: List[float] = []
    seen = 0

    for epoch in range(max(epochs, 1)):
        fog_nodes = build_fog_swarm()
        iot, centres = build_iot_devices()
        assign_mobility(fog_nodes, centres)
        tasks = generate_task_stream(max(n_steps, batch_size))
        execution = ExecutionEngine(fog_nodes)
        sim_time_s = 0.0

        batch_losses = []
        for step, task in enumerate(tasks[:n_steps]):
            sim_time_s = max(sim_time_s, task.arrival_s)
            execution.run(sim_time_s)
            iot_dev = iot[random.randint(0, len(iot) - 1)]

            update_memory_occupancy(fog_nodes)
            me_dict = memory_efficiency(fog_nodes)
            x_feat = build_candidate_features(fog_nodes, task, iot_dev, me_dict)
            cand_mask, _ = filter_candidates(fog_nodes, task, iot_dev)
            target_p, target_b = fodas_select(task, iot_dev, fog_nodes, sim_time_s)
            if target_p == _NO_NODE:
                continue

            H, c = encoder(x_feat, update_stats=True)
            soc_vec = torch.tensor([n.soc for n in fog_nodes], dtype=torch.float32)

            z_p = actor._primary_logits(H, c)
            mask_p = actor._primary_mask(soc_vec)
            cand_additive = torch.zeros_like(mask_p)
            cand_additive[~cand_mask] = float("-inf")
            logits_p = z_p + mask_p + cand_additive
            if torch.isinf(logits_p[target_p]):
                continue
            loss = F.cross_entropy(logits_p.unsqueeze(0), torch.tensor([target_p]))

            if target_b != _NO_NODE:
                z_b = actor._backup_logits(H, c, H[target_p])
                logits_b = z_b + actor._backup_mask(soc_vec, target_p)
                if not torch.isinf(logits_b[target_b]):
                    loss = loss + F.cross_entropy(logits_b.unsqueeze(0), torch.tensor([target_b]))

            batch_losses.append(loss)
            seen += 1

            backup_plan = (
                BackupPlan(target_b, 0.0, 0.0)
                if target_b != _NO_NODE else None)
            execution.submit_task(
                task_id=task.task_id,
                arrival_s=task.arrival_s,
                deadline_s=task.arrival_s + task.deadline_ms / 1000.0,
                cycles=task.cycles,
                size_kb=task.size_kb,
                primary_node_id=target_p,
                backup=backup_plan,
            )

            if len(batch_losses) >= batch_size:
                total = torch.stack(batch_losses).mean()
                opt.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(actor.parameters()),
                    max_norm=0.5,
                )
                opt.step()
                losses.append(float(total.item()))
                batch_losses = []

        if batch_losses:
            total = torch.stack(batch_losses).mean()
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(actor.parameters()),
                max_norm=0.5,
            )
            opt.step()
            losses.append(float(total.item()))

        if tasks and n_steps > 0:
            execution.drain_relevant(
                tasks[min(n_steps, len(tasks)) - 1].arrival_s)

    return {
        "bc_loss": float(np.mean(losses)) if losses else 0.0,
        "bc_batches": float(len(losses)),
        "bc_samples": float(seen),
    }


# ==============================================================================
