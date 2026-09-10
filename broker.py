"""
broker.py
=========
Stage 1 of the control loop: the Master Fog Head election.

BrokerSelector runs the four-step rank-reversal-free MCDM pipeline once per
control tick — measurement smoothing -> MEREC weights -> KL-divergence gate ->
SPOTIS ranking — to elect the broker UAV. The KL gate suppresses the expensive
SPOTIS re-rank on ticks where the MEREC weight vector barely moved.

The trailing _demo_broker_selector is an illustrative, never-called example.
"""

import math
import time
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import *
from models import Task, FogNode, IoTDevice
from world import FOG_SWARM, IOT_DEVICES, TASK_STREAM
from physics import *

# SECTION 11 — STAGE 1: BROKER SELECTOR  (§3 MCDM pipeline)
#
# Pipeline per control tick:
#   build_decision_matrix  ->  kalman_smooth  ->  merec_weights
#   ->  kl_gate  ->  spotis_select (conditional)
#
# Criteria order: [E_res(+), ME(+), R(+), D_ctrl(-), WL(-), ECT(+)]
# (+) = beneficial (higher is better); (-) = non-beneficial (lower is better)
# ME (memory efficiency) replaces the former PC (processing capability) at index 1.
# ==============================================================================

_MEREC_EPS: float = 1e-9   # prevents log(0) in MEREC normalization; negligibly small vs real values


def residual_coordination_time_s(
    node: FogNode, fog_nodes: List[FogNode],
    comm_range_m: float = HEAD_COMM_RANGE_M,
    horizon_s: float = HEAD_AVAILABILITY_HORIZON_S,
) -> float:
    """10th-percentile constant-velocity link-expiration time to peers."""
    expiries = []
    for peer in fog_nodes:
        if peer.node_id == node.node_id:
            continue
        rx, ry, rz = peer.x - node.x, peer.y - node.y, peer.z - node.z
        vx = getattr(peer, "vx_ms", 0.0) - getattr(node, "vx_ms", 0.0)
        vy = getattr(peer, "vy_ms", 0.0) - getattr(node, "vy_ms", 0.0)
        c = rx * rx + ry * ry + rz * rz - comm_range_m * comm_range_m
        if c >= 0.0:
            expiries.append(0.0)
            continue
        a = vx * vx + vy * vy
        if a <= _MEREC_EPS:
            expiries.append(horizon_s)
            continue
        b = 2.0 * (rx * vx + ry * vy)
        discriminant = max(0.0, b * b - 4.0 * a * c)
        expiry = (-b + math.sqrt(discriminant)) / (2.0 * a)
        expiries.append(float(np.clip(expiry, 0.0, horizon_s)))
    return float(np.percentile(expiries, 10)) if expiries else horizon_s


def energy_coverage_time_s(node: FogNode) -> float:
    """Coordinator endurance under flight and nominal gated Stage-1 duty."""
    gated_cycles = CONTROLLER_REFERENCE_HZ * (
        T_MEREC_MS + T_KALMAN_MS + T_KL_MS) / 1000.0
    duty = min(gated_cycles / node.fr_avg_hz() / CONTROL_TICK_S, 1.0)
    return node.E_res_j / max(aero_power_w(node.speed_ms) + P_PROC * duty,
                              _MEREC_EPS)


def handover_broadcast_airtime_s(head: FogNode, fog_nodes: List[FogNode]) -> float:
    """Sequential state-sync airtime from a new head to all peers."""
    peers = [node for node in fog_nodes if node.node_id != head.node_id]
    if not peers:
        return 0.0
    mean_distance = float(np.mean([
        uav_uav_distance_m(head, node) for node in peers]))
    return (SIGNAL_KB * 1024.0 * 8.0 / data_rate_bps(
        mean_distance, P_UAV_TX_W)) * len(peers)


class BrokerSelector:
    """
    Persistent per-episode object that runs the four-step MCDM broker election.

    State carried across ticks:
      w_prev  — last MEREC weight vector (N_CRITERIA-dim)
      prev_head          — FogNode elected as broker at last SPOTIS run
      prev_spotis_ranking — full sorted list from last SPOTIS run (failover order)
    """

    def __init__(self, final_fix: bool = False) -> None:
        self.final_fix = final_fix
        self.n_criteria = N_CRITERIA if final_fix else 5
        # Initial weight vector: equal weights (MEREC will differentiate them each tick).
        self.w_prev: np.ndarray = np.array(
            MCDM_INIT_WEIGHTS if final_fix else [0.2] * 5, dtype=float)

        # Initial Kalman error covariance: uninformed prior, P0 * I5.
        self.P: np.ndarray = KALMAN_P0 * np.eye(self.n_criteria)
        self.measurement_state: Dict[int, np.ndarray] = {}

        self.prev_head: Optional[FogNode]          = None   # no broker elected yet
        self.prev_spotis_ranking: Optional[List]   = None   # no ranking yet
        self.head_tenure: int                      = 0      # ticks since current head was elected
        self.head_score:  float                    = float('inf')  # SPOTIS distance of current head (lower = better)
        self.ticks_since_ranking: int              = 0      # bounds stale rankings under stable weights

        # Fixed SPOTIS bounds from CONFIG — set a priori, never recomputed from the
        # live candidate set.  This is the core requirement of [Dez20 §III Step 1]:
        # bounds must be independent of which alternatives are currently present so
        # that adding/removing a UAV cannot change the relative ordering of others.
        self.S_min: np.ndarray = np.array(
            SPOTIS_S_MIN if final_fix else LEGACY_SPOTIS_S_MIN, dtype=float)
        self.S_max: np.ndarray = np.array(
            SPOTIS_S_MAX if final_fix else LEGACY_SPOTIS_S_MAX, dtype=float)

        # ISP_j = S_max_j for beneficial criteria, S_min_j for non-beneficial [Dez20 §III.A].
        # Beneficial  (E_res, ME, R):  ideal = maximum possible value  => S_max_j
        # Non-beneficial (D, WL):      ideal = minimum possible value  => S_min_j = 0
        self.ISP: np.ndarray = np.where(
            np.array(CRITERIA_BENEFICIAL[:self.n_criteria]),
            self.S_max,   # beneficial: ideal is the upper bound
            self.S_min,   # non-beneficial: ideal is the lower bound (0)
        ).astype(float)

        # Pre-compute range for the SPOTIS distance denominator; guard against /0.
        # S_range_j = S_max_j - S_min_j; all CONFIG bounds have S_max > S_min by construction.
        self.S_range: np.ndarray = np.maximum(self.S_max - self.S_min, _MEREC_EPS)
        self.max_observed_spread = np.zeros(self.n_criteria, dtype=float)

    # ------------------------------------------------------------------
    # (1)  Decision matrix
    # ------------------------------------------------------------------
    def build_decision_matrix(
        self,
        fog_nodes: List[FogNode],
        per_node_metrics: Dict[int, Dict],
    ) -> np.ndarray:
        """
        Assemble the raw (unnormalized) N×6 decision matrix.

        Columns (criteria index matches CRITERIA_BENEFICIAL order everywhere):
          col 0: E_res_j  — residual battery energy, J            (+) higher = more endurance
          col 1: ME_j     — memory efficiency [0,1]                (+) higher = more free memory
          col 2: R_j      — representative task reliability [0,1]  (+) higher = more reliable
          col 3: D_j      — projected controller service, ms        (-) lower  = faster
          col 4: WL_j     — pending CPU workload, cycles            (-) lower  = less loaded
          col 5: ECT_j    — energy coverage time, s                 (+) higher = more endurance

        per_node_metrics: dict mapping node_id ->
            {'me': float, 'r0': float, 'd_ms': float, 'wl': float}
        The caller (main loop) is responsible for computing these per tick.
        """
        matrix = np.zeros((len(fog_nodes), self.n_criteria), dtype=float)

        for i, node in enumerate(fog_nodes):
            m = per_node_metrics[node.node_id]
            matrix[i, 0] = node.E_res_j    # J   — live from FogNode state
            matrix[i, 1] = m['me']         # [0,1] — from memory_efficiency()
            matrix[i, 2] = m['r0']         # [0,1] — from base_reliability(), nominal task
            matrix[i, 3] = m['d_ms']       # ms  — from total_delay_ms(), nominal task
            matrix[i, 4] = m['wl']         # cycles — from node_workload_cycles()
            if self.final_fix:
                matrix[i, 5] = energy_coverage_time_s(node)

        return matrix

    # ------------------------------------------------------------------
    # Internal: paper-exact SPOTIS distance  [Dez20 Eq. 6 + Step 4]
    # ------------------------------------------------------------------
    def _spotis_distance(self, matrix: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Paper-exact SPOTIS score per alternative (Dezert et al. 2020, Eq. 6):

          d_i = sum_j  w_j * |S_ij - ISP_j| / |S_max_j - S_min_j|

        Bounds (S_min, S_max) and ISP are FIXED (set in __init__), so the score
        of any node is independent of the other candidates — rank-reversal-free.
        Lower d_i = closer to ideal = better broker.

        Operates on the RAW decision matrix (real units: J, ms, cycles …).
        No direction-flipping needed: ISP already encodes direction —
          beneficial criteria: ISP_j = S_max_j  => closer raw value = smaller |·|
          non-beneficial:      ISP_j = S_min_j  => closer raw value = smaller |·|
        """
        clipped   = np.clip(matrix, self.S_min, self.S_max)          # keep scores in-bounds
        norm_dist = np.abs(clipped - self.ISP[None, :]) / self.S_range[None, :]  # shape (N, 5)
        return (norm_dist * w[None, :]).sum(axis=1)                   # shape (N,)

    # ------------------------------------------------------------------
    # Internal: fixed-bound distance-to-ideal normalization in [0,1]
    # ------------------------------------------------------------------
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        MEREC and SPOTIS must agree about scale.  This returns the exact
        candidate-independent normalized distance that SPOTIS scores.  A
        criterion tied across candidates is neutralized to 1 so it cannot
        receive removal-effect weight while contributing no discrimination.
        """
        if not self.final_fix:
            norm = np.zeros_like(matrix, dtype=float)
            for j in range(self.n_criteria):
                col = matrix[:, j]
                col_min, col_max = float(col.min()), float(col.max())
                if col_max - col_min < _MEREC_EPS:
                    norm[:, j] = 0.5
                elif CRITERIA_BENEFICIAL[j]:
                    norm[:, j] = col_min / np.maximum(col, _MEREC_EPS)
                else:
                    norm[:, j] = col / max(col_max, _MEREC_EPS)
            return np.clip(norm, _MEREC_EPS, 1.0)
        clipped = np.clip(matrix, self.S_min, self.S_max)
        norm = np.abs(clipped - self.ISP[None, :]) / self.S_range[None, :]
        spreads = np.ptp(norm, axis=0)
        norm[:, spreads < CRITERION_SPREAD_FLOOR] = 1.0
        return np.clip(norm, _MEREC_EPS, 1.0)

    def decision_power(self, matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Share of live weighted discrimination carried by each criterion."""
        clipped = np.clip(matrix, self.S_min, self.S_max)
        spread = np.ptp(
            np.abs(clipped - self.ISP[None, :]) / self.S_range[None, :],
            axis=0)
        power = spread * weights
        total = float(power.sum())
        return power / total if total > _MEREC_EPS else np.zeros_like(power)

    def assert_live_criteria(self) -> None:
        """Fail a completed operating trace if any configured criterion stayed dead."""
        dead = np.flatnonzero(
            self.max_observed_spread < CRITERION_SPREAD_FLOOR)
        if dead.size:
            raise AssertionError(
                f"dead SPOTIS criteria {dead.tolist()}; max fixed-bound spreads "
                f"were {self.max_observed_spread.tolist()}")

    # ------------------------------------------------------------------
    # (2)  MEREC weight derivation  [Kesh21]
    # ------------------------------------------------------------------
    def merec_weights(self, matrix: np.ndarray) -> np.ndarray:
        """
        Derive objective criteria weights via MEREC (Method based on the Removal Effects
        of Criteria) [Kesh21].

        Intuition: a criterion whose removal causes a large change in node scores carries
        more discriminating information and therefore deserves a higher weight.

        Algorithm:
          1. Normalize matrix using MEREC ratio-based normalization [Kesh21 Eq. (2)]:
               all criteria converted to minimization type, values in [EPS, 1].
          2. Overall performance of each node [Kesh21 Eq. (3)]:
               S_x = ln( 1 + (1/m) * sum_j |ln(n_xj)| )
          3. Removal effect of criterion j [Kesh21 Eq. (4)]:
               S'_x(j) = ln( 1 + (1/m) * sum_{k≠j} |ln(n_xk)| )
               Note: paper uses 1/m (not 1/(m-1)) in the removal step.
          4. Absolute deviations [Kesh21 Eq. (5)]:
               E_j  = sum_x |S'_x(j) - S_x|
          5. Normalize [Kesh21 Eq. (6)]: z_j = E_j / sum_c E_c  (sums to 1)

        Returns: z, shape (5,), summing to 1.
        """
        m = self.n_criteria
        norm = self._normalize_matrix(matrix)              # shape (N, 5), values in [EPS, 1]

        log_abs = np.abs(np.log(norm))                     # |ln(n_xj)|, shape (N, 5)

        # Overall performance score per node (all criteria included).
        S = np.log(1.0 + (1.0 / m) * log_abs.sum(axis=1))  # shape (N,)

        # Removal effect: recompute S leaving out one criterion at a time.
        delta = np.zeros(m)
        for j in range(m):
            kept_cols = [k for k in range(m) if k != j]            # indices of surviving criteria
            log_reduced = log_abs[:, kept_cols]                     # shape (N, m-1)
            S_prime = np.log(1.0 + (1.0 / m) * log_reduced.sum(axis=1))  # [Kesh21 Eq. (4)]; shape (N,)
            delta[j] = float(np.abs(S_prime - S).sum())            # total score change from removing j

        # Normalize delta to obtain weights.
        total = delta.sum()
        if total < _MEREC_EPS:
            z = np.ones(m) / m    # degenerate: all criteria equally uninformative -> uniform
        else:
            z = delta / total     # z sums to 1

        return z   # shape (5,)

    # ------------------------------------------------------------------
    # (3)  Kalman smoothing  [Kal60]
    # ------------------------------------------------------------------
    def kalman_smooth(
        self, matrix: np.ndarray, node_ids: Optional[Iterable[int]] = None,
    ) -> np.ndarray:
        """
        Smooth each node's raw criterion observations independently.

        Identity dynamics and diagonal noise make the update separable, so the
        steady-state Kalman filter is the first-order update ``x += k*(z-x)``,
        where ``x=(Q+sqrt(Q**2+4*Q*R))/2`` and ``k=x/(x+R)``.  A full N×7×7
        covariance contains no additional information.  One gain in raw units
        is equivalent to filtering the fixed-bound-normalized matrix because
        the process noise scales with each criterion's fixed range.

        New nodes initialize from their first observation; departed-node state
        is discarded.  The legacy five-criterion path retains its weight-level
        Kalman filter and accepts a single weight vector.
        """
        if self.final_fix:
            observations = np.asarray(matrix, dtype=float)
            if observations.ndim != 2 or observations.shape[1] != self.n_criteria:
                raise ValueError(
                    f"expected an N×{self.n_criteria} decision matrix")
            observations = np.clip(observations, self.S_min, self.S_max)
            ids = list(range(len(observations))) if node_ids is None else list(node_ids)
            if len(ids) != len(observations) or len(set(ids)) != len(ids):
                raise ValueError("node_ids must uniquely match the matrix rows")
            active = set(ids)
            self.measurement_state = {
                node_id: state for node_id, state in self.measurement_state.items()
                if node_id in active}
            q, r = KALMAN_M_Q, KALMAN_M_R
            x = (q + math.sqrt(q * q + 4.0 * q * r)) / 2.0
            gain = x / (x + r)
            filtered = observations.copy()
            for row, node_id in enumerate(ids):
                previous = self.measurement_state.get(node_id)
                if previous is not None:
                    filtered[row] = previous + gain * (observations[row] - previous)
                self.measurement_state[node_id] = filtered[row].copy()
            return filtered

        z = np.asarray(matrix, dtype=float)
        I5 = np.eye(self.n_criteria)

        # --- Predict ---
        w_pred = self.w_prev.copy()                            # state extrapolation (identity)
        P_pred = self.P + KALMAN_Q * I5                        # covariance grows with process noise

        # --- Update ---
        S_innov = P_pred + KALMAN_R * I5                       # innovation covariance
        K = P_pred @ np.linalg.inv(S_innov)                    # Kalman gain: 5×5 matrix
        innov = z - w_pred                                     # innovation: MEREC obs minus prediction
        w = w_pred + K @ innov                                 # posterior weight estimate
        self.P = (I5 - K) @ P_pred                             # posterior covariance

        # Renormalize: Kalman update can nudge w slightly off the probability simplex.
        w = np.clip(w, 0.0, None)                              # no negative weights
        w_sum = w.sum()
        w = (w / w_sum if w_sum > _MEREC_EPS
             else np.ones(self.n_criteria) / self.n_criteria)

        self.w_prev = w.copy()   # persist for next tick's predict step
        return w                 # shape (5,), sums to 1

    # ------------------------------------------------------------------
    # (4)  KL divergence gate
    # ------------------------------------------------------------------
    def kl_gate(self, w_new: np.ndarray, w_old: np.ndarray) -> Tuple[bool, float]:
        """
        Decide whether to re-run (expensive) SPOTIS this tick.

          KL(w_new || w_old) = sum_j w_new_j * ln(w_new_j / w_old_j)

        If KL > KL_THRESHOLD (0.05 nats): weights changed significantly => re-rank.
        Otherwise: weights are stable => reuse previous SPOTIS ranking (save CPU).

        Tiny values are clipped to _MEREC_EPS before log to prevent -inf.
        Returns: (run_spotis: bool, kl_value: float).
        """
        w_n = np.clip(w_new, _MEREC_EPS, None)   # guard log(0) in numerator
        w_o = np.clip(w_old, _MEREC_EPS, None)   # guard log(0) in denominator
        kl = float(np.sum(w_n * np.log(w_n / w_o)))   # KL divergence, nats; always >= 0
        return kl > KL_THRESHOLD, kl

    # ------------------------------------------------------------------
    # (5)  SPOTIS ranking  [Dez20]
    # SPOTIS scores each node by its weighted normalized distance to a FIXED
    # ideal solution profile; bounds are external and alternative-independent
    # (Dezert et al. 2020), which is what guarantees rank-reversal freedom.
    # ------------------------------------------------------------------
    def spotis_select(
        self,
        fog_nodes: List[FogNode],
        matrix: np.ndarray,
        w: np.ndarray,
    ) -> Tuple[FogNode, List[FogNode]]:
        """
        Rank fog nodes using SPOTIS (Stable Preference Ordering Towards Ideal Solution) [Dez20].

        SPOTIS avoids rank-reversal by measuring each alternative's distance to a fixed
        Ideal Solution Profile (ISP) using fixed a priori bounds [Dez20 §III]:

          d_i = sum_j  w_j * |S_ij - ISP_j| / (S_max_j - S_min_j)

        Both ISP and [S_min, S_max] are set once in __init__ from CONFIG constants and
        never recomputed from the live candidate set.  This is what makes SPOTIS RRF:
        adding or removing a UAV cannot shift bounds and thereby reorder survivors.

        Operates on the raw decision matrix (real units: J, ms, cycles).
        Direction is encoded in ISP (beneficial => ISP = S_max; non-beneficial => ISP = S_min),
        so no separate normalization or direction-flip is needed here.

        Lower d_i = closer to ideal = better broker candidate.
        Returns: (head_node, ranked_list_of_FogNodes) ascending by d_i.
        """
        distances    = self._spotis_distance(matrix, w)   # fixed-bound, RRF score; shape (N,)
        ranked_idx   = list(np.argsort(distances))        # ascending: index 0 is best
        ranked_nodes = [fog_nodes[i] for i in ranked_idx]
        head_node    = ranked_nodes[0]
        return head_node, ranked_nodes

    # ------------------------------------------------------------------
    # (6)  Top-level tick entry point
    # ------------------------------------------------------------------
    def select_head(
        self,
        fog_nodes: List[FogNode],
        per_node_metrics: Dict[int, Dict],
        episode: int,
        tick: int,
    ) -> Dict:
        """
        Run the full Stage-1 pipeline for one control tick.

        Steps executed every tick: build matrix, measurement Kalman, MEREC.
        Step executed conditionally: SPOTIS (only when KL gate fires OR first tick).

        A one-tick safety floor prevents same-tick flapping. Thereafter a switch
        is allowed only when projected control-time savings over the candidate's
        residual coordination horizon exceed handover setup and broadcast time.

        Re-ranking is event-triggered by weight-vector KL or an unsafe current
        head.  Filtering the candidate measurements keeps transient queue state
        from bypassing that gate.

        If SPOTIS is skipped, the previous head and ranking are reused unchanged —
        this is the core efficiency gain of the KL gate: broker re-election costs O(N log N)
        in SPOTIS but is paid only when weights shift meaningfully.

        Returns a dict with all values the main loop needs to print:
          z, w        — the same MEREC weight vector (both keys retained for
                        result-schema stability), shape (N_CRITERIA,)
          kl_value    — KL divergence (nats) between w and the prior tick's w
          run_spotis  — bool: True if SPOTIS ran this tick
          head_id     — node_id of elected broker
          head_node   — FogNode object of elected broker
          ranking     — List[FogNode] in failover order (best first)
          head_changed— True if SPOTIS ran AND the elected head differs from the previous head
          candidate_head_id — raw SPOTIS winner before hysteresis
          switch_block_reason — kl_gate / minimum_tenure / switch_cost,
                                or empty when no switch was blocked
        """
        candidates = [node for node in fog_nodes if node.available]
        if not candidates:
            raise ValueError("no available fog-head candidates")

        # Filter noisy per-node measurements before deriving MEREC weights.
        matrix = self.build_decision_matrix(candidates, per_node_metrics)
        if self.final_fix:
            matrix = self.kalman_smooth(
                matrix, [node.node_id for node in candidates])
        fixed_norm = np.abs(
            np.clip(matrix, self.S_min, self.S_max) - self.ISP[None, :]
        ) / self.S_range[None, :]
        self.max_observed_spread = np.maximum(
            self.max_observed_spread, np.ptp(fixed_norm, axis=0))

        z = self.merec_weights(matrix)
        w_old = self.w_prev.copy()
        w = z.copy() if self.final_fix else self.kalman_smooth(z)
        if self.final_fix:
            self.w_prev = w.copy()

        kl_trigger, kl_value = self.kl_gate(w, w_old)

        state_trigger = False
        safety_trigger = False
        if self.prev_head is not None:
            safety_trigger = (not self.prev_head.available
                              or self.prev_head.soc < SOC_PRIMARY_MIN)
            if not self.final_fix and not safety_trigger:
                live_scores = self._spotis_distance(matrix, w)
                live_best_idx = int(np.argmin(live_scores))
                current_idx = next(
                    i for i, node in enumerate(candidates)
                    if node.node_id == self.prev_head.node_id)
                improvement = float(
                    live_scores[current_idx] - live_scores[live_best_idx])
                state_trigger = (
                    candidates[live_best_idx].node_id != self.prev_head.node_id
                    and improvement >= LEGACY_HEAD_SWITCH_MARGIN)

        periodic_trigger = (
            not self.final_fix
            and self.ticks_since_ranking >= HEAD_REEVALUATION_TICKS)
        run_spotis = kl_trigger or safety_trigger
        if not self.final_fix:
            run_spotis = run_spotis or state_trigger or periodic_trigger

        # Always run SPOTIS on the very first tick (no previous result to reuse).
        if self.prev_head is None:
            run_spotis = True

        prev_head_id = self.prev_head.node_id if self.prev_head is not None else None
        candidate_head_id = prev_head_id
        switch_block_reason = "kl_gate" if not run_spotis else ""

        if run_spotis:
            self.ticks_since_ranking = 0
            head_node, ranking = self.spotis_select(candidates, matrix, w)
            candidate_head_id  = head_node.node_id

            # ---- Hysteresis check: should we actually switch? ----
            switch_permitted = True

            if prev_head_id is not None and head_node.node_id != prev_head_id:
                # Check if the current head is still safe (SOC >= primary floor).
                cur_head_idx    = next(
                    (i for i, n in enumerate(candidates)
                     if n.node_id == prev_head_id), None
                )
                cur_head_safe = (
                    cur_head_idx is not None
                    and candidates[cur_head_idx].available
                    and candidates[cur_head_idx].soc >= SOC_PRIMARY_MIN
                )

                # If current head is unsafe, always allow switch (safety override).
                if not cur_head_safe:
                    switch_permitted = True
                else:
                    # Tenure check: head must have served minimum ticks.
                    tenure_floor = (HEAD_MIN_TENURE_TICKS if self.final_fix
                                    else LEGACY_HEAD_MIN_TENURE_TICKS)
                    if self.head_tenure < tenure_floor:
                        switch_permitted = False
                        switch_block_reason = "minimum_tenure"

                    # Lookahead: switch only when its integrated control-time
                    # benefit exceeds setup plus state-sync airtime.
                    if switch_permitted and self.final_fix:
                        new_idx = candidates.index(head_node)
                        if not self._switch_pays(
                            candidates, matrix, cur_head_idx, new_idx
                        ):
                            switch_permitted = False
                            switch_block_reason = "switch_cost"
                    elif switch_permitted:
                        cur_score = float(self._spotis_distance(
                            matrix, w)[cur_head_idx])
                        if cur_score - float(self._spotis_distance(
                            matrix, w)[candidates.index(head_node)]) < LEGACY_HEAD_SWITCH_MARGIN:
                            switch_permitted = False
                            switch_block_reason = "switch_margin"

            if switch_permitted:
                # Execute the switch (or first-time election).
                self.prev_head           = head_node
                self.prev_spotis_ranking = ranking
                self.head_score          = float(
                    self._spotis_distance(matrix, w)[candidates.index(head_node)]
                )
                if prev_head_id is not None and head_node.node_id != prev_head_id:
                    self.head_tenure = 0   # reset tenure on actual switch
                else:
                    self.head_tenure += 1  # re-elected same node counts as tenure
            else:
                # Hysteresis blocked the switch — keep the previous head.
                head_node = self.prev_head
                ranking   = self.prev_spotis_ranking
                self.head_tenure += 1
        else:
            # KL gate: weights stable enough — no re-ranking needed this tick.
            head_node = self.prev_head
            ranking   = self.prev_spotis_ranking
            self.head_tenure += 1

        ranking = [node for node in ranking if node.available]
        self.ticks_since_ranking += 1

        # head_changed: True only when SPOTIS ran AND the winner differs from last time.
        head_changed = (
            run_spotis
            and prev_head_id is not None
            and head_node.node_id != prev_head_id
        )

        return {
            'z':            z,
            'w':            w,
            'kl_value':     kl_value,
            'run_spotis':   run_spotis,
            'head_id':      head_node.node_id,
            'head_node':    head_node,
            'ranking':      ranking,
            'head_changed': head_changed,
            'candidate_head_id': candidate_head_id,
            'switch_block_reason': switch_block_reason,
            'kl_trigger':     kl_trigger,
            'state_trigger':  state_trigger,
            'periodic_trigger': periodic_trigger,
            'decision_power': self.decision_power(matrix, w),
            'criterion_spread': np.ptp(fixed_norm, axis=0),
        }

    @staticmethod
    def _switch_pays(
        fog_nodes: List[FogNode], matrix: np.ndarray,
        current_idx: int, candidate_idx: int,
    ) -> bool:
        ticks_per_s = 1.0 / CONTROL_TICK_S
        horizon_s = HEAD_AVAILABILITY_HORIZON_S
        benefit_s = max(
            0.0, (float(matrix[current_idx, 3])
                  - float(matrix[candidate_idx, 3])) / 1000.0
        ) * ticks_per_s * horizon_s
        candidate = fog_nodes[candidate_idx]
        cost_s = (HANDOVER_SETUP_MS / 1000.0
                  + handover_broadcast_airtime_s(candidate, fog_nodes))
        return benefit_s > cost_s


# ------------------------------------------------------------------------------
# §11 — DEBUG DEMO
# ------------------------------------------------------------------------------
def _demo_broker_selector() -> None:
    """
    Run one tick of the full Stage-1 pipeline on the live FOG_SWARM.
    Uses a nominal 200 KB task at each node's mean distance to IoT devices.
    Not called in normal execution; toggle the `if False` guard to run.
    """
    # --- Build per_node_metrics for every fog node ---
    # Use nominal values: 200 KB task, each node's Euclidean mean distance to all IoT devices.
    nominal_size_kb = 200.0
    nominal_cycles  = nominal_size_kb * 1024 * CYCLES_PER_BYTE

    update_memory_occupancy(FOG_SWARM)                  # set occupancy from queues (empty at startup)
    me_dict = memory_efficiency(FOG_SWARM)
    wl_dict, _ = system_workload_imbalance(FOG_SWARM)   # all zeros at startup

    per_node_metrics: Dict[int, Dict] = {}
    for node in FOG_SWARM:
        # Mean Euclidean distance from this node to all IoT devices.
        mean_dist = float(np.mean([
            math.hypot(node.x - d.x, node.y - d.y) for d in IOT_DEVICES
        ]))
        mean_dist = max(mean_dist, 1.0)   # enforce 1 m minimum

        # Use population-mean transmit power (weighted by group size) as representative.
        mean_p_tx = (N_LOW_POWER_IOT * P_TX_LOW_W + N_HIGH_POWER_IOT * P_TX_HIGH_W) / N_IOT

        TR_bps = data_rate_bps(mean_dist, mean_p_tx)
        r0     = base_reliability(node, nominal_cycles, nominal_size_kb, TR_bps)
        d_ms   = total_delay_ms(
            task      = TASK_STREAM[0],   # first task as a representative sample
            node      = node,
            dist_m    = mean_dist,
            p_tx_w    = mean_p_tx,
            spotis_ran= True,
        )['total_ms']

        per_node_metrics[node.node_id] = {
            'me':   me_dict[node.node_id],
            'r0':   r0,
            'd_ms': d_ms,
            'wl':   wl_dict[node.node_id],
        }

    # --- Run one tick of the selector ---
    selector = BrokerSelector()
    result   = selector.select_head(FOG_SWARM, per_node_metrics, episode=0, tick=0)

    head = result['head_node']
    z    = result['z']
    w    = result['w']

    print("\n[DEMO §11] Stage-1 broker selection — tick 0")
    print(f"  MEREC z  : [{', '.join(f'{v:.4f}' for v in z)}]  (sums to {z.sum():.4f})")
    print(f"  Kalman w : [{', '.join(f'{v:.4f}' for v in w)}]  (sums to {w.sum():.4f})")
    print(f"  KL       : {result['kl_value']:.6f} nats  ->  run_SPOTIS={result['run_spotis']}")
    print(f"  Head     : UAV#{head.node_id:02d} Tier-{head.tier} "
          f"fr={head.fr_avg_ghz:.2f} GHz  SOC={head.soc:.3f}")
    print(f"  Failover : {[n.node_id for n in result['ranking']]}")
    print(f"  Head changed from previous: {result['head_changed']}")
